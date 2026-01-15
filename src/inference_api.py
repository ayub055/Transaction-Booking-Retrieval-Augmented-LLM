"""
Inference API for Transaction Tagging with Similar Transaction Retrieval

This module provides a clean API for:
1. Predicting categories for new transactions
2. Retrieving similar historical transactions
3. Loading and managing trained models and indices

Usage:
    from src.inference_api import TransactionPredictor

    predictor = TransactionPredictor(
        artifacts_path="training_artifacts/training_artifacts.pkl",
        model_path="experiments/model/fusion_encoder_best.pth",
        faiss_index_path="transaction_index.faiss"
    )

    new_transaction = {
        'tran_partclr': 'GROCERY PURCHASE',
        'tran_mode': 'POS',
        'dr_cr_indctor': 'D',
        'sal_flag': 'N',
        'tran_amt_in_ac': 50.00
    }

    result = predictor.predict(new_transaction, top_k=5)
    print(result['predicted_category'])
    for similar_txn in result['similar_transactions']:
        print(similar_txn)
"""

import torch
import pandas as pd
import faiss
import numpy as np
import pickle
import os
from transformers import BertTokenizer
from typing import Dict, List, Optional
from src.fusion_encoder import FusionEncoder


class TransactionPredictor:
    """
    A class for predicting transaction categories and retrieving similar transactions.
    """

    def __init__(
        self,
        artifacts_path: str,
        model_path: str,
        faiss_index_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize the predictor with trained artifacts.

        Args:
            artifacts_path: Path to training artifacts pickle file
            model_path: Path to trained model checkpoint
            faiss_index_path: Path to FAISS index file
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.artifacts_path = artifacts_path
        self.model_path = model_path
        self.faiss_index_path = faiss_index_path
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Load all components
        self._load_artifacts()
        self._load_model()
        self._load_index()
        self._load_metadata()

    def _load_artifacts(self):
        """Load training artifacts (vocabularies, scalers, configs)."""
        print(f"Loading training artifacts from {self.artifacts_path}...")
        with open(self.artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.config = artifacts['config']
        self.categorical_cols = self.config['categorical_cols']
        self.numeric_cols = self.config['numeric_cols']
        self.label_col = self.config['label_col']
        self.cat_vocab = artifacts['cat_vocab']
        self.scaler = artifacts['scaler']
        self.label_mapping = artifacts['label_mapping']
        self.categorical_dims = artifacts['categorical_dims']

        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['bert_model'])

        print(f"  ✓ Loaded {len(self.label_mapping)} categories")
        print(f"  ✓ Categorical features: {len(self.categorical_cols)}")
        print(f"  ✓ Numeric features: {len(self.numeric_cols)}")

    def _load_model(self):
        """Load the trained fusion encoder model."""
        print(f"Loading model from {self.model_path}...")

        self.encoder = FusionEncoder(
            bert_model_name=self.config['bert_model'],
            categorical_dims=self.categorical_dims,
            numeric_dim=len(self.numeric_cols),
            text_proj_dim=self.config['text_proj_dim'],
            final_dim=self.config['final_dim'],
            p=self.config['dropout']
        )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.encoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.encoder.load_state_dict(checkpoint)
            print(f"  ✓ Loaded model state dict")

        self.encoder.to(self.device)
        self.encoder.eval()

    def _load_index(self):
        """Load FAISS index for similarity search."""
        print(f"Loading FAISS index from {self.faiss_index_path}...")
        self.index = faiss.read_index(self.faiss_index_path)
        print(f"  ✓ Index size: {self.index.ntotal} transactions")

    def _load_metadata(self):
        """Load transaction metadata for retrieval."""
        metadata_path = self.faiss_index_path.replace('.faiss', '_metadata.pkl')

        if os.path.exists(metadata_path):
            print(f"Loading transaction metadata from {metadata_path}...")
            with open(metadata_path, 'rb') as f:
                metadata_cache = pickle.load(f)
                self.transaction_metadata = metadata_cache.get('transaction_metadata', [])
                self.historical_labels = metadata_cache.get('historical_labels', None)
            print(f"  ✓ Loaded {len(self.transaction_metadata)} transaction records")
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
            print("Similar transaction retrieval will be limited.")
            self.transaction_metadata = []
            self.historical_labels = None

    def encode_transaction(self, txn: Dict) -> np.ndarray:
        """
        Encode a transaction into an embedding vector.

        Args:
            txn: Dictionary with transaction features

        Returns:
            Embedding vector as numpy array
        """
        # Text encoding
        text = f"The transaction description is: {txn['tran_partclr']}"
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Categorical encoding
        categorical_indices = [
            self.cat_vocab[col].get(txn.get(col, ''), 0)
            for col in self.categorical_cols
        ]
        categorical = torch.tensor([categorical_indices], dtype=torch.long).to(self.device)

        # Numeric encoding
        numeric_values = [[txn.get(col, 0.0) for col in self.numeric_cols]]
        numeric = torch.tensor(
            self.scaler.transform(numeric_values),
            dtype=torch.float
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            embedding = self.encoder(
                input_ids,
                attention_mask,
                categorical,
                numeric
            ).cpu().numpy().astype('float32')

        return embedding

    def predict(self, new_txn: Dict, top_k: int = 5) -> Dict:
        """
        Predict category and retrieve similar transactions for a new transaction.

        Args:
            new_txn: Dictionary containing transaction details
            top_k: Number of similar transactions to retrieve

        Returns:
            Dictionary with prediction results and similar transactions
        """
        # Encode the new transaction
        query_emb = self.encode_transaction(new_txn)

        # Search for similar transactions
        distances, indices = self.index.search(query_emb, top_k)

        if self.historical_labels is not None:
            retrieved_labels = self.historical_labels[indices[0]]
            pred_label_code = np.bincount(retrieved_labels).argmax()
            predicted_category = self.label_mapping[pred_label_code]
            confidence = np.bincount(retrieved_labels).max() / top_k
            top_k_labels = [self.label_mapping[label] for label in retrieved_labels]
        else:
            predicted_category = "Unknown"
            confidence = 0.0
            top_k_labels = []

        # Retrieve actual similar transactions
        similar_transactions = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.transaction_metadata):
                txn_data = self.transaction_metadata[idx]
                similar_transactions.append({
                    'transaction': {
                        'description': txn_data.get('tran_partclr', 'N/A'),
                        'amount': txn_data.get('tran_amt_in_ac', 0.0),
                        'dr_cr': txn_data.get('dr_cr_indctor', 'N/A'),
                        'mode': txn_data.get('tran_mode', 'N/A'),
                        'merchant': txn_data.get('merchant', 'N/A'),
                        'date': txn_data.get('tran_date', 'N/A'),
                        'category': txn_data.get(self.label_col, 'N/A')
                    },
                    'similarity_score': float(distance),
                    'label': self.label_mapping.get(
                        self.historical_labels[idx] if self.historical_labels is not None else -1,
                        'Unknown'
                    )
                })

        return {
            'predicted_category': predicted_category,
            'confidence': confidence,
            'top_k_labels': top_k_labels,
            'distances': distances[0].tolist(),
            'similar_transactions': similar_transactions
        }

    def predict_batch(self, transactions: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Predict categories for multiple transactions.

        Args:
            transactions: List of transaction dictionaries
            top_k: Number of similar transactions to retrieve per transaction

        Returns:
            List of prediction results
        """
        return [self.predict(txn, top_k) for txn in transactions]


if __name__ == "__main__":
    # Example usage
    predictor = TransactionPredictor(
        artifacts_path="training_artifacts/training_artifacts.pkl",
        model_path="experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth",
        faiss_index_path="transaction_index.faiss"
    )

    # Test transaction
    sample_txn = {
        'tran_partclr': 'GROCERY STORE PURCHASE',
        'tran_mode': 'POS',
        'dr_cr_indctor': 'D',
        'sal_flag': 'N',
        'tran_amt_in_ac': 75.50
    }

    result = predictor.predict(sample_txn, top_k=5)

    print("\n" + "=" * 80)
    print("PREDICTION RESULT")
    print("=" * 80)
    print(f"Transaction: {sample_txn['tran_partclr']}")
    print(f"Amount: ${sample_txn['tran_amt_in_ac']:.2f}")
    print(f"\nPredicted Category: {result['predicted_category']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Top-5 Categories: {', '.join(result['top_k_labels'])}")

    print("\n" + "=" * 80)
    print("SIMILAR HISTORICAL TRANSACTIONS")
    print("=" * 80)
    for i, similar_txn in enumerate(result['similar_transactions'], 1):
        txn = similar_txn['transaction']
        print(f"\n{i}. Category: {txn['category']} | Similarity: {similar_txn['similarity_score']:.4f}")
        print(f"   Description: {txn['description']}")
        print(f"   Amount: ${txn['amount']:.2f} | Mode: {txn['mode']} | DR/CR: {txn['dr_cr']}")
        print(f"   Merchant: {txn['merchant']} | Date: {txn['date']}")
    print("\n" + "=" * 80)

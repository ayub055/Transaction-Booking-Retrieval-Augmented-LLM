import torch
import pandas as pd
import faiss
import numpy as np
import pickle
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from src.fusion_encoder import FusionEncoder
from src.data_loader import TransactionDataset, collate_fn
import os

csv_path = "sample_txn.csv" 
artifacts_path = "training_artifacts/training_artifacts.pkl"  
model_path = "experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth"
faiss_index_path = "transaction_index.faiss"
top_k = 5

print("Loading training artifacts...")

with open(artifacts_path, 'rb') as f:
    artifacts = pickle.load(f)

config = artifacts['config']
categorical_cols = config['categorical_cols']
numeric_cols = config['numeric_cols']
label_col = config['label_col']
bert_model = config['bert_model']
text_proj_dim = config['text_proj_dim']
final_dim = config['final_dim']
dropout = config['dropout']
categorical_dims = artifacts['categorical_dims']
transaction_metadata = artifacts.get('transaction_metadata', [])

print(f"  Categorical dimensions: {categorical_dims}")
print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Label categories: {len(artifacts['label_mapping'])}")
print(f"  Transaction metadata: {len(transaction_metadata)} records")
print(f"  Model architecture: proj={text_proj_dim}, final={final_dim}")
print()

print("Loading historical data...")
# Try to load from metadata file if FAISS index exists, otherwise from CSV
metadata_path = faiss_index_path.replace('.faiss', '_metadata.pkl')
if os.path.exists(metadata_path):
    print(f"Loading from existing metadata file: {metadata_path}")
    with open(metadata_path, 'rb') as f:
        metadata_cache = pickle.load(f)
        transaction_metadata = metadata_cache.get('transaction_metadata', artifacts.get('transaction_metadata', []))
        historical_labels_cached = metadata_cache.get('historical_labels', None)
    historical_df = pd.DataFrame(transaction_metadata) if transaction_metadata else pd.read_csv(csv_path)
else:
    historical_df = pd.read_csv(csv_path)

print(f"Loaded {len(historical_df)} transactions")
print()


print("Creating dataset with training preprocessing...")
tokenizer = BertTokenizer.from_pretrained(bert_model)
dataset = TransactionDataset(historical_df, tokenizer, categorical_cols, numeric_cols, label_col)

dataset.cat_vocab = artifacts['cat_vocab']
dataset.scaler = artifacts['scaler']
dataset.label_mapping = artifacts['label_mapping']
dataset.numeric_data = dataset.scaler.transform(historical_df[numeric_cols])
dataset.labels = historical_df[label_col].astype('category').cat.codes

dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn, shuffle=False)


encoder = FusionEncoder(
    bert_model_name=bert_model,
    categorical_dims=categorical_dims,
    numeric_dim=len(numeric_cols),
    text_proj_dim=text_proj_dim,
    final_dim=final_dim,
    p=dropout
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint = torch.load(model_path, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print(f"Checkpoint format detected")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_recall5' in checkpoint:
        print(f"  Best validation recall@5: {checkpoint['val_recall5']:.4f}")
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"  Validation accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Validation MRR: {val_metrics.get('mrr', 'N/A'):.4f}")
    encoder.load_state_dict(checkpoint['model_state_dict'])
else:
    print("Legacy state_dict format detected")
    encoder.load_state_dict(checkpoint)

encoder.to(device)
encoder.eval()
print(" Model loaded successfully!")
print()


print("Encoding historical transactions...")
historical_embeddings = []
historical_labels = []

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        categorical = batch['categorical'].to(device)
        numeric = batch['numeric'].to(device)

        emb = encoder(input_ids, attention_mask, categorical, numeric)
        historical_embeddings.append(emb.cpu())
        historical_labels.append(batch['labels'].cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Processed {(batch_idx + 1) * len(input_ids)} / {len(dataset)} transactions")

historical_embeddings = torch.cat(historical_embeddings).numpy().astype('float32')
historical_labels = torch.cat(historical_labels).numpy()

print(f"Total embeddings stored: {historical_embeddings.shape[0]}")
print(f"Embedding dimension: {historical_embeddings.shape[1]}")
print()
        
print("Building FAISS index...")
index = faiss.IndexFlatL2(historical_embeddings.shape[1])
index.add(historical_embeddings)
print(f"FAISS index size: {index.ntotal}")

faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved at: {faiss_index_path}")

# Save transaction metadata alongside FAISS index for retrieval
metadata_path = faiss_index_path.replace('.faiss', '_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump({
        'transaction_metadata': transaction_metadata,
        'historical_labels': historical_labels
    }, f)
print(f"Transaction metadata saved at: {metadata_path}")
print()

label_mapping = dataset.label_mapping
print(f"Label mapping loaded: {len(label_mapping)} unique categories")

def predict_gl_account_faiss(new_txn: dict):
    """
    Predict GL account for a new transaction and return similar historical transactions.

    Args:
        new_txn: Dictionary containing transaction details

    Returns:
        Dictionary with predicted category, confidence, and similar transactions
    """
    text = f"The transaction description is: {new_txn['tran_partclr']}"
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    categorical_indices = [dataset.cat_vocab[col].get(new_txn[col], 0) for col in categorical_cols]
    categorical = torch.tensor([categorical_indices], dtype=torch.long).to(device)

    numeric_values = [[new_txn[col] for col in numeric_cols]]
    numeric = torch.tensor(dataset.scaler.transform(numeric_values), dtype=torch.float).to(device)

    with torch.no_grad():
        query_emb = encoder(input_ids, attention_mask, categorical, numeric).cpu().numpy().astype('float32')

    distances, indices = index.search(query_emb, top_k)
    retrieved_labels = historical_labels[indices[0]]

    pred_label_code = np.bincount(retrieved_labels).argmax()
    predicted_category = label_mapping[pred_label_code]

    # Retrieve actual similar transactions
    similar_transactions = []
    for idx, distance in zip(indices[0], distances[0]):
        txn_data = transaction_metadata[idx]
        similar_transactions.append({
            'transaction': {
                'description': txn_data.get('tran_partclr', 'N/A'),
                'amount': txn_data.get('tran_amt_in_ac', 0.0),
                'dr_cr': txn_data.get('dr_cr_indctor', 'N/A'),
                'mode': txn_data.get('tran_mode', 'N/A'),
                'merchant': txn_data.get('merchant', 'N/A'),
                'date': txn_data.get('tran_date', 'N/A'),
                'category': txn_data.get(label_col, 'N/A')
            },
            'similarity_score': float(distance),
            'label': label_mapping[historical_labels[idx]]
        })

    return {
        'predicted_category': predicted_category,
        'confidence': np.bincount(retrieved_labels).max() / top_k,
        'top_k_labels': [label_mapping[label] for label in retrieved_labels],
        'distances': distances[0].tolist(),
        'similar_transactions': similar_transactions
    }


sample_txn = {
    'tran_partclr': 'WIRE TRANSFER PAYMENT TO VENDOR ABC',
    'tran_mode': historical_df.iloc[0]['tran_mode'],
    'dr_cr_indctor': historical_df.iloc[0]['dr_cr_indctor'],
    'sal_flag': historical_df.iloc[0]['sal_flag'],
    'tran_amt_in_ac': historical_df.iloc[0]['tran_amt_in_ac']
}

print(f"Transaction: {sample_txn['tran_partclr']}")
print(f"Amount: {sample_txn['tran_amt_in_ac']}")
print()

result = predict_gl_account_faiss(sample_txn)
print(f"Predicted Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top-{top_k} similar categories: {', '.join(result['top_k_labels'])}")
print()

print("=" * 80)
print("SIMILAR HISTORICAL TRANSACTIONS")
print("=" * 80)
for i, similar_txn in enumerate(result['similar_transactions'], 1):
    txn = similar_txn['transaction']
    print(f"\n{i}. Category: {txn['category']} | Similarity Score: {similar_txn['similarity_score']:.4f}")
    print(f"   Description: {txn['description']}")
    print(f"   Amount: {txn['amount']} | Mode: {txn['mode']} | DR/CR: {txn['dr_cr']}")
    print(f"   Merchant: {txn['merchant']} | Date: {txn['date']}")
print("\n" + "=" * 80)

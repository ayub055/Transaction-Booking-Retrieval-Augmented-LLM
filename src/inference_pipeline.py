"""
Enhanced Inference Pipeline for Transaction Tagging with RAG

Implements the Amazon Science paper "Cash booking with Retrieval Augmented LLM":
1. Build a STRATIFIED golden-record FAISS index (class-capped to prevent majority
   class density bias in k-NN retrieval).
2. Compute per-class prototype (centroid) embeddings.
3. Calibrate an OOD distance threshold using real "other"-labelled examples.
4. At inference: retrieve top-k → distance-weighted vote + prototype similarity
   → confidence → OOD flag → fast path or hand-off to LLM.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import faiss
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from src.fusion_encoder import FusionEncoder
from src.data_loader import TransactionDataset, collate_fn, _normalise_df


# ---------------------------------------------------------------------------
# Helper: diverse representative selection via mini-batch k-medoids
# ---------------------------------------------------------------------------
def _select_diverse_representatives(embeddings: np.ndarray, k: int) -> np.ndarray:
    """
    Select k indices that best span the cluster's diversity.
    Uses greedy farthest-point sampling (cheap approximate k-medoids).

    Returns indices into `embeddings`.
    """
    if len(embeddings) <= k:
        return np.arange(len(embeddings))

    selected = [np.random.randint(len(embeddings))]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(embeddings - embeddings[selected][:, None], axis=2),
            axis=0,
        )
        selected.append(int(np.argmax(dists)))
    return np.array(selected)


# ---------------------------------------------------------------------------
# GoldenRecordIndexer
# ---------------------------------------------------------------------------
class GoldenRecordIndexer:
    """
    Encodes a labeled golden-record CSV and builds:
      • A STRATIFIED FAISS HNSW index (majority classes capped to prevent
        density bias in k-NN retrieval).
      • Per-class prototype (centroid) embeddings.
      • An OOD distance threshold (calibrated from within-corpus distances or
        supplied "other"-labelled examples).
    """

    def __init__(
        self,
        artifacts_path: str,
        model_path: str,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        self.artifacts_path = artifacts_path
        self.model_path     = model_path
        self.device         = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        print(f"Initializing GoldenRecordIndexer on {self.device}")
        self._load_artifacts()
        self._load_model()

    # ------------------------------------------------------------------
    def _load_artifacts(self):
        print(f"Loading artifacts from {self.artifacts_path}...")
        with open(self.artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        self.config          = artifacts['config']
        self.categorical_cols = self.config['categorical_cols']
        self.numeric_cols    = self.config['numeric_cols']
        self.label_col       = self.config['label_col']
        self.cat_vocab       = artifacts['cat_vocab']
        self.scaler          = artifacts['scaler']
        self.label_mapping   = artifacts['label_mapping']
        self.categorical_dims = artifacts['categorical_dims']
        self.tokenizer       = BertTokenizer.from_pretrained(self.config['bert_model'])
        print(f"  ✓ Loaded {len(self.label_mapping)} label categories")

    def _load_model(self):
        print(f"Loading model from {self.model_path}...")
        self.encoder = FusionEncoder(
            bert_model_name=self.config['bert_model'],
            categorical_dims=self.categorical_dims,
            numeric_dim=len(self.numeric_cols),
            text_proj_dim=self.config['text_proj_dim'],
            final_dim=self.config['final_dim'],
            p=self.config['dropout'],
        )
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.encoder.load_state_dict(ckpt['model_state_dict'])
            print(f"  ✓ Loaded checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
        else:
            self.encoder.load_state_dict(ckpt)
        self.encoder.to(self.device).eval()
        if self.use_fp16:
            self.encoder = self.encoder.half()
            print(f"  ✓ Model converted to FP16")

    # ------------------------------------------------------------------
    def _encode_dataset(self, df: pd.DataFrame, batch_size: int = 512):
        """Encode all rows in df → (embeddings [N,D], labels [N])."""
        dataset = TransactionDataset(
            df, self.tokenizer,
            self.categorical_cols, self.numeric_cols, self.label_col,
        )
        dataset.cat_vocab    = self.cat_vocab
        dataset.scaler       = self.scaler
        dataset.label_mapping = self.label_mapping
        dataset.numeric_data  = self.scaler.transform(df[self.numeric_cols])
        dataset.labels        = df[self.label_col].map(
            {v: k for k, v in self.label_mapping.items()}
        ).fillna(-1).astype(int)

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for batch in loader:
                ids    = batch['input_ids'].to(self.device)
                mask   = batch['attention_mask'].to(self.device)
                cat    = batch['categorical'].to(self.device)
                num    = batch['numeric'].to(self.device)
                if self.use_fp16:
                    num = num.half()
                emb = self.encoder(ids, mask, cat, num).cpu().float()
                all_emb.append(emb)
                all_lbl.append(batch['labels'].cpu())

        embeddings = torch.cat(all_emb).numpy().astype('float32')
        labels     = torch.cat(all_lbl).numpy()
        return embeddings, labels

    # ------------------------------------------------------------------
    def build_index(
        self,
        csv_path: str,
        output_path: str,
        batch_size: int = 512,
        index_type: str = 'HNSW',
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        max_per_class: int = 500,
        ood_csv_path: Optional[str] = None,
        ood_label: str = 'other',
        ood_percentile: float = 95.0,
    ) -> Tuple[int, int]:
        """
        Build a stratified FAISS index, compute class prototypes, and
        calibrate the OOD distance threshold.

        Args:
            csv_path:            CSV of 43-category labeled golden records.
            output_path:         Where to write the .faiss file.
            batch_size:          Encoding batch size.
            index_type:          'HNSW', 'L2', 'IP', or 'IVF'.
            hnsw_m:              HNSW M parameter.
            hnsw_ef_construction: HNSW build-time depth.
            max_per_class:       Max examples per class in the retrieval index.
                                 Majority classes (> max_per_class) are reduced
                                 to max_per_class diverse representatives.
            ood_csv_path:        Optional CSV containing "other"-labelled
                                 transactions for calibrating the OOD threshold.
                                 If None, uses the 95th-percentile intra-corpus
                                 nearest-neighbour distance instead.
            ood_label:           Label string to treat as OOD.
            ood_percentile:      Fallback percentile for intra-corpus threshold.
        """
        print(f"\n{'='*70}")
        print("Building Stratified Golden Record Index")
        print(f"{'='*70}")

        # ------------------------------------------------------------------ #
        # 1. Load & encode full golden record CSV (43-category rows only)
        # ------------------------------------------------------------------ #
        df = pd.read_csv(csv_path)
        df = _normalise_df(df)
        df = df[df[self.label_col] != ood_label].reset_index(drop=True)
        print(f"  ✓ Loaded {len(df)} in-distribution golden records")

        all_embeddings, all_labels = self._encode_dataset(df, batch_size)
        print(f"  ✓ Encoded {len(all_embeddings)} embeddings (dim={all_embeddings.shape[1]})")

        # ------------------------------------------------------------------ #
        # 2. Build stratified index: cap majority classes via diverse selection
        # ------------------------------------------------------------------ #
        label_to_idx: dict = defaultdict(list)
        for i, lbl in enumerate(all_labels):
            label_to_idx[int(lbl)].append(i)

        selected_global = []
        for lbl, idx_list in label_to_idx.items():
            if len(idx_list) <= max_per_class:
                selected_global.extend(idx_list)
            else:
                class_embs    = all_embeddings[idx_list]
                diverse_local = _select_diverse_representatives(class_embs, max_per_class)
                selected_global.extend([idx_list[i] for i in diverse_local])

        selected_global = np.array(selected_global)
        index_embeddings = all_embeddings[selected_global].astype('float32')
        index_labels     = all_labels[selected_global]
        index_metadata   = [df.to_dict('records')[i] for i in selected_global]

        print(f"  ✓ Stratified index: {len(index_embeddings)} records "
              f"(max {max_per_class}/class from {len(df)} total)")

        # ------------------------------------------------------------------ #
        # 3. Compute per-class prototypes (L2-normalised centroids)
        # ------------------------------------------------------------------ #
        prototypes: dict = {}
        for lbl in label_to_idx.keys():
            class_embs = all_embeddings[all_labels == lbl]
            centroid   = class_embs.mean(axis=0)
            # L2-normalise
            centroid  /= np.linalg.norm(centroid) + 1e-9
            prototypes[int(lbl)] = centroid.astype('float32')
        print(f"  ✓ Computed {len(prototypes)} class prototypes")

        # ------------------------------------------------------------------ #
        # 4. Build FAISS index from stratified embeddings
        # ------------------------------------------------------------------ #
        dim = index_embeddings.shape[1]
        if index_type == 'HNSW':
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
            index.hnsw.efConstruction = hnsw_ef_construction
        elif index_type == 'L2':
            index = faiss.IndexFlatL2(dim)
        elif index_type == 'IP':
            index = faiss.IndexFlatIP(dim)
        elif index_type == 'IVF':
            n_clusters = min(int(np.sqrt(len(index_embeddings))), 1024)
            quantizer  = faiss.IndexFlatL2(dim)
            index      = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
            index.train(index_embeddings)
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        index.add(index_embeddings)
        faiss.write_index(index, output_path)
        print(f"  ✓ FAISS {index_type} index saved → {output_path}")

        # ------------------------------------------------------------------ #
        # 5. Calibrate OOD distance threshold
        # ------------------------------------------------------------------ #
        ood_threshold = self._calibrate_ood_threshold(
            index, index_embeddings, index_labels,
            ood_csv_path=ood_csv_path, ood_label=ood_label,
            batch_size=batch_size, percentile=ood_percentile,
        )
        print(f"  ✓ OOD threshold = {ood_threshold:.4f}")

        # ------------------------------------------------------------------ #
        # 6. Save metadata
        # ------------------------------------------------------------------ #
        metadata_path = output_path.replace('.faiss', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'transaction_metadata': index_metadata,
                'labels':               index_labels,
                'embeddings':           index_embeddings,
                'prototypes':           prototypes,
                'ood_threshold':        ood_threshold,
                'label_mapping':        self.label_mapping,
            }, f)
        print(f"  ✓ Metadata saved → {metadata_path}")
        print(f"\n{'='*70}")
        print(f"Index build complete: {len(index_embeddings)} vectors, dim={dim}")
        print(f"{'='*70}\n")
        return len(index_embeddings), dim

    # ------------------------------------------------------------------
    def _calibrate_ood_threshold(
        self,
        index,
        index_embeddings: np.ndarray,
        index_labels: np.ndarray,
        ood_csv_path: Optional[str],
        ood_label: str,
        batch_size: int,
        percentile: float,
    ) -> float:
        """
        If ood_csv_path is provided: find the threshold that best separates
        known "other" transactions (OOD) from golden-record queries.
        Otherwise: use the 95th-percentile of within-corpus NN distances.
        """
        # Within-corpus: each golden record queries itself → take k=2 to skip
        # self-match (for HNSW, self is not guaranteed to be returned, so k=2
        # is a safe proxy).
        sample_n  = min(5000, len(index_embeddings))
        sample_idx = np.random.choice(len(index_embeddings), sample_n, replace=False)
        corpus_dists, _ = index.search(index_embeddings[sample_idx], k=2)
        in_dist_dists   = corpus_dists[:, 1]   # nearest non-self neighbor

        if ood_csv_path and os.path.exists(ood_csv_path):
            ood_df = pd.read_csv(ood_csv_path)
            ood_df = _normalise_df(ood_df)
            ood_df = ood_df[ood_df[self.label_col] == ood_label].reset_index(drop=True)

            if len(ood_df) > 0:
                ood_embs, _ = self._encode_dataset(
                    ood_df.assign(**{self.label_col: ood_label}), batch_size
                )
                # Use a dummy label_mapping for OOD encoding
                ood_dists, _ = index.search(ood_embs.astype('float32'), k=1)
                ood_dists    = ood_dists[:, 0]

                # Find threshold that maximises F1 for OOD detection
                all_dists  = np.concatenate([in_dist_dists, ood_dists])
                all_labels = np.array([0] * len(in_dist_dists) + [1] * len(ood_dists))
                best_f1, best_thresh = 0.0, float(np.median(all_dists))

                for thresh in np.percentile(all_dists, np.arange(10, 96, 2)):
                    preds = (all_dists > thresh).astype(int)
                    tp = ((preds == 1) & (all_labels == 1)).sum()
                    fp = ((preds == 1) & (all_labels == 0)).sum()
                    fn = ((preds == 0) & (all_labels == 1)).sum()
                    if tp + fp == 0 or tp + fn == 0:
                        continue
                    prec = tp / (tp + fp)
                    rec  = tp / (tp + fn)
                    f1   = 2 * prec * rec / (prec + rec + 1e-9)
                    if f1 > best_f1:
                        best_f1, best_thresh = f1, float(thresh)

                print(f"  OOD threshold calibrated from {len(ood_df)} 'other' examples "
                      f"(best F1={best_f1:.3f})")
                return best_thresh

        # Fallback: percentile of within-corpus distances
        return float(np.percentile(in_dist_dists, percentile))


# ---------------------------------------------------------------------------
# TransactionInferencePipeline
# ---------------------------------------------------------------------------
class TransactionInferencePipeline:
    """
    Full inference pipeline:
      encode → stratified k-NN → distance-weighted vote + prototype score
             → OOD check → prediction (or hand-off to LLM).
    """

    def __init__(
        self,
        artifacts_path: str,
        model_path: str,
        index_path: str,
        device: Optional[str] = None,
        use_fp16: bool = True,
        hnsw_ef_search: int = 128,
        knn_weight: float = 0.6,       # α in: α·knn + (1-α)·prototype
        confidence_threshold: float = 0.75,
    ):
        self.artifacts_path      = artifacts_path
        self.model_path          = model_path
        self.index_path          = index_path
        self.device              = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.use_fp16            = use_fp16 and self.device.type == 'cuda'
        self.hnsw_ef_search      = hnsw_ef_search
        self.knn_weight          = knn_weight
        self.confidence_threshold = confidence_threshold
        self._load_components()

    # ------------------------------------------------------------------
    def _load_components(self):
        with open(self.artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        self.config           = artifacts['config']
        self.categorical_cols = self.config['categorical_cols']
        self.numeric_cols     = self.config['numeric_cols']
        self.label_col        = self.config['label_col']
        self.cat_vocab        = artifacts['cat_vocab']
        self.scaler           = artifacts['scaler']
        self.label_mapping    = artifacts['label_mapping']
        self.categorical_dims = artifacts['categorical_dims']
        self.tokenizer        = BertTokenizer.from_pretrained(self.config['bert_model'])

        self.encoder = FusionEncoder(
            bert_model_name=self.config['bert_model'],
            categorical_dims=self.categorical_dims,
            numeric_dim=len(self.numeric_cols),
            text_proj_dim=self.config['text_proj_dim'],
            final_dim=self.config['final_dim'],
            p=self.config['dropout'],
        )
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.encoder.load_state_dict(ckpt['model_state_dict'])
        else:
            self.encoder.load_state_dict(ckpt)
        self.encoder.to(self.device).eval()
        if self.use_fp16:
            self.encoder = self.encoder.half()

        self.index = faiss.read_index(self.index_path)
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.hnsw_ef_search

        metadata_path = self.index_path.replace('.faiss', '_metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            meta = pickle.load(f)

        self.transaction_metadata = meta.get('transaction_metadata', [])
        self.golden_labels        = meta.get('labels', None)
        self.golden_embeddings    = meta.get('embeddings', None)
        self.prototypes           = meta.get('prototypes', {})     # {int_label: ndarray [D]}
        self.ood_threshold        = meta.get('ood_threshold', None)

        # Stack prototypes into a matrix for fast batch scoring [num_classes, D]
        if self.prototypes:
            sorted_labels         = sorted(self.prototypes.keys())
            self._proto_labels    = sorted_labels
            self._proto_matrix    = np.stack(
                [self.prototypes[l] for l in sorted_labels], axis=0
            ).astype('float32')   # [C, D]
        else:
            self._proto_labels  = []
            self._proto_matrix  = None

        print(f"Pipeline ready | index={self.index.ntotal} vectors "
              f"| prototypes={len(self.prototypes)} "
              f"| OOD threshold={self.ood_threshold:.4f if self.ood_threshold else 'N/A'}")

    # ------------------------------------------------------------------
    def encode_transaction(self, txn: Dict) -> np.ndarray:
        text = f"The transaction description is: {txn.get('tran_partclr', '')}"
        enc  = self.tokenizer(text, padding='max_length', truncation=True,
                               max_length=128, return_tensors='pt')
        ids  = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)
        cat  = torch.tensor(
            [[self.cat_vocab[col].get(
                str(txn.get(col, 'NULL')) if txn.get(col) is not None else 'NULL', 0
            ) for col in self.categorical_cols]],
            dtype=torch.long,
        ).to(self.device)
        num  = torch.tensor(
            self.scaler.transform([[txn.get(c, 0.0) for c in self.numeric_cols]]),
            dtype=torch.float,
        ).to(self.device)
        if self.use_fp16:
            num = num.half()
        with torch.no_grad():
            emb = self.encoder(ids, mask, cat, num).cpu().float().numpy().astype('float32')
        return emb

    # ------------------------------------------------------------------
    def _distance_weighted_vote(
        self, retrieved_labels: np.ndarray, distances: np.ndarray
    ) -> Tuple[int, float, dict]:
        """
        Inverse-distance weighted majority vote.
        Returns (predicted_label_code, confidence, vote_distribution).
        """
        weights = 1.0 / (distances + 1e-6)
        vote_map: dict = defaultdict(float)
        for lbl, w in zip(retrieved_labels, weights):
            vote_map[int(lbl)] += float(w)

        total_w      = sum(vote_map.values())
        pred_label   = max(vote_map, key=vote_map.get)
        confidence   = vote_map[pred_label] / total_w

        vote_dist = {
            self.label_mapping[lbl]: round(w / total_w, 4)
            for lbl, w in sorted(vote_map.items(), key=lambda x: -x[1])
        }
        return pred_label, float(confidence), vote_dist

    # ------------------------------------------------------------------
    def _prototype_scores(self, query_embedding: np.ndarray) -> dict:
        """
        Cosine similarity of query to each class prototype.
        Returns {label_code: score} sorted descending.
        """
        if self._proto_matrix is None:
            return {}
        # query is already L2-normalised; prototypes are L2-normalised
        sims = (self._proto_matrix @ query_embedding.T).flatten()  # [C]
        return {self._proto_labels[i]: float(sims[i]) for i in range(len(self._proto_labels))}

    # ------------------------------------------------------------------
    def _combine_scores(
        self,
        knn_vote_dist: dict,      # {label_code: weight_fraction}
        proto_scores: dict,       # {label_code: cosine_sim  [-1,1]}
    ) -> Tuple[int, float]:
        """
        Combine k-NN distance-weighted votes with prototype similarity scores.
        Returns (predicted_label_code, combined_confidence).
        """
        all_labels = set(list(knn_vote_dist.keys()) + list(proto_scores.keys()))
        combined: dict = {}
        # Normalise prototype scores to [0,1] via min-max
        if proto_scores:
            p_vals = list(proto_scores.values())
            p_min, p_range = min(p_vals), max(p_vals) - min(p_vals)
            p_norm = {l: (s - p_min) / (p_range + 1e-9) for l, s in proto_scores.items()}
        else:
            p_norm = {}

        for lbl in all_labels:
            knn_s   = knn_vote_dist.get(lbl, 0.0)
            proto_s = p_norm.get(lbl, 0.0)
            combined[lbl] = self.knn_weight * knn_s + (1 - self.knn_weight) * proto_s

        pred_label = max(combined, key=combined.get)
        total      = sum(combined.values()) + 1e-9
        confidence = combined[pred_label] / total
        return pred_label, float(confidence)

    # ------------------------------------------------------------------
    def predict(
        self,
        new_txn: Dict,
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> Dict:
        """
        Predict category for a single transaction.

        Returns dict with:
          predicted_category, confidence, is_ood, ood_distance,
          vote_distribution, similar_transactions, distances,
          needs_llm  (True when confidence < threshold or is_ood),
          [query_embedding] (optional)
        """
        query_emb = self.encode_transaction(new_txn)

        distances, indices = self.index.search(query_emb, top_k)
        distances = distances[0]
        indices   = indices[0]

        # OOD check
        min_dist = float(distances.min())
        is_ood   = (self.ood_threshold is not None) and (min_dist > self.ood_threshold)

        retrieved_labels = self.golden_labels[indices]

        # Distance-weighted vote
        knn_label_scores: dict = defaultdict(float)
        weights = 1.0 / (distances + 1e-6)
        total_w = weights.sum()
        for lbl, w in zip(retrieved_labels, weights):
            knn_label_scores[int(lbl)] += float(w) / float(total_w)

        # Prototype scores
        proto_scores = self._prototype_scores(query_emb)

        # Combined prediction
        pred_label, confidence = self._combine_scores(knn_label_scores, proto_scores)
        predicted_category     = self.label_mapping.get(pred_label, 'other')

        # If flagged OOD and confidence is low → definitely "other"
        if is_ood and confidence < self.confidence_threshold:
            predicted_category = 'other'
            confidence         = 0.0

        needs_llm = is_ood or (confidence < self.confidence_threshold)

        # Build vote distribution for interpretability
        vote_dist = {
            self.label_mapping.get(lbl, str(lbl)): round(score, 4)
            for lbl, score in sorted(knn_label_scores.items(), key=lambda x: -x[1])
        }

        similar_transactions = [
            {
                'index': int(idx),
                'transaction': {
                    'description': self.transaction_metadata[idx].get('tran_partclr', 'N/A'),
                    'amount':      float(self.transaction_metadata[idx].get('tran_amt_in_ac', 0.0)),
                    'dr_cr':       self.transaction_metadata[idx].get('dr_cr_indctor', 'N/A'),
                    'mode':        self.transaction_metadata[idx].get('tran_mode', 'N/A'),
                    'date':        self.transaction_metadata[idx].get('tran_date', 'N/A'),
                    'category':    self.transaction_metadata[idx].get(self.label_col, 'N/A'),
                },
                'distance':  float(dist),
                'label':     self.label_mapping.get(int(self.golden_labels[idx]), 'N/A'),
            }
            for idx, dist in zip(indices, distances)
        ]

        result = {
            'predicted_category':  predicted_category,
            'confidence':          confidence,
            'is_ood':              is_ood,
            'ood_distance':        min_dist,
            'ood_threshold':       self.ood_threshold,
            'needs_llm':           needs_llm,
            'vote_distribution':   vote_dist,
            'similar_transactions': similar_transactions,
            'distances':           distances.tolist(),
        }
        if return_embeddings:
            result['query_embedding'] = query_emb
        return result

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        transactions: List[Dict],
        top_k: int = 10,
        batch_size: int = 512,
    ) -> List[Dict]:
        """Predict categories for a list of transactions (batched encoding)."""
        if not transactions:
            return []

        all_embeddings = []
        for i in range(0, len(transactions), batch_size):
            batch_txns = transactions[i:i + batch_size]
            texts      = [f"The transaction description is: {t.get('tran_partclr', '')}"
                          for t in batch_txns]
            enc = self.tokenizer(texts, padding='max_length', truncation=True,
                                  max_length=128, return_tensors='pt')
            ids  = enc['input_ids'].to(self.device)
            mask = enc['attention_mask'].to(self.device)

            cat_batch = [
                [self.cat_vocab[col].get(
                    str(t.get(col, 'NULL')) if t.get(col) is not None else 'NULL', 0
                ) for col in self.categorical_cols]
                for t in batch_txns
            ]
            cat = torch.tensor(cat_batch, dtype=torch.long).to(self.device)
            num_batch = [[t.get(c, 0.0) for c in self.numeric_cols] for t in batch_txns]
            num = torch.tensor(self.scaler.transform(num_batch), dtype=torch.float).to(self.device)
            if self.use_fp16:
                num = num.half()

            with torch.no_grad():
                embs = self.encoder(ids, mask, cat, num).cpu().float().numpy().astype('float32')
            all_embeddings.append(embs)

        all_embeddings = np.vstack(all_embeddings)
        all_distances, all_indices = self.index.search(all_embeddings, top_k)

        results = []
        for i, txn in enumerate(transactions):
            distances        = all_distances[i]
            indices          = all_indices[i]
            retrieved_labels = self.golden_labels[indices]
            query_emb        = all_embeddings[i:i+1]

            min_dist = float(distances.min())
            is_ood   = (self.ood_threshold is not None) and (min_dist > self.ood_threshold)

            knn_label_scores: dict = defaultdict(float)
            weights = 1.0 / (distances + 1e-6)
            total_w = weights.sum()
            for lbl, w in zip(retrieved_labels, weights):
                knn_label_scores[int(lbl)] += float(w) / float(total_w)

            proto_scores = self._prototype_scores(query_emb)
            pred_label, confidence = self._combine_scores(knn_label_scores, proto_scores)
            predicted_category = self.label_mapping.get(pred_label, 'other')

            if is_ood and confidence < self.confidence_threshold:
                predicted_category = 'other'
                confidence         = 0.0

            vote_dist = {
                self.label_mapping.get(lbl, str(lbl)): round(score, 4)
                for lbl, score in sorted(knn_label_scores.items(), key=lambda x: -x[1])
            }

            results.append({
                'predicted_category':  predicted_category,
                'confidence':          confidence,
                'is_ood':              is_ood,
                'ood_distance':        min_dist,
                'needs_llm':           is_ood or (confidence < self.confidence_threshold),
                'vote_distribution':   vote_dist,
                'distances':           distances.tolist(),
                'top_k_labels': [
                    self.label_mapping.get(int(l), 'N/A') for l in retrieved_labels
                ],
            })

        return results


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------
def print_prediction_result(result: Dict, transaction: Dict, top_k: int = 10):
    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Query:  {transaction.get('tran_partclr', 'N/A')}")
    print(f"Amount: {transaction.get('tran_amt_in_ac', 0):.2f} | "
          f"Mode: {transaction.get('tran_mode', 'N/A')} | "
          f"DR/CR: {transaction.get('dr_cr_indctor', 'N/A')}")
    print(f"\nPrediction : {result['predicted_category']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"OOD flag   : {result['is_ood']} (distance={result['ood_distance']:.4f}, "
          f"threshold={result.get('ood_threshold', 'N/A')})")
    print(f"Needs LLM  : {result['needs_llm']}")

    print("\nVote distribution:")
    for cat, score in list(result['vote_distribution'].items())[:5]:
        print(f"  {cat:30s} {score:.4f}")

    print(f"\nTop-{top_k} similar transactions:")
    for i, sim in enumerate(result.get('similar_transactions', [])[:top_k], 1):
        t = sim['transaction']
        print(f"  {i}. [{sim['label']}] d={sim['distance']:.4f}  "
              f"{t['description'][:60]}  ₹{t['amount']:.0f}")
    print("=" * 70 + "\n")

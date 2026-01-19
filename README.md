# Transaction Tagger using Retreival augmented LLM

## 📋 Project

**Transaction Tagger** is an automated financial transaction categorization system that solves the problem of transaction tagging in finance using a **Retrieval-Augmented Generation (RAG)** approach, inspired by [Amazon Science's "Cash transaction booking via retrieval augmented LLM"](https://www.amazon.science/publications/cash-transaction-booking-via-retrieval-augmented-llm). The system employs a multi-modal fusion encoder combining BERT for text understanding, learned categorical embeddings, and numerical features, trained with triplet margin loss for metric learning. Inference leverages FAISS HNSW indexing for fast similarity search over golden records, with majority voting for robust predictions. Optimized with FP16 precision and batch processing, achieving 10-50x speedup to process 1M transactions in ~15 minutes.

---

## 📋 Workflow Guide

This guide shows the **complete step-by-step workflow** from training to inference.

## ⚡ New Update (19Jan'25) : Optimizations (10-50x Faster!)

**Latest update:** Inference pipeline now includes HNSW index + FP16 + batch processing for **10-50x speedup!**

- **HNSW index:** 50x faster search
- **FP16 precision:** 2x faster encoding
- **Batch processing:** 2-5x faster batches
- **Result:** Process 1M transactions in ~15 minutes

**Quick start:**
```bash
python run_inference.py  # Auto-uses optimizations!
```
---


## Step 1: Train Your Model 

```
experiments/
└── tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/
    ├── fusion_encoder_best.pth          # Your trained model
    ├── fusion_encoder_epoch_*.pth
    └── logs/
        └── training_logs.json
```

**Verify you have:**
```bash
# Check your trained model exists
ls experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth
```

If you need to train:
```bash
python -m src.train_multi_expt --config experiments.yaml --format yaml
```

---

## Step 2: Save Training Artifacts

**What it does:** Extracts vocabularies, scalers, and label mappings from your training data. SInce Inference needs the same preprocessing (vocab, scaler) used during training.

```bash
python save_training_artifacts.py
```

**What it creates:**
```
training_artifacts/
├── training_artifacts.pkl     
└── model_config.json          # Model config
```

**Inside `training_artifacts.pkl`:**
- `cat_vocab` - Categorical vocabularies (tran_mode, dr_cr_indctor, sal_flag)
- `scaler` - StandardScaler for numeric features
- `label_mapping` - Maps label codes to category names
- `categorical_dims` - Vocabulary sizes for model initialization
- `transaction_metadata` - Full transaction records (optional, for golden records)
- `config` - Model hyperparameters

**Important:** Run this with the **same training data** you used for training!

---

## Step 3: Build Golden Record Index

Creates a FAISS index from your golden records (labeled historical transactions). This is later used based on RAG approach to retrieve similar transactions from this index.

### Option A: Using the Pipeline (Recommended)

```bash
python run_inference.py
```

This will:
1. Check if index exists
2. Build it if missing
3. Run sample predictions

### Option B: Programmatically

```python
from src.inference_pipeline import GoldenRecordIndexer

indexer = GoldenRecordIndexer(
    artifacts_path="training_artifacts/training_artifacts.pkl",
    model_path="experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth"
)

indexer.build_index(
    csv_path="data/sample_txn.csv",  # Your golden records
    output_path="golden_records.faiss",
    batch_size=128
)
```

**What it creates:**
```
golden_records.faiss           # FAISS index - embeddings
golden_records_metadata.pkl    # Transaction metadata + labels + embeddings
```

- Use the full dataset with labels (same as training data)
- This is "retrieval corpus"
- You can rebuild this when you have new labeled data

## Step 4: Run Inference

### Option A: Using the Ready-Made Script

```bash
python run_inference.py --skip-build
```
This will run predictions on sample transactions.

### Option B: In Your Own Code

```python
from src.inference_pipeline import TransactionInferencePipeline

# Initialize pipeline (do this once)
pipeline = TransactionInferencePipeline(
    artifacts_path="training_artifacts/training_artifacts.pkl",
    model_path="experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth",
    index_path="golden_records.faiss"
)

# Predict for a new transaction
new_transaction = {
    'tran_partclr': 'AMAZON PURCHASE ELECTRONICS',
    'tran_mode': 'ONLINE',
    'dr_cr_indctor': 'D',
    'sal_flag': 'N',
    'tran_amt_in_ac': 299.99
}

result = pipeline.predict(new_transaction, top_k=5)

print(f"Predicted: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Vote distribution: {result['vote_distribution']}")

# Show similar transactions
for similar in result['similar_transactions']:
    print(f"  - {similar['transaction']['description']}")
    print(f"    Category: {similar['label']}, Distance: {similar['similarity_distance']:.4f}")
```

## Quick Reference

### First Time Setup (Run Once)

```bash
# 1. Train model (if not done)
python -m src.train_multi_expt --config experiments.yaml --format yaml

# 2. Save artifacts
python save_training_artifacts.py

# 3. Build index and test
python run_inference.py
```

### Daily Usage (Reuse Existing Files)

```python
from src.inference_pipeline import TransactionInferencePipeline

pipeline = TransactionInferencePipeline(
    artifacts_path="training_artifacts/training_artifacts.pkl",
    model_path="experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth",
    index_path="golden_records.faiss"
)

# Predict on new transactions
result = pipeline.predict(new_txn, top_k=5)
```

## ✅ Checklist

Before running inference, make sure you have:

- [ ] Trained model: `experiments/.../fusion_encoder_best.pth`
- [ ] Training artifacts: `training_artifacts/training_artifacts.pkl`
- [ ] FAISS index: `golden_records.faiss`
- [ ] Metadata: `golden_records_metadata.pkl`
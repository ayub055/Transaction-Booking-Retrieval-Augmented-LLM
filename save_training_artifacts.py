"""
Save training artifacts (vocabularies, scalers, label mappings) from training data.
Run this once with your full training dataset to create artifacts for inference.

Usage:
    python save_training_artifacts.py
"""
import pandas as pd
import pickle
import json
import os
from transformers import BertTokenizer
from src.data_loader import TransactionDataset

csv_path = "./data/sample_txn.csv"
categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
numeric_cols = ['tran_amt_in_ac']
label_col = 'category'
bert_model = 'bert-base-uncased'
text_proj_dim = 256
final_dim = 256
dropout = 0.1

output_dir = "./training_artifacts"
os.makedirs(output_dir, exist_ok=True)

print("Loading training data...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} transactions")
print()

print("Building vocabularies and scalers...")
tokenizer = BertTokenizer.from_pretrained(bert_model)
dataset = TransactionDataset(df, tokenizer, categorical_cols, numeric_cols, label_col)

transaction_metadata = df.to_dict('records')  # Store all transaction details

artifacts = {
    # Preprocessing artifacts
    'cat_vocab': dataset.cat_vocab,
    'scaler': dataset.scaler,
    'label_mapping': dataset.label_mapping,

    # Model architecture info
    'categorical_dims': [len(dataset.cat_vocab[col]) for col in categorical_cols],

    # Original transaction data for similarity matching
    'transaction_metadata': transaction_metadata,

    # Configuration
    'config': {
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'label_col': label_col,
        'bert_model': bert_model,
        'text_proj_dim': text_proj_dim,
        'final_dim': final_dim,
        'dropout': dropout,
        'num_categories': len(dataset.label_mapping)
    }
}

artifacts_path = os.path.join(output_dir, "training_artifacts.pkl")
with open(artifacts_path, 'wb') as f:
    pickle.dump(artifacts, f)

print("=" * 60)
print("✓ Saved training artifacts")
print("=" * 60)
print(f"Location: {artifacts_path}")
print()
print("Artifacts include:")
print(f"  - Categorical vocabularies: {artifacts['categorical_dims']}")
print(f"  - Numeric scaler: StandardScaler with {len(numeric_cols)} features")
print(f"  - Label mapping: {len(dataset.label_mapping)} categories")
print(f"  - Transaction metadata: {len(transaction_metadata)} records")
print()

config_path = os.path.join(output_dir, "model_config.json")
config_dict = {
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'label_col': label_col,
    'bert_model': bert_model,
    'text_proj_dim': text_proj_dim,
    'final_dim': final_dim,
    'dropout': dropout,
    'categorical_dims': artifacts['categorical_dims'],
    'num_categories': len(dataset.label_mapping)
}
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print(f"✓ Saved model config: {config_path}")
print()

print("=" * 60)
print("Vocabulary Details")
print("=" * 60)
for col in categorical_cols:
    vocab_size = len(dataset.cat_vocab[col])
    print(f"{col}: {vocab_size} unique values (including <UNK>)")
print()

print("=" * 60)
print("Label Categories")
print("=" * 60)
for code, category in dataset.label_mapping.items():
    count = (df[label_col] == category).sum()
    print(f"  {code}: {category} ({count} samples)")
print()

print("=" * 60)
print("Done! Use these artifacts in inference:")
print("=" * 60)
print(f"""
# In your inference script:
import pickle

with open('{artifacts_path}', 'rb') as f:
    artifacts = pickle.load(f)

# Use artifacts instead of building from data
dataset.cat_vocab = artifacts['cat_vocab']
dataset.scaler = artifacts['scaler']
dataset.label_mapping = artifacts['label_mapping']
categorical_dims = artifacts['categorical_dims']
transaction_metadata = artifacts['transaction_metadata']
""")


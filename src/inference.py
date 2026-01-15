import torch
import pandas as pd
import faiss
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from src.fusion_encoder import FusionEncoder
from src.data_loader import TransactionDataset, collate_fn

# -----------------------------
# CONFIGURATION
# -----------------------------
csv_path = "sample_txn.csv"  # Historical transactions
categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
numeric_cols = ['tran_amt_in_ac']
label_col = 'category'
bert_model = 'bert-base-uncased'
faiss_index_path = "transaction_index.faiss"
model_path = "fusion_encoder.pth"
top_k = 5

# -----------------------------
# STEP 1: Load trained encoder
# -----------------------------
print("Loading trained FusionEncoder model...")
historical_df = pd.read_csv(csv_path)

tokenizer = BertTokenizer.from_pretrained(bert_model)
dataset = TransactionDataset(historical_df, tokenizer, categorical_cols, numeric_cols, label_col)
dataloader = DataLoader(dataset, batch_size=128, collate_fn=collate_fn)

categorical_dims = [len(dataset.cat_vocab[col]) for col in categorical_cols]
encoder = FusionEncoder(categorical_dims=categorical_dims, numeric_dim=len(numeric_cols))
encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
encoder.eval()

# -----------------------------
# STEP 2: Encode historical transactions
# -----------------------------
print("Encoding historical transactions...")
historical_embeddings = []
historical_labels = []

with torch.no_grad():
    for batch in dataloader:
        emb = encoder(batch['input_ids'], batch['attention_mask'], batch['categorical'], batch['numeric'])
        historical_embeddings.append(emb.cpu())
        historical_labels.append(batch['labels'].cpu())

historical_embeddings = torch.cat(historical_embeddings).numpy().astype('float32')
historical_labels = torch.cat(historical_labels).numpy()

print(f"Total embeddings stored: {historical_embeddings.shape[0]}")

# -----------------------------
# STEP 3: Build FAISS index and save
# -----------------------------
print("Building FAISS index...")
index = faiss.IndexFlatL2(historical_embeddings.shape[1])
index.add(historical_embeddings)
print(f"FAISS index size: {index.ntotal}")

faiss.write_index(index, faiss_index_path)
print(f"FAISS index saved at {faiss_index_path}")

# -----------------------------
# STEP 4: Define reusable prediction function
# -----------------------------
label_mapping = dataset.label_mapping

def predict_gl_account_faiss(new_txn: dict):
    text = f"The transaction description is: {new_txn['tran_partclr']}"
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    categorical = torch.tensor([[dataset.cat_vocab[col][new_txn[col]] for col in categorical_cols]], dtype=torch.long)
    numeric_values = [[new_txn[col] for col in numeric_cols]]
    numeric = torch.tensor(dataset.scaler.transform(numeric_values), dtype=torch.float)

    with torch.no_grad():
        query_emb = encoder(input_ids, attention_mask, categorical, numeric).cpu().numpy().astype('float32')

    distances, indices = index.search(query_emb, top_k)
    retrieved_labels = historical_labels[indices[0]]

    pred_label_code = np.bincount(retrieved_labels).argmax()
    return label_mapping[pred_label_code]

# -----------------------------
# DEMO: Predict for a sample transaction
# -----------------------------
sample_txn = {
    'tran_partclr': 'WIRE TRANSFER PAYMENT TO VENDOR ABC',
    'tran_mode': historical_df.iloc[0]['tran_mode'],
    'dr_cr_indctor': historical_df.iloc[0]['dr_cr_indctor'],
    'sal_flag': historical_df.iloc[0]['sal_flag'],
    'tran_amt_in_ac': historical_df.iloc[0]['tran_amt_in_ac']
}

print("Predicted category:", predict_gl_account_faiss(sample_txn))
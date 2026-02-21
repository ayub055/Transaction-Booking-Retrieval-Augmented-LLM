import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer

# Pre-known banking modes so real-data values (UPI, IMPS, etc.) are never <UNK>
KNOWN_TRAN_MODES = [
    'UPI', 'IMPS', 'NEFT', 'RTGS', 'ONLINE', 'WIRE',
    'POS', 'CASH', 'CHECK', 'ATM', 'NACH', 'ECS', 'NULL',
]
KNOWN_DR_CR     = ['D', 'C', 'NULL']
KNOWN_SAL_FLAGS = ['Y', 'N', 'NULL']

# Map real-data column names → canonical names used internally
COLUMN_ALIASES = {
    'tran_type':        'tran_mode',
    'category_of_txn':  'category',
}

# Which known-value lists to pre-seed per canonical column name
_KNOWN_VALUES: dict = {
    'tran_mode':    KNOWN_TRAN_MODES,
    'dr_cr_indctor': KNOWN_DR_CR,
    'sal_flag':     KNOWN_SAL_FLAGS,
}


def _normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns and fill NaN categoricals with 'NULL'."""
    df = df.rename(columns=COLUMN_ALIASES)
    # Fill NaN in string-like columns with the literal string 'NULL'
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna('NULL')
    return df


class TransactionDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer,
                 categorical_cols,
                 numeric_cols,
                 label_col,
                 max_length=128,
                 prompt="The transaction description is: "):
        """
        Args:
            df: pandas DataFrame containing transaction data.
            tokenizer: HuggingFace tokenizer for text encoding.
            categorical_cols: list of categorical feature column names
                              (use canonical names, e.g. 'tran_mode').
            numeric_cols: list of numeric feature column names.
            label_col: column name for category label.
            max_length: max token length for text.
            prompt: fixed prompt to prepend to description.
        """
        df = _normalise_df(df.copy())

        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_col = label_col
        self.max_length = max_length
        self.prompt = prompt

        # Build categorical vocab: pre-seed with known banking values first,
        # then add any extra values found in this specific dataset.
        self.cat_vocab = {}
        for col in categorical_cols:
            known = _KNOWN_VALUES.get(col, [])
            vocab: dict = {'<UNK>': 0}
            # 1. Pre-seed with known values (guarantees UPI, IMPS, etc. are present)
            for v in known:
                if v not in vocab:
                    vocab[v] = len(vocab)
            # 2. Add any unseen values from the actual data
            for val in df[col].unique():
                val_str = str(val) if val is not None else 'NULL'
                if val_str not in vocab:
                    vocab[val_str] = len(vocab)
            self.cat_vocab[col] = vocab

        # Standardize numeric features
        self.scaler = StandardScaler()
        self.numeric_data = self.scaler.fit_transform(df[numeric_cols])

        # Labels
        self.labels = df[label_col].astype('category').cat.codes
        self.label_mapping = dict(enumerate(df[label_col].astype('category').cat.categories))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text processing
        text = f"{self.prompt}{row['tran_partclr']}"
        encoding = self.tokenizer(text, padding='max_length', truncation=True,
                                  max_length=self.max_length, return_tensors='pt')

        categorical_indices = [
            self.cat_vocab[col].get(str(row[col]) if row[col] is not None else 'NULL', 0)
            for col in self.categorical_cols
        ]
        numeric_features = torch.tensor(self.numeric_data[idx], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Metadata for tracking
        metadata = {
            "tran_partclr": row["tran_partclr"],
            "dr_cr_indctor": row["dr_cr_indctor"],
            "tran_amt_in_ac": row["tran_amt_in_ac"],
            "label": row[self.label_col]}

        # Add categorical columns to metadata dynamically
        for col in self.categorical_cols:
            if col in row:
                metadata[col] = row[col]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "categorical": torch.tensor(categorical_indices, dtype=torch.long),
            "numeric": numeric_features,
            "label": label,
            "metadata": metadata}


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    numeric = torch.stack([item['numeric'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'categorical': categorical,
        'numeric': numeric,
        'labels': labels
    }

if __name__ == "__main__":
    df = pd.read_csv('sample_txn.csv')
    print(f'--- Data Read---- {df.shape}')
    
    # Define columns
    categorical_cols = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
    numeric_cols = ['tran_amt_in_ac']
    label_col = 'category'
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TransactionDataset(df, tokenizer, categorical_cols, numeric_cols, label_col)

    for i in range(3):
        sample = dataset[i]
        print(f"--- Sample {i} ---")
        print("Description Tokens:", sample['input_ids'][:10].tolist())  # first 10 token IDs
        print("Attention Mask:", sample['attention_mask'][:10].tolist())
        print("Categorical Indices:", sample['categorical'].tolist())
        print("Numeric Features:", sample['numeric'].tolist())
        print("Label:", sample['label'].item())
        print()

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    
    # Print batch shapes for verification
    print("Batch Shapes:")
    print("input_ids:", batch['input_ids'].shape)
    print("attention_mask:", batch['attention_mask'].shape)
    print("categorical:", batch['categorical'].shape)
    print("numeric:", batch['numeric'].shape)
    print("labels:", batch['labels'].shape)

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer


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
            categorical_cols: list of categorical feature column names.
            numeric_cols: list of numeric feature column names.
            label_col: column name for GL-account label.
            max_length: max token length for text.
            prompt: fixed prompt to prepend to description.
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.label_col = label_col
        self.max_length = max_length
        self.prompt = prompt

        # Build categorical vocab mapping
        self.cat_vocab = {col: {val: idx for idx, val in enumerate(df[col].unique())}
                          for col in categorical_cols}

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

        categorical_indices = [self.cat_vocab[col][row[col]] for col in self.categorical_cols]
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

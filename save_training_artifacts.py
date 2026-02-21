"""
Save training artifacts (vocabularies, scalers, label mappings) from training data.

Run this once with your full labeled training dataset (35K+ transactions across
43 specific categories) to create the artifacts used by inference.

Key behaviours
--------------
- "other" rows are EXCLUDED from the label mapping and vocab fitting (OOD label).
- Column aliases are applied automatically so both rgs.csv and sample_txn.csv
  column naming conventions work out of the box.
- The categorical vocabulary is pre-seeded with all known Indian banking modes
  (UPI, IMPS, NEFT, RTGS ...) so they are never mapped to <UNK> at inference.

Usage
-----
    python save_training_artifacts.py [--csv PATH] [--ood-label LABEL]
    python save_training_artifacts.py --csv data/golden_35k.csv --ood-label other
"""
import argparse
import json
import os
import pickle

import pandas as pd
from transformers import BertTokenizer

from src.data_loader import TransactionDataset, _normalise_df


# ---------------------------------------------------------------------------
# Defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH       = "./data/sample_txn.csv"
DEFAULT_CATEGORICAL    = ['tran_mode', 'dr_cr_indctor', 'sal_flag']
DEFAULT_NUMERIC        = ['tran_amt_in_ac']
DEFAULT_LABEL_COL      = 'category'
DEFAULT_OOD_LABEL      = 'other'
DEFAULT_BERT_MODEL     = 'bert-base-uncased'
DEFAULT_TEXT_PROJ_DIM  = 256
DEFAULT_FINAL_DIM      = 256
DEFAULT_DROPOUT        = 0.1
DEFAULT_OUTPUT_DIR     = "./training_artifacts"


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Build and save training artifacts for the FusionEncoder pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--csv',         default=DEFAULT_CSV_PATH,       help='Path to labeled training CSV')
    parser.add_argument('--label-col',   default=DEFAULT_LABEL_COL,      help='Column name for category labels')
    parser.add_argument('--ood-label',   default=DEFAULT_OOD_LABEL,      help='Label string to treat as OOD (excluded from training)')
    parser.add_argument('--bert-model',  default=DEFAULT_BERT_MODEL,     help='HuggingFace BERT model name or path')
    parser.add_argument('--proj-dim',    type=int, default=DEFAULT_TEXT_PROJ_DIM, help='Text projection dimension')
    parser.add_argument('--final-dim',   type=int, default=DEFAULT_FINAL_DIM,     help='Final embedding dimension')
    parser.add_argument('--dropout',     type=float, default=DEFAULT_DROPOUT,     help='Dropout probability')
    parser.add_argument('--output-dir',  default=DEFAULT_OUTPUT_DIR,     help='Directory to save artifacts')
    args = parser.parse_args()

    categorical_cols = DEFAULT_CATEGORICAL
    numeric_cols     = DEFAULT_NUMERIC
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load CSV and apply column aliases
    # -----------------------------------------------------------------------
    print(f"Loading training data from: {args.csv}")
    df = pd.read_csv(args.csv)
    df = _normalise_df(df)    # applies tran_type→tran_mode, category_of_txn→category, etc.
    print(f"  Raw rows: {len(df)}")

    # -----------------------------------------------------------------------
    # 2. Exclude OOD rows from training artifacts
    # -----------------------------------------------------------------------
    ood_mask = df[args.label_col] == args.ood_label
    ood_count = ood_mask.sum()
    df_train = df[~ood_mask].reset_index(drop=True)
    print(f"  OOD rows excluded ('{args.ood_label}'): {ood_count}")
    print(f"  Training rows (in-distribution):        {len(df_train)}")

    if len(df_train) == 0:
        raise ValueError(
            f"No training rows remaining after excluding '{args.ood_label}'. "
            "Check --csv and --ood-label arguments."
        )

    # -----------------------------------------------------------------------
    # 3. Build TransactionDataset (fits vocab + scaler on in-distribution data)
    # -----------------------------------------------------------------------
    print("\nFitting vocabularies and scaler...")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    dataset   = TransactionDataset(
        df_train, tokenizer, categorical_cols, numeric_cols, args.label_col,
    )

    # -----------------------------------------------------------------------
    # 4. Report per-class counts (useful for spotting heavy imbalance)
    # -----------------------------------------------------------------------
    class_counts = {}
    for code, category in dataset.label_mapping.items():
        count = (df_train[args.label_col] == category).sum()
        class_counts[category] = int(count)

    # -----------------------------------------------------------------------
    # 5. Assemble and save artifacts
    # -----------------------------------------------------------------------
    artifacts = {
        # Preprocessing
        'cat_vocab':      dataset.cat_vocab,
        'scaler':         dataset.scaler,
        'label_mapping':  dataset.label_mapping,          # {int_code: category_str}
        'label_inverse':  {v: k for k, v in dataset.label_mapping.items()},  # {category_str: int_code}

        # Model architecture
        'categorical_dims': [len(dataset.cat_vocab[col]) for col in categorical_cols],

        # Training metadata (stored in golden record metadata for inference)
        'transaction_metadata': df_train.to_dict('records'),

        # Class distribution (for imbalance analysis and weight computation)
        'class_counts': class_counts,

        # Full config (used by GoldenRecordIndexer and TransactionInferencePipeline)
        'config': {
            'categorical_cols': categorical_cols,
            'numeric_cols':     numeric_cols,
            'label_col':        args.label_col,
            'ood_label':        args.ood_label,
            'bert_model':       args.bert_model,
            'text_proj_dim':    args.proj_dim,
            'final_dim':        args.final_dim,
            'dropout':          args.dropout,
            'num_categories':   len(dataset.label_mapping),
        }
    }

    artifacts_path = os.path.join(args.output_dir, "training_artifacts.pkl")
    with open(artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f)

    # -----------------------------------------------------------------------
    # 6. Save human-readable JSON config
    # -----------------------------------------------------------------------
    config_path = os.path.join(args.output_dir, "model_config.json")
    config_dict = dict(artifacts['config'])
    config_dict['categorical_dims'] = artifacts['categorical_dims']
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    # -----------------------------------------------------------------------
    # 7. Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Artifacts saved")
    print("=" * 60)
    print(f"  {artifacts_path}")
    print(f"  {config_path}")
    print()

    print("=" * 60)
    print("Categorical Vocabularies")
    print("=" * 60)
    for col in categorical_cols:
        vocab_size = len(dataset.cat_vocab[col])
        print(f"  {col}: {vocab_size} values (incl. <UNK>)")
    print()

    print("=" * 60)
    print(f"Label Categories ({len(dataset.label_mapping)} in-distribution)")
    print("=" * 60)
    sorted_cats = sorted(class_counts.items(), key=lambda x: -x[1])
    for category, count in sorted_cats:
        bar = '#' * min(40, count // max(1, max(class_counts.values()) // 40))
        print(f"  {category:<35} {count:>6}  {bar}")
    minority = [(c, n) for c, n in sorted_cats if n <= 500]
    if minority:
        print(f"\n  Minority classes (≤500 samples): {len(minority)}")
        print("  These benefit most from SupCon + balanced sampling.")
    print()

    print("=" * 60)
    print("Next steps")
    print("=" * 60)
    print(f"""
  1. Train the FusionEncoder:
       python train.py --config experiments_with_validation.yaml

  2. Build the stratified FAISS index (supply --ood-csv for better OOD threshold):
       python run_inference.py \\
           --csv data/golden_records.csv \\
           --ood-csv data/rgs.csv \\
           --artifacts {artifacts_path}

  3. Tag new transactions:
       python run_inference.py \\
           --skip-build \\
           --input-csv data/new_txns.csv \\
           --output-csv results.csv
""")


if __name__ == "__main__":
    main()

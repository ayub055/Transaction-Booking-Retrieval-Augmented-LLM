"""
Run Inference Pipeline - Transaction Tagging with RAG

Usage examples:

  # Build index and tag transactions from a CSV:
  python run_inference.py --csv data/golden_records.csv --input-csv data/new_txns.csv

  # Use real "other"-labeled data for OOD threshold calibration:
  python run_inference.py --csv data/golden_records.csv --ood-csv data/rgs.csv \
      --input-csv data/new_txns.csv

  # Enable LLM fallback for uncertain / OOD transactions:
  python run_inference.py --csv data/golden_records.csv --ood-csv data/rgs.csv \
      --input-csv data/new_txns.csv --use-llm --llm-backend ollama --llm-model llama3.1

  # Skip rebuilding the index if it already exists:
  python run_inference.py --skip-build --input-csv data/new_txns.csv
"""

import argparse
import os

import pandas as pd

from src.inference_pipeline import (
    GoldenRecordIndexer,
    TransactionInferencePipeline,
)

# Column aliases: handles both rgs.csv and sample_txn.csv column naming
_COLUMN_ALIASES = {
    'tran_type':        'tran_mode',
    'tran_partlcr':     'tran_partclr',   # typo variant in some CSVs
    'category_of_txn':  'category',
}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _normalise_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column aliases so both rgs.csv and sample_txn.csv column names work."""
    return df.rename(columns=_COLUMN_ALIASES)


def load_transactions_from_csv(csv_path: str):
    """Load a CSV of transactions into a list of dicts for the pipeline."""
    print(f"\nLoading transactions from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _normalise_input_df(df)

    # Accept either 'tran_partclr' or 'tran_partlcr' (CSV header typo common)
    desc_col = 'tran_partclr' if 'tran_partclr' in df.columns else 'tran_partlcr'

    required = ['tran_amt_in_ac']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} transactions")
    print(f"Columns: {list(df.columns)}")

    transactions = []
    for _, row in df.iterrows():
        txn = {
            'cust_id':        row.get('cust_id', ''),
            'tran_date':      row.get('tran_date', ''),
            'tran_partclr':   row.get(desc_col, ''),
            'tran_mode':      row.get('tran_mode', 'NULL'),
            'dr_cr_indctor':  row.get('dr_cr_indctor', 'NULL'),
            'sal_flag':       row.get('sal_flag', 'NULL'),
            'tran_amt_in_ac': float(row.get('tran_amt_in_ac', 0.0)),
        }
        transactions.append(txn)

    return transactions, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run transaction inference pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Paths ----
    parser.add_argument('--artifacts', default='training_artifacts/training_artifacts.pkl',
                        help='Path to training artifacts (.pkl)')
    parser.add_argument('--model',
                        default='experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth',
                        help='Path to trained FusionEncoder checkpoint (.pth)')
    parser.add_argument('--csv', default='data/sample_txn.csv',
                        help='CSV with labeled golden records (43 categories, no "other")')
    parser.add_argument('--ood-csv', default=None,
                        help='CSV with "other"-labeled transactions for OOD threshold '
                             'calibration (e.g. data/rgs.csv). If omitted, uses 95th-percentile '
                             'of within-corpus distances.')
    parser.add_argument('--index', default='golden_records.faiss',
                        help='Path to FAISS index file')
    parser.add_argument('--input-csv', default=None,
                        help='CSV with transactions to tag (columns: tran_partclr, '
                             'tran_amt_in_ac, tran_mode, dr_cr_indctor, ...)')
    parser.add_argument('--output-csv', default='inference_results.csv',
                        help='Output CSV file for prediction results')

    # ---- Index build settings ----
    parser.add_argument('--skip-build', action='store_true',
                        help='Skip index build if .faiss file already exists')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for encoding during index build')
    parser.add_argument('--index-type', default='HNSW',
                        choices=['L2', 'IP', 'HNSW', 'IVF'],
                        help='FAISS index type')
    parser.add_argument('--max-per-class', type=int, default=500,
                        help='Max examples per class in stratified FAISS index. '
                             'Majority classes exceeding this are reduced to diverse '
                             'representatives to prevent density bias in k-NN retrieval.')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 precision')

    # ---- Inference settings ----
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of nearest neighbors to retrieve')
    parser.add_argument('--confidence-threshold', type=float, default=0.75,
                        help='Confidence below which LLM fallback is triggered '
                             '(also used for OOD flag)')
    parser.add_argument('--knn-weight', type=float, default=0.6,
                        help='Weight α for k-NN votes in combined score '
                             '(1-α given to prototype similarity)')

    # ---- LLM fallback ----
    parser.add_argument('--use-llm', action='store_true',
                        help='Enable LLM few-shot fallback for uncertain / OOD transactions')
    parser.add_argument('--llm-backend', default='ollama',
                        choices=['anthropic', 'openai', 'ollama'],
                        help='LLM backend for the few-shot classifier')
    parser.add_argument('--llm-model', default=None,
                        help='LLM model name (defaults: anthropic→claude-haiku-4-5-20251001, '
                             'openai→gpt-4o-mini, ollama→llama3.1)')

    args = parser.parse_args()

    use_fp16 = not args.no_fp16

    # ========================================================================
    # STEP 1: Build Golden Record Index
    # ========================================================================
    if not args.skip_build or not os.path.exists(args.index):
        print("\n" + "=" * 80)
        print("BUILDING GOLDEN RECORD INDEX")
        print("=" * 80)

        indexer = GoldenRecordIndexer(
            artifacts_path=args.artifacts,
            model_path=args.model,
            use_fp16=use_fp16,
        )
        n_vectors, dim = indexer.build_index(
            csv_path=args.csv,
            output_path=args.index,
            batch_size=args.batch_size,
            index_type=args.index_type,
            max_per_class=args.max_per_class,
            ood_csv_path=args.ood_csv,
        )
        print(f"\nIndex built: {n_vectors} vectors, dim={dim}")
        print(f"  Index type:      {args.index_type}")
        print(f"  Max/class:       {args.max_per_class}")
        print(f"  OOD calibration: {'from ' + args.ood_csv if args.ood_csv else '95th-percentile fallback'}")
        print(f"  FP16:            {use_fp16}")
    else:
        print(f"\nSkipping index build (already exists: {args.index})")

    # ========================================================================
    # STEP 2: Initialize Inference Pipeline
    # ========================================================================
    print("\n" + "=" * 80)
    print("INITIALIZING INFERENCE PIPELINE")
    print("=" * 80)

    pipeline = TransactionInferencePipeline(
        artifacts_path=args.artifacts,
        model_path=args.model,
        index_path=args.index,
        use_fp16=use_fp16,
        knn_weight=args.knn_weight,
        confidence_threshold=args.confidence_threshold,
    )

    # ========================================================================
    # STEP 3: (Optional) Load LLM classifier
    # ========================================================================
    llm_clf = None
    if args.use_llm:
        try:
            from src.llm_classifier import LLMClassifier
            backend_kwargs = {}
            if args.llm_model:          # don't pass None — let backend use its default
                backend_kwargs['model'] = args.llm_model
            llm_clf = LLMClassifier(
                pipeline,               # required: pipeline reference for fallback encoding
                categories=list(pipeline.label_mapping.values()),
                provider=args.llm_backend,
                confidence_threshold=args.confidence_threshold,
                **backend_kwargs,
            )
            print(f"\nLLM fallback enabled: backend={args.llm_backend}, "
                  f"model={llm_clf.llm.model}")
        except Exception as exc:
            print(f"\n[WARNING] Could not load LLM classifier: {exc}")
            print("         Proceeding without LLM fallback.\n")

    # ========================================================================
    # STEP 4: Load transactions and run predictions
    # ========================================================================
    if not args.input_csv:
        print("\nNo --input-csv provided. Index built successfully.")
        print("=" * 80)
        print("INFERENCE COMPLETE")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("RUNNING PREDICTIONS")
    print("=" * 80)

    test_transactions, _ = load_transactions_from_csv(args.input_csv)
    print(f"Predicting {len(test_transactions)} transactions "
          f"(top-k={args.top_k}, confidence_threshold={args.confidence_threshold})\n")

    batch_results = pipeline.predict_batch(test_transactions, top_k=args.top_k)

    # LLM pass for transactions that need it
    llm_override_count = 0
    if llm_clf is not None:
        for i, (txn, result) in enumerate(zip(test_transactions, batch_results)):
            if result.get('needs_llm', False):
                try:
                    llm_result = llm_clf.classify(txn, result)
                    batch_results[i]['predicted_category'] = llm_result['predicted_category']
                    batch_results[i]['confidence']         = llm_result['confidence']
                    batch_results[i]['llm_reasoning']      = llm_result.get('reasoning', '')
                    batch_results[i]['llm_overridden']     = True
                    llm_override_count += 1
                except Exception as exc:
                    batch_results[i]['llm_overridden'] = False
                    print(f"  [LLM] Transaction {i} failed: {exc}")

    # ========================================================================
    # STEP 5: Print summary table
    # ========================================================================
    ood_count  = sum(1 for r in batch_results if r.get('is_ood', False))
    llm_needed = sum(1 for r in batch_results if r.get('needs_llm', False))
    avg_conf   = sum(r['confidence'] for r in batch_results) / max(len(batch_results), 1)

    print(f"\n{'#':<5} {'Description':<42} {'Category':<24} {'Conf':>6} {'OOD':>4} {'LLM':>4}")
    print("-" * 90)
    display_limit = min(20, len(test_transactions))
    for i, (txn, result) in enumerate(zip(test_transactions[:display_limit],
                                          batch_results[:display_limit]), 1):
        desc     = str(txn.get('tran_partclr', ''))[:39] + ('…' if len(str(txn.get('tran_partclr', ''))) > 39 else '')
        category = result['predicted_category'][:23]
        conf     = f"{result['confidence']:.0%}"
        ood_flag = 'Y' if result.get('is_ood') else '-'
        llm_flag = 'Y' if result.get('llm_overridden') else ('-' if result.get('needs_llm') else ' ')
        print(f"{i:<5} {desc:<42} {category:<24} {conf:>6} {ood_flag:>4} {llm_flag:>4}")

    if len(test_transactions) > display_limit:
        print(f"  ... and {len(test_transactions) - display_limit} more transactions")

    print(f"\nSummary:")
    print(f"  Total:              {len(batch_results)}")
    print(f"  OOD ('other'):      {ood_count} ({ood_count/len(batch_results):.1%})")
    print(f"  Needs LLM:          {llm_needed} ({llm_needed/len(batch_results):.1%})")
    if llm_clf:
        print(f"  LLM overrides:      {llm_override_count}")
    print(f"  Avg confidence:     {avg_conf:.1%}")

    # ========================================================================
    # STEP 6: Export results CSV
    # ========================================================================
    print(f"\nExporting to {args.output_csv} ...")
    rows = []
    for txn, result in zip(test_transactions, batch_results):
        # Top-3 retrieved similar transaction descriptions
        similar = result.get('similar_transactions', [])
        top3_desc = ' | '.join(
            s['transaction'].get('description', 'N/A')[:30] for s in similar[:3]
        )
        top3_cats = ' | '.join(
            s.get('label', 'N/A') for s in similar[:3]
        )
        rows.append({
            'cust_id':              txn.get('cust_id', ''),
            'tran_date':            txn.get('tran_date', ''),
            'description':          txn.get('tran_partclr', ''),
            'amount':               txn.get('tran_amt_in_ac', ''),
            'mode':                 txn.get('tran_mode', ''),
            'dr_cr':                txn.get('dr_cr_indctor', ''),
            'predicted_category':   result['predicted_category'],
            'confidence':           round(result['confidence'], 4),
            'is_ood':               result.get('is_ood', False),
            'ood_distance':         round(result.get('ood_distance', 0.0), 4),
            'needs_llm':            result.get('needs_llm', False),
            'llm_overridden':       result.get('llm_overridden', False),
            'top3_retrieved_desc':  top3_desc,
            'top3_retrieved_cats':  top3_cats,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(results_df)} rows → {args.output_csv}")
    print(f"\nSample (first 5):")
    print(results_df[['description', 'predicted_category', 'confidence', 'is_ood']].head().to_string(index=False))

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"  FAISS index:  {args.index}")
    print(f"  Metadata:     {args.index.replace('.faiss', '_metadata.pkl')}")
    print(f"  Results:      {args.output_csv}")
    print()


if __name__ == "__main__":
    main()

"""
Run Inference Pipeline - Transaction Tagging with RAG

This script demonstrates how to use the inference pipeline for transaction tagging.

Usage:
    # Build index and run inference
    python run_inference.py

    # Skip index building if already exists
    python run_inference.py --skip-build

    # Use custom paths
    python run_inference.py --model path/to/model.pth --index path/to/index.faiss
"""

import argparse
import os
import pandas as pd
from src.inference_pipeline import (
    GoldenRecordIndexer,
    TransactionInferencePipeline,
    print_prediction_result
)


def main():
    parser = argparse.ArgumentParser(description='Run transaction inference pipeline')
    parser.add_argument(
        '--artifacts',
        default='training_artifacts/training_artifacts.pkl',
        help='Path to training artifacts'
    )
    parser.add_argument(
        '--model',
        default='experiments/tagger_proj256_final256_freeze-gradual_bs2048_lr5.66e-05/fusion_encoder_best.pth',
        help='Path to trained model'
    )
    parser.add_argument(
        '--csv',
        default='data/sample_txn.csv',
        help='Path to CSV with golden records'
    )
    parser.add_argument(
        '--index',
        default='golden_records.faiss',
        help='Path to FAISS index'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of similar transactions to retrieve'
    )
    parser.add_argument(
        '--skip-build',
        action='store_true',
        help='Skip building index if it exists'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for encoding during index building'
    )

    args = parser.parse_args()

    # ========================================================================
    # STEP 1: Build Golden Record Index
    # ========================================================================
    if not args.skip_build or not os.path.exists(args.index):
        print("\n" + "="*80)
        print("BUILDING GOLDEN RECORD INDEX")
        print("="*80 + "\n")

        indexer = GoldenRecordIndexer(
            artifacts_path=args.artifacts,
            model_path=args.model
        )

        indexer.build_index(
            csv_path=args.csv,
            output_path=args.index,
            batch_size=args.batch_size
        )
    else:
        print(f"\nSkipping index build (already exists at {args.index})")

    # ========================================================================
    # STEP 2: Initialize Pipeline
    # ========================================================================
    print("\n" + "="*80)
    print("INITIALIZING INFERENCE PIPELINE")
    print("="*80)

    pipeline = TransactionInferencePipeline(
        artifacts_path=args.artifacts,
        model_path=args.model,
        index_path=args.index
    )

    # ========================================================================
    # STEP 3: Run Sample Predictions
    # ========================================================================
    print("\n" + "="*80)
    print("RUNNING SAMPLE PREDICTIONS")
    print("="*80)

    # Define test cases covering different transaction types
    test_transactions = [
        {
            'tran_partclr': 'WALMART GROCERY PURCHASE',
            'tran_mode': 'POS',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 125.50
        },
        {
            'tran_partclr': 'MONTHLY SALARY CREDIT',
            'tran_mode': 'NEFT',
            'dr_cr_indctor': 'C',
            'sal_flag': 'Y',
            'tran_amt_in_ac': 5500.00
        },
        {
            'tran_partclr': 'ATM CASH WITHDRAWAL',
            'tran_mode': 'ATM',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 500.00
        },
        {
            'tran_partclr': 'ELECTRIC BILL PAYMENT ONLINE',
            'tran_mode': 'IMPS',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 85.00
        },
        {
            'tran_partclr': 'TRANSFER TO SAVINGS ACCOUNT',
            'tran_mode': 'NEFT',
            'dr_cr_indctor': 'D',
            'sal_flag': 'N',
            'tran_amt_in_ac': 1000.00
        }
    ]

    # Predict for each transaction
    for i, txn in enumerate(test_transactions, 1):
        print(f"\n{'#'*80}")
        print(f"TEST CASE {i}/{len(test_transactions)}")
        print(f"{'#'*80}")

        result = pipeline.predict(txn, top_k=args.top_k)
        print_prediction_result(result, txn, args.top_k)

    # ========================================================================
    # STEP 4: Batch Prediction Demo
    # ========================================================================
    print("\n" + "="*80)
    print("BATCH PREDICTION SUMMARY")
    print("="*80 + "\n")

    batch_results = pipeline.predict_batch(test_transactions, top_k=args.top_k)

    print(f"{'#':<4} {'Description':<50} {'Predicted Category':<20} {'Confidence':<10}")
    print("-"*90)

    for i, (txn, result) in enumerate(zip(test_transactions, batch_results), 1):
        desc = txn['tran_partclr'][:47] + '...' if len(txn['tran_partclr']) > 50 else txn['tran_partclr']
        category = result['predicted_category']
        confidence = f"{result['confidence']:.1%}"

        print(f"{i:<4} {desc:<50} {category:<20} {confidence:<10}")

    # ========================================================================
    # STEP 5: Export Results
    # ========================================================================
    print("\n" + "="*80)
    print("EXPORTING RESULTS")
    print("="*80 + "\n")

    # Create results dataframe
    results_data = []
    for txn, result in zip(test_transactions, batch_results):
        results_data.append({
            'description': txn['tran_partclr'],
            'amount': txn['tran_amt_in_ac'],
            'mode': txn['tran_mode'],
            'dr_cr': txn['dr_cr_indctor'],
            'predicted_category': result['predicted_category'],
            'confidence': result['confidence'],
            'top_k_matches': ', '.join(result['top_k_labels'][:3])
        })

    results_df = pd.DataFrame(results_data)
    output_csv = 'inference_results.csv'
    results_df.to_csv(output_csv, index=False)

    print(f"Results exported to: {output_csv}")
    print(f"Total predictions: {len(results_df)}")
    print(f"\nSample results:")
    print(results_df.to_string(index=False))

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    print(f"\nFiles generated:")
    print(f"  - {args.index} (FAISS index)")
    print(f"  - {args.index.replace('.faiss', '_metadata.pkl')} (metadata)")
    print(f"  - {output_csv} (prediction results)")
    print("\n")


if __name__ == "__main__":
    main()

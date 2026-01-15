"""
Example usage demonstrating the two key improvements:
1. Embedding Normalization (#4)
2. Validation Split + Early Stopping (#10)

Run this script to see how the improvements work together.
"""

import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from src.data_loader import TransactionDataset, collate_fn
from src.fusion_encoder import FusionEncoder
from src.validation import EarlyStopping, evaluate_retrieval_metrics, compute_validation_loss

def demo_embedding_normalization():
    """
    Demo #4: Embedding Normalization

    Shows how L2-normalized embeddings improve metric learning by ensuring
    all embeddings lie on a unit hypersphere, making distance computations
    more meaningful and stable.
    """
    print("\n" + "="*70)
    print("DEMO #4: EMBEDDING NORMALIZATION")
    print("="*70)

    # Create a simple dummy dataset
    print("\n1. Creating FusionEncoder with normalization=True (default)...")
    encoder_normalized = FusionEncoder(
        categorical_dims=[5, 3],
        numeric_dim=1,
        text_proj_dim=128,
        final_dim=128,
        normalize_embeddings=True  # NEW: Enable normalization
    )

    print("2. Creating FusionEncoder with normalization=False (old behavior)...")
    encoder_unnormalized = FusionEncoder(
        categorical_dims=[5, 3],
        numeric_dim=1,
        text_proj_dim=128,
        final_dim=128,
        normalize_embeddings=False  # Disable normalization
    )

    # Create dummy input
    batch_size = 4
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    categorical = torch.randint(0, 5, (batch_size, 2))
    numeric = torch.randn(batch_size, 1)

    print("\n3. Computing embeddings...")
    with torch.no_grad():
        emb_normalized = encoder_normalized(input_ids, attention_mask, categorical, numeric)
        emb_unnormalized = encoder_unnormalized(input_ids, attention_mask, categorical, numeric)

    # Check norms
    norms_normalized = torch.norm(emb_normalized, p=2, dim=1)
    norms_unnormalized = torch.norm(emb_unnormalized, p=2, dim=1)

    print("\n4. Results:")
    print(f"   Normalized embeddings L2 norms:   {norms_normalized.tolist()}")
    print(f"   Unnormalized embeddings L2 norms: {norms_unnormalized.tolist()}")

    print("\n5. Analysis:")
    print(f"   ✓ Normalized: All norms ≈ 1.0 (stable, consistent scale)")
    print(f"   ✗ Unnormalized: Varying norms (unstable, inconsistent scale)")
    print("\n   Benefits of normalization:")
    print("   • Embeddings lie on unit hypersphere")
    print("   • Cosine similarity ≈ Euclidean distance")
    print("   • Better triplet loss convergence")
    print("   • More stable training dynamics")


def demo_validation_and_early_stopping():
    """
    Demo #10: Validation Split + Early Stopping

    Shows how to properly split data, track validation metrics,
    and use early stopping to prevent overfitting.
    """
    print("\n" + "="*70)
    print("DEMO #10: VALIDATION SPLIT + EARLY STOPPING")
    print("="*70)

    print("\n1. Creating synthetic transaction data...")
    # Create a synthetic dataset for demonstration
    num_samples = 1000
    dummy_df = pd.DataFrame({
        'tran_partclr': [f'Transaction {i}' for i in range(num_samples)],
        'tran_mode': ['CASH' if i % 2 == 0 else 'ONLINE' for i in range(num_samples)],
        'dr_cr_indctor': ['D' if i % 3 == 0 else 'C' for i in range(num_samples)],
        'sal_flag': ['Y' if i % 4 == 0 else 'N' for i in range(num_samples)],
        'tran_amt_in_ac': [100.0 + i * 10 for i in range(num_samples)],
        'category': [f'CAT_{i % 10}' for i in range(num_samples)]
    })

    print(f"   Total samples: {len(dummy_df)}")

    # Split into train/validation
    print("\n2. Splitting data with stratification...")
    train_df, val_df = train_test_split(
        dummy_df,
        test_size=0.15,
        random_state=42,
        stratify=dummy_df['category']
    )

    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples:   {len(val_df)}")

    # Show class distribution is maintained
    print("\n3. Verifying stratified split (class distribution):")
    train_dist = train_df['category'].value_counts(normalize=True).sort_index()
    val_dist = val_df['category'].value_counts(normalize=True).sort_index()

    print("\n   Class | Train % | Val %")
    print("   " + "-"*30)
    for cls in sorted(dummy_df['category'].unique()):
        print(f"   {cls:6s} | {train_dist.get(cls, 0)*100:6.2f}% | {val_dist.get(cls, 0)*100:6.2f}%")

    print("\n4. Setting up Early Stopping...")
    early_stopping = EarlyStopping(
        patience=5,
        min_delta=0.001,
        mode='max',  # We want to maximize recall@5
        verbose=True
    )

    print("\n5. Simulating training with validation...")
    print("\n   Simulated validation Recall@5 over epochs:")

    # Simulate improving then plateauing metrics
    simulated_recall5 = [0.45, 0.52, 0.61, 0.68, 0.72, 0.74, 0.745, 0.746, 0.746, 0.745, 0.744, 0.743]

    for epoch, recall5 in enumerate(simulated_recall5, start=1):
        print(f"\n   Epoch {epoch}: Recall@5 = {recall5:.3f}")

        should_stop = early_stopping(recall5, epoch)

        if should_stop:
            print(f"\n   🛑 Early stopping at epoch {epoch}!")
            print(f"   Best score: {early_stopping.best_score:.3f} at epoch {early_stopping.best_epoch}")
            break

    print("\n6. Benefits of validation + early stopping:")
    print("   ✓ Prevents overfitting to training data")
    print("   ✓ Saves training time (stops when no improvement)")
    print("   ✓ Automatically finds best model checkpoint")
    print("   ✓ Tracks multiple metrics (accuracy, recall@K, MRR)")
    print("   ✓ Stratified split maintains class distribution")


def show_training_command():
    """Show how to run training with the new improvements."""
    print("\n" + "="*70)
    print("HOW TO USE IN YOUR TRAINING")
    print("="*70)

    print("\n1. Train with the updated script:")
    print("\n   python -m src.train_multi_expt --config experiments_with_validation.yaml --format yaml")

    print("\n2. Key parameters in config file:")
    print("""
   val_split: 0.15          # Use 15% of data for validation
   patience: 5              # Stop if no improvement for 5 epochs
   min_delta: 0.001         # Minimum improvement threshold
   epochs: 20               # Max epochs (may stop early)
    """)

    print("\n3. What you'll see during training:")
    print("""
   • Train loss per epoch
   • Validation loss per epoch
   • Validation metrics: Accuracy, Recall@1/5/10, MRR
   • Early stopping notifications
   • Best model automatically saved
    """)

    print("\n4. Outputs generated:")
    print("""
   experiments/
   └── tagger_proj256_final256_freeze-freeze_lr1e-05/
       ├── logs/
       │   └── training_logs.json         # All metrics
       ├── plots/
       │   ├── loss_curve.png             # Train/val loss
       │   ├── validation_metrics.png     # Recall, accuracy curves
       │   └── grad_norms.png
       └── fusion_encoder_best.pth        # Best model checkpoint
    """)

    print("\n5. Loading the best model for inference:")
    print("""
   checkpoint = torch.load('experiments/.../fusion_encoder_best.pth')
   encoder.load_state_dict(checkpoint['model_state_dict'])

   # Access best metrics:
   best_recall5 = checkpoint['val_recall5']
   best_epoch = checkpoint['epoch']
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TRANSACTION TAGGER: KEY IMPROVEMENTS DEMO")
    print("="*70)
    print("\nThis demo shows two critical improvements for metric learning:")
    print("  #4:  Embedding Normalization")
    print("  #10: Validation Split + Early Stopping")

    # Run demos
    demo_embedding_normalization()
    demo_validation_and_early_stopping()
    show_training_command()

    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the updated code in src/fusion_encoder.py")
    print("  2. Review validation utilities in src/validation.py")
    print("  3. Run training with experiments_with_validation.yaml")
    print("  4. Monitor validation curves in the plots/ directory")
    print("\n")

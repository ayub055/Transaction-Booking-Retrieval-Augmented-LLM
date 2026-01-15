import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import json
import yaml
import argparse

from src.data_loader import TransactionDataset
from src.fusion_encoder import FusionEncoder
from src.plotting import plot_loss_curves, plot_grad_norms, plot_embedding_projection, plot_validation_curves
from src.validation import EarlyStopping, evaluate_validation_metrics, print_validation_report


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical = torch.stack([item['categorical'] for item in batch])
    numeric = torch.stack([item['numeric'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    metadata = [item['metadata'] for item in batch]  

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'categorical': categorical,
        'numeric': numeric,
        'labels': labels,
        'metadata': metadata
    }


# ---------------------- Freeze Strategy ----------------------
def apply_freeze_strategy(encoder, strategy, epoch=None):
    if strategy == "freeze":
        for param in encoder.bert.parameters(): param.requires_grad = False
    elif strategy == "gradual":
        total_layers = len(encoder.bert.encoder.layer)
        layers_to_unfreeze = min(epoch, total_layers)
        print(f'Total Layers : {total_layers} | # Unforzen Layers gradually : {layers_to_unfreeze}')
        for i in range(layers_to_unfreeze):
            for param in encoder.bert.encoder.layer[i].parameters(): param.requires_grad = True
    elif strategy == "full":
        for param in encoder.bert.parameters(): param.requires_grad = True

def sample_triplets(embeddings, labels, margin=0.5, max_triplets=64):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    triplets = []
    batch_size = len(labels)

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        pos_candidates = (labels == anchor_label).nonzero(as_tuple=True)[0]
        pos_candidates = pos_candidates[pos_candidates != anchor_idx]
        if len(pos_candidates) == 0: continue

        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
        neg_candidates = (labels != anchor_label).nonzero(as_tuple=True)[0]
        neg_idx = neg_candidates[torch.randint(0, len(neg_candidates), (1,)).item()]

        triplets.append((anchor_idx, pos_idx.item(), neg_idx.item()))
        if len(triplets) >= max_triplets: break

    return triplets

# ---------------------- Training  ----------------------
def run_experiment(config):
    # Gradient accumulation setup
    accumulation_steps = config.get("accumulation_steps", 1)
    base_batch_size = config.get("base_batch_size", 256)
    effective_batch_size = config["batch_size"] * accumulation_steps

    # Learning rate scaling: scale with sqrt of batch size ratio
    if "base_lr" in config:
        batch_ratio = effective_batch_size / base_batch_size
        scaled_lr = config["base_lr"] * (batch_ratio ** 0.5)
        config["lr"] = scaled_lr
        print(f"LR Scaling: base_lr={config['base_lr']:.2e}, batch_ratio={batch_ratio:.2f}, scaled_lr={scaled_lr:.2e}")

    # Prepare directories
    exp_name = f"tagger_proj{config['text_proj_dim']}_final{config['final_dim']}_freeze-{config['freeze_strategy']}_bs{effective_batch_size}_lr{config['lr']:.2e}"
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    # os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {exp_name} on {device}")
    print(f"Batch config: per_device={config['batch_size']}, accumulation={accumulation_steps}, effective={effective_batch_size}")

    # Load data and split into train/validation
    df = pd.read_csv(config["csv_path"]).sample(n=config.get("sample_size", 40000), random_state=42).reset_index(drop=True)

    # Stratified split to maintain class distribution
    val_split = config.get("val_split", 0.15)
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        random_state=42,
        stratify=df[config["label_col"]]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    # Create train and validation datasets
    train_dataset = TransactionDataset(train_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"])
    val_dataset = TransactionDataset(val_df, tokenizer, config["categorical_cols"], config["numeric_cols"], config["label_col"])

    # Copy vocab and scaler from train to val dataset for consistency
    val_dataset.cat_vocab = train_dataset.cat_vocab
    val_dataset.scaler = train_dataset.scaler
    val_dataset.numeric_data = val_dataset.scaler.transform(val_df[config["numeric_cols"]])
    val_dataset.label_mapping = train_dataset.label_mapping

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    categorical_dims = [len(train_dataset.cat_vocab[col]) for col in config["categorical_cols"]]

    # ENCODER INSTANT
    encoder = FusionEncoder(
        bert_model_name=config["bert_model"],
        categorical_dims=categorical_dims,
        numeric_dim=len(config["numeric_cols"]),
        text_proj_dim=config["text_proj_dim"],
        final_dim=config["final_dim"],
        p=config.get("dropout", 0.1)).to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, encoder.parameters()), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    triplet_loss_fn = nn.TripletMarginLoss(margin=config["margin"])  # LOSS

    # Mixed precision training setup
    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (AMP)")

    # Early stopping setup
    early_stopping = EarlyStopping(
        patience=config.get("patience", 5),
        min_delta=config.get("min_delta", 0.001),
        mode='max',  # Using recall@5 as primary metric
        verbose=True
    )

    # Training history
    epoch_losses, step_losses, grad_norms = [], [], []
    val_losses, val_recall5, val_accuracies = [], [], []
    best_val_recall5 = 0.0

    for epoch in tqdm(range(config["epochs"]), desc="Training Epochs"):
        # ========== TRAINING PHASE ==========
        encoder.train()
        apply_freeze_strategy(encoder, config["freeze_strategy"], epoch)
        total_loss, batch_count = 0.0, 0
        optimizer.zero_grad()  # Initialize gradients at epoch start

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical = batch['categorical'].to(device)
            numeric = batch['numeric'].to(device)
            labels = batch['labels'].to(device)

            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                embeddings = encoder(input_ids, attention_mask, categorical, numeric)
                triplets = sample_triplets(embeddings, labels, config["margin"], max_triplets=64)
                if not triplets: continue

                anchor_emb = torch.stack([embeddings[a] for a, _, _ in triplets])
                pos_emb = torch.stack([embeddings[p] for _, p, _ in triplets])
                neg_emb = torch.stack([embeddings[n] for _, _, n in triplets])

                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / accumulation_steps

            # Backward pass with gradient scaling
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate gradients and step optimizer every accumulation_steps
            if batch_idx % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)  # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

                # Track gradient norms (only on accumulation steps)
                grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
                grad_norms.append(grad_norm)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            # Track loss (unscaled for logging)
            step_losses.append(loss.item() * accumulation_steps)
            total_loss += loss.item() * accumulation_steps
            batch_count += 1

            if batch_idx % (3 * accumulation_steps) == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{batch_idx}], Loss: {loss.item() * accumulation_steps:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config['epochs']}] Train Loss: {avg_loss:.4f}")

        # ========== VALIDATION PHASE (OPTIMIZED) ==========
        print("Evaluating on validation set...")
        # Single pass: compute both loss and retrieval metrics
        val_metrics = evaluate_validation_metrics(
            encoder, val_dataloader, triplet_loss_fn, device,
            k_values=[1, 5, 10], margin=config["margin"], max_triplets=64
        )

        # Store validation metrics
        val_losses.append(val_metrics['val_loss'])
        val_recall5.append(val_metrics['recall@5'])
        val_accuracies.append(val_metrics['accuracy'])

        # Print validation report
        print_validation_report(val_metrics, epoch + 1)

        # Save best model based on recall@5
        if val_metrics['recall@5'] > best_val_recall5:
            best_val_recall5 = val_metrics['recall@5']
            best_model_path = f"{exp_dir}/fusion_encoder_best.pth"
            print(f'New best model! Saving to {best_model_path}')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall5': best_val_recall5,
                'val_metrics': val_metrics
            }, best_model_path)

        # Periodic checkpoint
        if epoch % 3 == 0:
            model_path = f"{exp_dir}/fusion_encoder_epoch_{epoch+1}.pth"
            print(f'Saving checkpoint at {model_path}')
            torch.save(encoder.state_dict(), model_path)

        # Early stopping check
        if early_stopping(val_metrics['recall@5'], epoch + 1):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # Save logs with validation metrics
    log_data = {
        "epoch_losses": epoch_losses,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "val_losses": val_losses,
        "val_recall5": val_recall5,
        "val_accuracies": val_accuracies,
        "best_val_recall5": best_val_recall5,
        "best_epoch": early_stopping.best_epoch
    }

    with open(os.path.join(exp_dir, "logs", "training_logs.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    # Plot training curves
    plot_loss_curves(log_data, os.path.join(exp_dir, "plots", "loss_curve.png"))
    plot_grad_norms(log_data, os.path.join(exp_dir, "plots", "grad_norms.png"))

    # Plot validation metrics
    plot_validation_curves(log_data, os.path.join(exp_dir, "plots", "validation_metrics.png"))

    print(f"\nExperiment {exp_name} completed!")
    print(f"Best Recall@5: {best_val_recall5:.4f} at epoch {early_stopping.best_epoch}")
    print(f"Logs and plots saved in {exp_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple FusionEncoder experiments")
    parser.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    parser.add_argument("--single", type=str, help="Single experiment config as JSON string")
    parser.add_argument("--format", type=str, default="json", choices=["json", "yaml"], help="Config file format")
    args = parser.parse_args()

    if args.config:
        if args.format == "yaml":
            with open(args.config, "r") as f:
                configs = yaml.safe_load(f)
        else:
            with open(args.config, "r") as f:
                configs = json.load(f)
        for config in configs:
            run_experiment(config)
    elif args.single:
        config = json.loads(args.single)
        run_experiment(config)
    else:
        print("Please provide either --config or --single")

    


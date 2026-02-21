import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
import json
import yaml
import argparse

from src.data_loader import TransactionDataset, _normalise_df
from src.fusion_encoder import FusionEncoder
from src.triplet_sampler import ClassBalancedBatchSampler
from src.plotting import plot_loss_curves, plot_grad_norms, plot_embedding_projection, plot_validation_curves
from src.validation import EarlyStopping, evaluate_validation_metrics, print_validation_report


# ---------------------------------------------------------------------------
# collate_fn (kept here so it is importable from train_multi_expt as before)
# ---------------------------------------------------------------------------
def collate_fn(batch):
    input_ids      = torch.stack([item['input_ids']      for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    categorical    = torch.stack([item['categorical']    for item in batch])
    numeric        = torch.stack([item['numeric']        for item in batch])
    labels         = torch.stack([item['label']          for item in batch])
    metadata       = [item['metadata']                   for item in batch]
    return {
        'input_ids': input_ids, 'attention_mask': attention_mask,
        'categorical': categorical, 'numeric': numeric,
        'labels': labels, 'metadata': metadata,
    }


# ---------------------------------------------------------------------------
# Freeze strategy  — FIXED: unfreeze from TOP layers down (semantic → syntactic)
# ---------------------------------------------------------------------------
def apply_freeze_strategy(encoder, strategy, epoch=None):
    if strategy == "freeze":
        for param in encoder.bert.parameters():
            param.requires_grad = False

    elif strategy == "gradual":
        total_layers = len(encoder.bert.encoder.layer)
        # Unfreeze from the top (highest-layer index) downward
        layers_to_unfreeze = min(epoch + 1, total_layers)
        print(f'Total BERT layers: {total_layers} | Unfreezing top {layers_to_unfreeze}')
        for i in range(total_layers):
            # Layer indices from the top: total_layers-1, total_layers-2, ...
            layer_idx = total_layers - 1 - i
            requires_grad = i < layers_to_unfreeze
            for param in encoder.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = requires_grad

    elif strategy == "full":
        for param in encoder.bert.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (Khosla et al. 2020)
# ---------------------------------------------------------------------------
def supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    class_weights: dict = None,
    per_class_temperature: dict = None,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss.

    Args:
        embeddings:            [B, D] L2-normalised embeddings.
        labels:                [B]    integer class labels.
        temperature:           global temperature scalar (used when
                               per_class_temperature is None).
        class_weights:         {label_int: weight_float} inverse-frequency
                               weights.  None → uniform weighting.
        per_class_temperature: {label_int: temp_float} per-class temperature.
                               None → use single `temperature` for all.

    Returns:
        Scalar loss.
    """
    B = embeddings.shape[0]
    device = embeddings.device

    # Build per-sample temperature tensor
    if per_class_temperature is not None:
        temps = torch.tensor(
            [per_class_temperature.get(int(l), temperature) for l in labels],
            dtype=torch.float, device=device,
        )  # [B]
    else:
        temps = torch.full((B,), temperature, dtype=torch.float, device=device)

    # Pairwise cosine similarity matrix (embeddings are already L2-normalised)
    sim = torch.mm(embeddings, embeddings.T)          # [B, B]

    # Scale each row by its anchor's temperature
    sim = sim / temps.unsqueeze(1)                    # [B, B]

    # Masks
    mask_self = ~torch.eye(B, dtype=torch.bool, device=device)   # exclude self
    mask_pos  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & mask_self  # [B, B]

    # Skip anchors that have no positive in the batch (can happen for rare classes)
    valid_anchors = mask_pos.any(dim=1)               # [B]
    if not valid_anchors.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-sum-exp denominator (all non-self pairs)
    log_denom = torch.logsumexp(
        sim.masked_fill(~mask_self, float('-inf')), dim=1, keepdim=True
    )  # [B, 1]

    # Per-anchor SupCon loss
    log_prob     = sim - log_denom                    # [B, B]
    num_positives = mask_pos.sum(dim=1).clamp(min=1)  # [B]
    per_anchor    = -(log_prob * mask_pos).sum(dim=1) / num_positives  # [B]

    # Apply inverse-frequency sample weights
    if class_weights is not None:
        weights = torch.tensor(
            [class_weights.get(int(l), 1.0) for l in labels],
            dtype=torch.float, device=device,
        )
        per_anchor = per_anchor * weights

    # Only average over anchors that have at least one positive
    loss = per_anchor[valid_anchors].mean()
    return loss


# ---------------------------------------------------------------------------
# Semi-hard negative mining (replaces random sample_triplets)
# ---------------------------------------------------------------------------
def sample_triplets(embeddings, labels, margin=0.5, max_triplets=64):
    """
    Semi-hard negative mining: for each anchor-positive pair, select a
    negative n where  d(a,p) < d(a,n) < d(a,p) + margin.
    Falls back to the hardest negative if no semi-hard candidate exists.

    Kept as 'sample_triplets' so existing call-sites (validation.py) work.
    """
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    triplets = []
    batch_size = len(labels)

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        pos_mask = (labels == anchor_label)
        pos_mask[anchor_idx] = False
        pos_candidates = pos_mask.nonzero(as_tuple=True)[0]
        if len(pos_candidates) == 0:
            continue

        pos_idx = pos_candidates[torch.randint(0, len(pos_candidates), (1,)).item()]
        d_ap = dist_matrix[anchor_idx, pos_idx].item()

        neg_mask = labels != anchor_label
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) == 0:
            continue

        neg_dists = dist_matrix[anchor_idx, neg_indices]

        # Semi-hard: d_ap < d_an < d_ap + margin
        semi_hard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + margin)
        semi_hard_neg  = neg_indices[semi_hard_mask]

        if len(semi_hard_neg) > 0:
            chosen = semi_hard_neg[torch.randint(0, len(semi_hard_neg), (1,)).item()]
        else:
            # Fallback: hardest negative (closest from wrong class)
            chosen = neg_indices[neg_dists.argmin()]

        triplets.append((anchor_idx, pos_idx.item(), chosen.item()))
        if len(triplets) >= max_triplets:
            break

    return triplets


# ---------------------------------------------------------------------------
# Class-frequency weight helpers
# ---------------------------------------------------------------------------
def compute_class_weights(labels_array, power: float = 0.5) -> dict:
    """
    Inverse-frequency weights with sqrt dampening.
    weight_i = (total_N / class_count_i) ^ power

    power=0.5 gives ~4.5x weight for 250-sample vs 5000-sample class
    instead of 20x for raw inverse, avoiding gradient instability.
    """
    counts = Counter(int(l) for l in labels_array)
    total  = sum(counts.values())
    return {label: (total / count) ** power for label, count in counts.items()}


def compute_per_class_temperature(labels_array, base_temp: float = 0.07, power: float = 0.3) -> dict:
    """
    Per-class temperature: minority classes get lower temperature (sharper
    gradients); majority classes get higher temperature (prevents overcrowding).

    T_i = base_temp * (class_freq_i ^ power)
    """
    counts = Counter(int(l) for l in labels_array)
    total  = sum(counts.values())
    return {
        label: base_temp * ((count / total) ** power)
        for label, count in counts.items()
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_experiment(config):
    # Gradient accumulation setup
    accumulation_steps  = config.get("accumulation_steps", 1)
    base_batch_size     = config.get("base_batch_size", 256)
    effective_batch_size = config["batch_size"] * accumulation_steps

    # Learning rate scaling: scale with sqrt of batch size ratio
    if "base_lr" in config:
        batch_ratio = effective_batch_size / base_batch_size
        scaled_lr   = config["base_lr"] * (batch_ratio ** 0.5)
        config["lr"] = scaled_lr
        print(f"LR Scaling: base_lr={config['base_lr']:.2e}, scaled_lr={scaled_lr:.2e}")

    loss_type   = config.get("loss_type", "supcon")   # "supcon" or "triplet"
    temperature = config.get("temperature", 0.07)

    exp_name = (
        f"tagger_proj{config['text_proj_dim']}_final{config['final_dim']}"
        f"_freeze-{config['freeze_strategy']}"
        f"_bs{effective_batch_size}_lr{config['lr']:.2e}"
        f"_{loss_type}"
    )
    exp_dir = os.path.join("experiments", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"),  exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiment: {exp_name} on {device}")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    df = pd.read_csv(config["csv_path"])
    df = _normalise_df(df)                          # apply column aliases + fill NaN

    # Exclude "other" label from training (it is OOD, not a real class)
    ood_label = config.get("ood_label", "other")
    if ood_label in df[config["label_col"]].values:
        n_before = len(df)
        df = df[df[config["label_col"]] != ood_label].reset_index(drop=True)
        print(f"Excluded '{ood_label}' OOD label: {n_before} → {len(df)} records")

    if "sample_size" in config and len(df) > config["sample_size"]:
        df = df.sample(n=config["sample_size"], random_state=42).reset_index(drop=True)

    val_split = config.get("val_split", 0.15)
    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=42, stratify=df[config["label_col"]]
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])

    train_dataset = TransactionDataset(
        train_df, tokenizer,
        config["categorical_cols"], config["numeric_cols"], config["label_col"]
    )
    val_dataset = TransactionDataset(
        val_df, tokenizer,
        config["categorical_cols"], config["numeric_cols"], config["label_col"]
    )
    # Share vocab/scaler from training into validation
    val_dataset.cat_vocab     = train_dataset.cat_vocab
    val_dataset.scaler        = train_dataset.scaler
    val_dataset.numeric_data  = val_dataset.scaler.transform(val_df[config["numeric_cols"]])
    val_dataset.label_mapping = train_dataset.label_mapping

    # ------------------------------------------------------------------
    # Class-balanced batch sampler (replaces shuffle=True)
    # ------------------------------------------------------------------
    balanced_sampler = ClassBalancedBatchSampler(
        train_dataset.labels.tolist(), config["batch_size"]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=balanced_sampler, collate_fn=collate_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    # ------------------------------------------------------------------
    # Inverse-frequency weights and per-class temperatures
    # ------------------------------------------------------------------
    all_train_labels = train_dataset.labels.tolist()
    class_weights          = compute_class_weights(all_train_labels, power=0.5)
    per_class_temps        = compute_per_class_temperature(
        all_train_labels, base_temp=temperature, power=0.3
    )

    print(f"Classes: {len(class_weights)} | min weight: {min(class_weights.values()):.2f} "
          f"| max weight: {max(class_weights.values()):.2f}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    categorical_dims = [len(train_dataset.cat_vocab[col]) for col in config["categorical_cols"]]
    encoder = FusionEncoder(
        bert_model_name=config["bert_model"],
        categorical_dims=categorical_dims,
        numeric_dim=len(config["numeric_cols"]),
        text_proj_dim=config["text_proj_dim"],
        final_dim=config["final_dim"],
        p=config.get("dropout", 0.1),
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=config["lr"]
    )
    scheduler      = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    triplet_loss_fn = nn.TripletMarginLoss(margin=config.get("margin", 0.5))  # used for val loss

    use_amp = config.get("use_amp", False) and torch.cuda.is_available()
    amp_scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision training enabled (AMP)")

    early_stopping = EarlyStopping(
        patience=config.get("patience", 5),
        min_delta=config.get("min_delta", 0.001),
        mode='max', verbose=True,
    )

    epoch_losses, step_losses, grad_norms = [], [], []
    val_losses, val_recall5, val_accuracies = [], [], []
    best_val_recall5 = 0.0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in tqdm(range(config["epochs"]), desc="Training Epochs"):
        encoder.train()
        apply_freeze_strategy(encoder, config["freeze_strategy"], epoch)
        total_loss, batch_count = 0.0, 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dataloader, start=1):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical    = batch['categorical'].to(device)
            numeric        = batch['numeric'].to(device)
            labels         = batch['labels'].to(device)

            with autocast(enabled=use_amp):
                embeddings = encoder(input_ids, attention_mask, categorical, numeric)

                if loss_type == "supcon":
                    loss = supcon_loss(
                        embeddings, labels,
                        temperature=temperature,
                        class_weights=class_weights,
                        per_class_temperature=per_class_temps,
                    )
                else:
                    # Fallback to semi-hard triplet loss
                    triplets = sample_triplets(
                        embeddings, labels, config.get("margin", 0.5), max_triplets=64
                    )
                    if not triplets:
                        continue
                    anchor_emb = torch.stack([embeddings[a] for a, _, _ in triplets])
                    pos_emb    = torch.stack([embeddings[p] for _, p, _ in triplets])
                    neg_emb    = torch.stack([embeddings[n] for _, _, n in triplets])
                    loss       = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

                loss = loss / accumulation_steps

            if use_amp:
                amp_scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_idx % accumulation_steps == 0:
                if use_amp:
                    amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                grad_norm = sum(
                    p.grad.norm().item() for p in encoder.parameters() if p.grad is not None
                )
                grad_norms.append(grad_norm)

                if use_amp:
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            step_losses.append(loss.item() * accumulation_steps)
            total_loss  += loss.item() * accumulation_steps
            batch_count += 1

            if batch_idx % (3 * accumulation_steps) == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{batch_idx}], "
                      f"Loss: {loss.item() * accumulation_steps:.4f}")

        scheduler.step()
        avg_loss = total_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{config['epochs']}] Train Loss: {avg_loss:.4f}")

        # Validation
        print("Evaluating on validation set...")
        val_metrics = evaluate_validation_metrics(
            encoder, val_dataloader, triplet_loss_fn, device,
            k_values=[1, 5, 10], margin=config.get("margin", 0.5), max_triplets=64
        )
        val_losses.append(val_metrics['val_loss'])
        val_recall5.append(val_metrics['recall@5'])
        val_accuracies.append(val_metrics['accuracy'])
        print_validation_report(val_metrics, epoch + 1)

        if val_metrics['recall@5'] > best_val_recall5:
            best_val_recall5 = val_metrics['recall@5']
            best_model_path  = f"{exp_dir}/fusion_encoder_best.pth"
            print(f'New best model! Saving to {best_model_path}')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_recall5': best_val_recall5,
                'val_metrics': val_metrics,
            }, best_model_path)

        if epoch % 3 == 0:
            torch.save(encoder.state_dict(), f"{exp_dir}/fusion_encoder_epoch_{epoch+1}.pth")

        if early_stopping(val_metrics['recall@5'], epoch + 1):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    log_data = {
        "epoch_losses": epoch_losses, "step_losses": step_losses,
        "grad_norms": grad_norms, "val_losses": val_losses,
        "val_recall5": val_recall5, "val_accuracies": val_accuracies,
        "best_val_recall5": best_val_recall5,
        "best_epoch": early_stopping.best_epoch,
    }
    with open(os.path.join(exp_dir, "logs", "training_logs.json"), "w") as f:
        json.dump(log_data, f, indent=2)

    plot_loss_curves(log_data, os.path.join(exp_dir, "plots", "loss_curve.png"))
    plot_grad_norms(log_data,  os.path.join(exp_dir, "plots", "grad_norms.png"))
    plot_validation_curves(log_data, os.path.join(exp_dir, "plots", "validation_metrics.png"))

    print(f"\nExperiment {exp_name} completed!")
    print(f"Best Recall@5: {best_val_recall5:.4f} at epoch {early_stopping.best_epoch}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FusionEncoder experiments")
    parser.add_argument("--config", type=str, help="Path to config file (YAML or JSON)")
    parser.add_argument("--single", type=str, help="Single experiment config as JSON string")
    parser.add_argument("--format", type=str, default="yaml", choices=["json", "yaml"])
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            configs = yaml.safe_load(f) if args.format == "yaml" else json.load(f)
        for config in configs:
            run_experiment(config)
    elif args.single:
        import json as _json
        run_experiment(_json.loads(args.single))
    else:
        print("Please provide either --config or --single")

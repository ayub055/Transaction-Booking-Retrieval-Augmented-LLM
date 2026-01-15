import torch
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=5, min_delta=0.0, mode='min', verbose=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/recall
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        """
        Call this method after each epoch with validation metric.
        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False

        # Check if current score is better
        if self.mode == 'min':
            is_better = current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            is_better = current_score > (self.best_score + self.min_delta)

        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"Validation metric improved to {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True

        return False


def evaluate_retrieval_metrics(encoder, dataloader, device, k_values=[1, 5, 10]):
    """
    Evaluate retrieval-based metrics: Recall@K and Mean Reciprocal Rank (MRR).

    Args:
        encoder: The FusionEncoder model
        dataloader: Validation dataloader
        device: torch device
        k_values: List of K values for Recall@K

    Returns:
        dict: Dictionary containing recall@k and MRR scores
    """
    encoder.eval()
    all_embeddings = []
    all_labels = []

    # Encode all validation samples
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical = batch['categorical'].to(device)
            numeric = batch['numeric'].to(device)
            labels = batch['labels'].to(device)

            embeddings = encoder(input_ids, attention_mask, categorical, numeric)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, embedding_dim]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # Compute pairwise distances
    # For normalized embeddings, cosine similarity = 1 - euclidean_distance^2 / 2
    # But we'll use L2 distance for consistency
    dist_matrix = torch.cdist(all_embeddings, all_embeddings, p=2)  # [N, N]

    # For each query, find nearest neighbors
    recall_at_k = {k: [] for k in k_values}
    reciprocal_ranks = []
    correct_predictions = []

    num_samples = len(all_labels)

    for i in range(num_samples):
        query_label = all_labels[i].item()

        # Get distances to all other samples (exclude self)
        distances = dist_matrix[i].clone()
        distances[i] = float('inf')  # Exclude self

        # Sort by distance (ascending)
        sorted_indices = torch.argsort(distances)
        sorted_labels = all_labels[sorted_indices]

        # Recall@K: Check if any of top-K neighbors have same label
        for k in k_values:
            top_k_labels = sorted_labels[:k]
            has_correct = (top_k_labels == query_label).any().item()
            recall_at_k[k].append(int(has_correct))

        # MRR: Find rank of first correct match
        correct_mask = (sorted_labels == query_label)
        if correct_mask.any():
            first_correct_rank = (correct_mask.nonzero(as_tuple=True)[0][0].item() + 1)
            reciprocal_ranks.append(1.0 / first_correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Classification accuracy: Top-1 prediction
        pred_label = sorted_labels[0].item()
        correct_predictions.append(int(pred_label == query_label))

    # Compute metrics
    metrics = {
        'accuracy': np.mean(correct_predictions),
        'mrr': np.mean(reciprocal_ranks)
    }

    for k in k_values:
        metrics[f'recall@{k}'] = np.mean(recall_at_k[k])

    return metrics


def compute_validation_loss(encoder, dataloader, triplet_loss_fn, device, margin=0.5, max_triplets=64):
    """
    Compute validation loss using triplet sampling.

    Args:
        encoder: The FusionEncoder model
        dataloader: Validation dataloader
        triplet_loss_fn: Triplet loss function
        device: torch device
        margin: Triplet margin
        max_triplets: Maximum triplets per batch

    Returns:
        float: Average validation loss
    """
    from src.train_multi_expt import sample_triplets

    encoder.eval()
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            categorical = batch['categorical'].to(device)
            numeric = batch['numeric'].to(device)
            labels = batch['labels'].to(device)

            embeddings = encoder(input_ids, attention_mask, categorical, numeric)
            triplets = sample_triplets(embeddings, labels, margin, max_triplets)

            if not triplets:
                continue

            anchor_emb = torch.stack([embeddings[a] for a, _, _ in triplets])
            pos_emb = torch.stack([embeddings[p] for _, p, _ in triplets])
            neg_emb = torch.stack([embeddings[n] for _, _, n in triplets])

            loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            total_loss += loss.item()
            batch_count += 1

    avg_loss = total_loss / max(batch_count, 1)
    return avg_loss


def print_validation_report(metrics, epoch):
    """Pretty print validation metrics."""
    print("\n" + "="*60)
    print(f"Validation Report - Epoch {epoch}")
    print("="*60)
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"MRR:            {metrics['mrr']:.4f}")

    for k in [1, 5, 10]:
        if f'recall@{k}' in metrics:
            print(f"Recall@{k:2d}:      {metrics[f'recall@{k}']:.4f}")

    if 'val_loss' in metrics:
        print(f"Val Loss:       {metrics['val_loss']:.4f}")
    print("="*60 + "\n")

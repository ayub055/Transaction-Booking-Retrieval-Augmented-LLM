import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch

# ---------------------- Plot Loss Curves ----------------------
def plot_loss_curves(log_data, save_path):
    epoch_losses = log_data.get("epoch_losses", [])
    step_losses = log_data.get("step_losses", [])

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Epoch Loss")
    plt.title("Epoch Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

    # Optional: Step loss curve
    step_path = save_path.replace("loss_curve.png", "step_loss_curve.png")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(step_losses) + 1), step_losses, label="Step Loss", alpha=0.7)
    plt.title("Step Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(step_path)
    plt.close()

# ---------------------- Plot Gradient Norms ----------------------
def plot_grad_norms(log_data, save_path):
    grad_norms = log_data.get("grad_norms", [])
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(grad_norms) + 1), grad_norms, color='orange', alpha=0.8)
    plt.title("Gradient Norms Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ---------------------- TSNE Embedding Visualization ----------------------
def plot_embedding_projection(embeddings, labels, save_path, perplexity=30, n_iter=1000):
    """
    embeddings: torch.Tensor or numpy array of shape [num_samples, embedding_dim]
    labels: numpy array of shape [num_samples]
    """
    if isinstance(embeddings, torch.Tensor): embeddings = embeddings.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("TSNE Projection of Embeddings")
    plt.savefig(save_path)
    plt.close()

# ---------------------- Plot Validation Metrics ----------------------
def plot_validation_curves(log_data, save_path):
    """
    Plot validation metrics over epochs including:
    - Train loss vs Validation loss
    - Recall@5
    - Accuracy
    """
    epoch_losses = log_data.get("epoch_losses", [])
    val_losses = log_data.get("val_losses", [])
    val_recall5 = log_data.get("val_recall5", [])
    val_accuracies = log_data.get("val_accuracies", [])

    epochs = range(1, len(epoch_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Train vs Val Loss
    axes[0, 0].plot(epochs, epoch_losses, marker='o', label="Train Loss", color='blue')
    if val_losses:
        axes[0, 0].plot(epochs, val_losses, marker='s', label="Val Loss", color='red')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Validation Recall@5
    if val_recall5:
        axes[0, 1].plot(epochs, val_recall5, marker='o', label="Recall@5", color='green')
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Recall@5")
        axes[0, 1].set_title("Validation Recall@5")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim([0, 1.05])

    # Plot 3: Validation Accuracy
    if val_accuracies:
        axes[1, 0].plot(epochs, val_accuracies, marker='o', label="Accuracy", color='purple')
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Validation Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_ylim([0, 1.05])

    # Plot 4: Combined metrics
    if val_recall5 and val_accuracies:
        axes[1, 1].plot(epochs, val_recall5, marker='o', label="Recall@5", color='green')
        axes[1, 1].plot(epochs, val_accuracies, marker='s', label="Accuracy", color='purple')
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].set_title("Combined Validation Metrics")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

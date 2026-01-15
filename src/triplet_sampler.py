import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os

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

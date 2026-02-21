import random
import itertools
from collections import defaultdict
from torch.utils.data import Sampler


class TripletSampler(Sampler):
    """Original sampler — kept for backward compatibility."""

    def __init__(self, labels, num_triplets):
        self.labels = labels
        self.num_triplets = num_triplets
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        triplets = []
        for _ in range(self.num_triplets):
            anchor_label = random.choice(self.unique_labels)
            negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
            anchor_idx  = random.choice(self.label_to_indices[anchor_label])
            positive_idx = random.choice(
                [i for i in self.label_to_indices[anchor_label] if i != anchor_idx]
            )
            negative_idx = random.choice(self.label_to_indices[negative_label])
            triplets.append((anchor_idx, positive_idx, negative_idx))
        return iter(triplets)

    def __len__(self):
        return self.num_triplets


class ClassBalancedBatchSampler(Sampler):
    """
    Yields batches where every class is (approximately) equally represented.

    For 43 classes at batch_size=128 this gives ~3 examples/class/batch,
    preventing majority classes from dominating SupCon gradients.

    Usage with DataLoader:
        sampler = ClassBalancedBatchSampler(dataset.labels.tolist(), batch_size=128)
        loader  = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

    Note: pass as `batch_sampler`, NOT `sampler` — DataLoader must not receive
    batch_size when a batch_sampler is given.
    """

    def __init__(self, labels, batch_size: int):
        """
        Args:
            labels: integer label for each sample (list or 1-D tensor).
            batch_size: number of samples per batch.
        """
        self.labels = [int(l) for l in labels]
        self.batch_size = batch_size

        self.label_to_indices: dict = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.unique_labels = sorted(self.label_to_indices.keys())
        self.num_classes   = len(self.unique_labels)
        # At least 2 per class so SupCon always has a positive pair in the batch
        self.samples_per_class = max(2, batch_size // self.num_classes)

    # ------------------------------------------------------------------
    def __iter__(self):
        # Fresh shuffle of each class's indices every epoch
        shuffled = {
            label: random.sample(indices, len(indices))
            for label, indices in self.label_to_indices.items()
        }
        # Cyclic iterators so small classes are repeated within an epoch
        iters = {label: itertools.cycle(shuffled[label]) for label in self.unique_labels}

        for _ in range(len(self)):
            batch = []
            for label in self.unique_labels:
                for _ in range(self.samples_per_class):
                    batch.append(next(iters[label]))
            random.shuffle(batch)
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        # Number of batches ≈ enough to see the largest class at least once
        max_class_size = max(len(v) for v in self.label_to_indices.values())
        return max(1, max_class_size // self.samples_per_class)

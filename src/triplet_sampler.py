import random
from torch.utils.data import Sampler

class TripletSampler(Sampler):
    def __init__(self, 
                 labels, 
                 num_triplets):
        """
        Args:
            labels: list or tensor of labels for all samples in the dataset.
            num_triplets: number of triplets to sample per epoch.
        """
        self.labels = labels
        self.num_triplets = num_triplets

        # Group indices by label
        self.label_to_indices = {}
        for idx, label in enumerate(labels): self.label_to_indices.setdefault(label, []).append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        triplets = []
        for _ in range(self.num_triplets):
            # Anchor label
            anchor_label = random.choice(self.unique_labels)
            positive_label = anchor_label
            negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])

            # Sample indices
            anchor_idx = random.choice(self.label_to_indices[anchor_label])
            positive_idx = random.choice([i for i in self.label_to_indices[positive_label] if i != anchor_idx])
            negative_idx = random.choice(self.label_to_indices[negative_label])

            triplets.append((anchor_idx, positive_idx, negative_idx))

        return iter(triplets)

    def __len__(self):
        return self.num_triplets
import torch
import torch.nn as nn

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent

    def defuzzify(self):
        
        #  Normalize Gamma by dividing each element by the sum of all elements)
        normalized_gamma = self.parent.Gamma[0:self.parent.c] / (self.parent.Gamma[0:self.parent.c].sum())

        # Filter out unlabeled cluster labels (assuming -1 indicates unlabeled)
        #labeled_indices = self.parent.cluster_labels[0:self.parent.c] != -1

        # Select only labeled data for Gamma and cluster labels
        #labeled_gamma = normalized_gamma[labeled_indices]
        labeled_cluster_labels = self.parent.cluster_labels[0:self.parent.c]

        # Convert labeled cluster labels to one-hot encoding
        one_hot_labels = nn.functional.one_hot(
            labeled_cluster_labels.to(torch.int64),
            num_classes=self.parent.num_classes
        )

        # Multiply normalized memberships with one-hot labels and sum across clusters
        label_scores = torch.sum(
            normalized_gamma.unsqueeze(-1) * one_hot_labels.float(),
            dim=0
        )

        return label_scores

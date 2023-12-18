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
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * self.parent.cluster_labels[:self.parent.c], dim=0)

        # Find the index of the maximum value in Gamma
        max_index = torch.argmax(normalized_gamma)

        # Retrieve the corresponding label
        max_label = torch.argmax(self.parent.cluster_labels[max_index])

        return label_scores, max_label

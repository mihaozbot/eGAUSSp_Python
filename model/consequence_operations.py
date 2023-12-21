import torch
import torch.nn as nn

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent

    def defuzzify(self):
        
        # Filter out unlabeled cluster labels (assuming -1 indicates unlabeled)
        #labeled_indices = self.parent.cluster_labels[0:self.parent.c] != -1
        normalized_gamma = (self.parent.Gamma[0:self.parent.c])/(self.parent.Gamma[0:self.parent.c].sum())

        # Select only labeled data for Gamma and cluster labels
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * self.parent.cluster_labels[:self.parent.c], dim=0)

        # Find the index of the maximum value in Gamma
        #max_index = torch.argmax(gamma)

        # Retrieve the corresponding label
        #max_label = torch.argmax(self.parent.cluster_labels[max_index])
        
        max_label = torch.argmax(label_scores)
        
        return label_scores, max_label

    def defuzzify_batch(self):

        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.parent.Gamma / self.parent.Gamma.sum(dim=1, keepdim=True)

        # Select only labeled data for Gamma and cluster labels
        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)

        # Compute label scores
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels, dim=1)

        # Find the indices of the maximum values in label_scores along the label dimension
        max_labels = torch.argmax(label_scores, dim=1)

        return label_scores, max_labels

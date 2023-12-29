import torch
import torch.nn as nn

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent

    def defuzzify(self):
        
        # Filter out unlabeled cluster labels (assuming -1 indicates unlabeled)
        #labeled_indices = self.parent.cluster_labels[0:self.parent.c] != -1
        normalized_gamma = self.compute_normalized_gamma()

        # Select only labeled data for Gamma and cluster labels
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * self.parent.cluster_labels[:self.parent.c], dim=0)

        # Find the index of the maximum value in Gamma
        max_index = torch.argmax(normalized_gamma)

        # Retrieve the corresponding label
        max_label = torch.argmax(self.parent.cluster_labels[max_index])
        
        #max_label = torch.argmax(label_scores)
        
        return label_scores, max_label

    '''
    def defuzzify_batch(self):

        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.compute_batched_normalized_gamma()
        
        # Select only labeled data for Gamma and cluster labels
        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)

        # Compute label scores
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels, dim=1)

        # Find the indices of the maximum values in label_scores along the label dimension
        max_labels = torch.argmax(label_scores, dim=1)

        return label_scores, max_labels
    '''
    
    def defuzzify_batch(self):
        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.compute_batched_normalized_gamma()
        
        # Select only labeled data for Gamma and cluster labels
        # Ensure that cluster labels are a 1D tensor of class indices
        cluster_labels = self.parent.cluster_labels[:self.parent.c]

        # Compute label scores
        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)

        # Compute label scores
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels, dim=1)


        # Find the indices of the maximum values in normalized_gamma along the cluster dimension
        max_indices = torch.argmax(normalized_gamma, dim=1)

        # Use max_indices to select the corresponding one-hot encoded class labels
        one_hot_max_labels = expanded_cluster_labels[torch.arange(normalized_gamma.shape[0]), max_indices]

        # Convert one-hot encoding to class indices
        max_labels = torch.argmax(one_hot_max_labels, dim=1)

        return label_scores, max_labels


    def compute_normalized_gamma(self):
        
        # Compute normalized gamma
        gamma = self.parent.Gamma[0:self.parent.c]
        gamma_sum = gamma.sum()

        # Avoid division by zero or NaN; if gamma_sum is not valid, set normalized_gamma to zero
        if gamma_sum == 0 or torch.isnan(gamma_sum):
            normalized_gamma = torch.zeros_like(gamma)
        else:
            normalized_gamma = gamma / gamma_sum
            # Replace NaN values in normalized_gamma with zeros
            normalized_gamma = torch.nan_to_num(normalized_gamma)

        return normalized_gamma

    def compute_batched_normalized_gamma(self):
        # Compute normalized gamma
        gamma = self.parent.Gamma[:,:self.parent.c]
        gamma_sum = gamma.sum(dim=1, keepdim=True)

        # Replace NaN values in gamma_sum with zeros
        gamma_sum = torch.nan_to_num(gamma_sum, nan= 0.0)

        # Avoid division by zero; if gamma_sum is zero, set normalized_gamma to zero
        normalized_gamma = torch.where(gamma_sum > 0, gamma / gamma_sum, torch.zeros_like(gamma))

        return normalized_gamma


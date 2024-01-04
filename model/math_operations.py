from math import inf
from re import S
import torch
import numpy as np

 # Attempt to load the line_profiler extension

class MathOps():
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim

    def compute_activation(self, z):
        if self.parent.c == 0:
            return torch.empty(0, device=self.parent.device, requires_grad=False)
        
        if len(self.parent.matching_clusters) == 0:
            return torch.zeros(self.parent.c, device=self.parent.device, requires_grad=False)

        mu = self.parent.mu[self.parent.matching_clusters]
        n = self.parent.n[self.parent.matching_clusters]
        S_inv = self.parent.S_inv[self.parent.matching_clusters]

        z_expanded = z.unsqueeze(0).expand(mu.shape[0], -1)

        single_sample_mask = n == 1
        non_single_sample_mask = ~single_sample_mask
        inv_cov_diag = 1 /(self.parent.S_0.diagonal()*self.feature_dim)
        
        d2 = torch.zeros(len(self.parent.matching_clusters), dtype=torch.float32, device=self.parent.device)

        #if single_sample_mask.any():
        diff_single_sample = z_expanded[single_sample_mask] - mu[single_sample_mask]
        d2[single_sample_mask] = torch.sum(diff_single_sample**2 * inv_cov_diag, dim=1)
    
        #if non_single_sample_mask.any():
        diff = (z_expanded[non_single_sample_mask] - mu[non_single_sample_mask]).unsqueeze(-1)
        S_inv_adjusted = S_inv[non_single_sample_mask] #Is already divided by feature_dim
        d2[non_single_sample_mask] = torch.bmm(torch.bmm(diff.transpose(1, 2), S_inv_adjusted), diff).squeeze()

        # Error handling for negative distances
        if (d2 < 0).any():
            print("Critical error! Negative distance detected in Gamma computation, which should be impossible")

            # Identify the indices of negative distances
            negative_distance_indices = torch.where(d2 < 0)[0]

            # Remove corresponding clusters
            with torch.no_grad():
                for index in negative_distance_indices:
                    self.parent.removal_mech.remove_cluster(index)

            # Filter out the negative distances
            d2 = d2[d2 >= 0]

            #This is not optimal, but it is required because remove_clusters is not made for different class labels
            self.matching_clusters = torch.arange(self.parent.c, dtype=torch.int32, device=self.parent.device)

        Gamma = torch.exp(-d2)
        full_Gamma = torch.zeros(self.parent.c, dtype=torch.float32, device=self.parent.device)
        full_Gamma[self.parent.matching_clusters] = Gamma

        return full_Gamma
    
    def compute_batched_activation(self, Z):
        """Compute membership of the batch of samples Z to the existing rules/clusters"""

        if self.parent.c == 0:
            return torch.empty(Z.shape[0], 0, device=self.parent.device)

        batch_size = Z.shape[0]
        full_Gamma = torch.zeros(batch_size, self.parent.c, device=self.parent.device)

        # Parameters for all clusters
        mu = self.parent.mu[0: self.parent.c]
        #n = self.parent.n[0: self.parent.c]
        #Sigma = self.parent.S[0: self.parent.c] / n[0: self.parent.c].view(-1, 1, 1)

        # Expanding Z for vectorized operations
        Z_expanded = Z.unsqueeze(1).expand(-1, mu.shape[0], -1)

        # Initialize distance tensor
        d2 = torch.full((batch_size, self.parent.c), float('inf'), dtype=torch.float32, device=self.parent.device)
        
        '''
        # Mask for clusters with a single sample
        single_sample_mask = n == 1

        # Compute distances for clusters with a single sample
        if single_sample_mask.sum() > 0:
            diff_single_sample = Z_expanded[:, single_sample_mask, :] - mu[single_sample_mask]
            inv_cov_diag = 1 / self.parent.S_0.diagonal()
            d2_single_sample = torch.sum(diff_single_sample ** 2 * inv_cov_diag, dim=2)
            d2[:, single_sample_mask] = d2_single_sample

        # Compute Mahalanobis distances for other clusters
        non_single_sample_mask = ~single_sample_mask
        if non_single_sample_mask.sum() > 0:
            S_inv = torch.linalg.inv(Sigma[non_single_sample_mask])

            # Ensure S_inv is correctly broadcasted for bmm
            S_inv_expanded = S_inv.unsqueeze(0).expand(batch_size, -1, -1, -1)

            # Reshape diff for bmm
            diff = (Z_expanded[:, non_single_sample_mask, :] - mu[non_single_sample_mask]).unsqueeze(-2)

            # Perform batch matrix multiplication
            # Using matmul for better broadcasting support
            d2[:, non_single_sample_mask] = torch.matmul(torch.matmul(diff, S_inv_expanded), diff.transpose(-2, -1)).squeeze(-1).squeeze(-1)
        '''

        #S_inv = torch.linalg.inv(Sigma)

        
        # Ensure S_inv is correctly broadcasted for bmm
        S_inv_expanded = self.parent.S_inv[:self.parent.c].unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Reshape diff for bmm
        diff = (Z_expanded[:, :, :] - mu[:]).unsqueeze(-2)

        # Perform batch matrix multiplication
        # Using matmul for better broadcasting support
        d2 = torch.matmul(torch.matmul(diff, S_inv_expanded), diff.transpose(-2, -1)).squeeze(-1).squeeze(-1)

        #Initialize full_Gamma tensor
        full_Gamma = torch.zeros_like(d2)

        # Compute activations and assign them to their respective places in full_Gamma
        batch_indices = torch.arange(Z.shape[0], device=self.parent.device).unsqueeze(1)
        full_Gamma[batch_indices, self.parent.matching_clusters] = torch.exp(-d2[batch_indices, self.parent.matching_clusters]/self.feature_dim)

        return full_Gamma
from math import inf
import torch

 # Attempt to load the line_profiler extension

class MathOps():
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim

    def compute_activation(self, z):
        """Compute membership of the current sample z to the exisiting rules/clusters"""
         
        if self.parent.c == 0:
            return torch.empty(0, device=self.parent.device, requires_grad=True)

        if len(self.parent.matching_clusters) == 0:
            return torch.zeros(self.parent.c, device=self.parent.device,requires_grad=True)

        # Direct tensor indexing
        mu = self.parent.mu[self.parent.matching_clusters]
        n = self.parent.n[self.parent.matching_clusters]
        Sigma = self.parent.S[self.parent.matching_clusters]/n.view(-1, 1, 1) 

        # Expanding z for vectorized operations
        z_expanded = z.unsqueeze(0).expand(mu.shape[0], -1)

        # Initialize distance tensor
        d2 = torch.zeros(len(self.parent.matching_clusters), dtype=torch.float32, device=self.parent.device)

        # Mask for clusters with a single sample
        single_sample_mask = n == 1
        
        # Compute distances for clusters with a single sample,
        if single_sample_mask.sum() > 0:
            diff_single_sample = z_expanded[single_sample_mask] - mu[single_sample_mask]
            #inv_cov_diag = torch.linalg.pinv(self.parent.S_0).repeat(single_sample_mask.sum(), 1, 1).diagonal(dim1=-2, dim2=-1)
            inv_cov_diag = 1 / self.parent.S_0.diagonal()
            d2_single_sample = torch.sum(diff_single_sample**2 * inv_cov_diag, dim=1)
            d2[single_sample_mask] = d2_single_sample

        # Compute Mahalanobis distances for other clusters
        non_single_sample_mask = ~single_sample_mask
        if non_single_sample_mask.sum() > 0: 
            S_inv = torch.linalg.inv(Sigma[non_single_sample_mask])
            diff = (z_expanded[non_single_sample_mask] - mu[non_single_sample_mask]).unsqueeze(-1)
            d2_mahalanobis = torch.bmm(torch.bmm(diff.transpose(1, 2), S_inv), diff).squeeze()
            d2[non_single_sample_mask] = d2_mahalanobis

        if (d2 < 0).any():
            d2[d2<0]= inf
            print("Critical error! Negative distance detected in Gamma computation, which should be impossible")

        # Compute activations for the candidate clusters
        Gamma = torch.exp(-d2) #+ 1e-30 #*scaling_factor

        if torch.isnan(Gamma).any().item():
            print("Critical error! NaN detected in Gamma computation")

        # Expand activations and distances to the full set of clusters
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
        n = self.parent.n[0: self.parent.c]
        Sigma = self.parent.S[0: self.parent.c] / n[0: self.parent.c].view(-1, 1, 1)

        # Expanding Z for vectorized operations
        Z_expanded = Z.unsqueeze(1).expand(-1, mu.shape[0], -1)

        # Initialize distance tensor
        d2 = torch.full((batch_size, self.parent.c), float('inf'), dtype=torch.float32, device=self.parent.device)

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
            
        if (d2 < 0).any():
            d2[d2 < 0] = float('inf')
            print("Critical error! Negative distance detected in Gamma computation, which should be impossible")

        #Initialize full_Gamma tensor
        full_Gamma = torch.zeros_like(d2)

        # Compute activations and assign them to their respective places in full_Gamma
        batch_indices = torch.arange(Z.shape[0], device=self.parent.device).unsqueeze(1)
        full_Gamma[batch_indices, self.parent.matching_clusters] = torch.exp(-d2[batch_indices, self.parent.matching_clusters]) + 1e-30

        return full_Gamma
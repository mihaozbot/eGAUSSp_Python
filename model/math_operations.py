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
            try: 
                S_inv = torch.linalg.inv(Sigma[non_single_sample_mask])
            except:
                S_inv = torch.linalg.pinv(Sigma[non_single_sample_mask])
            diff = (z_expanded[non_single_sample_mask] - mu[non_single_sample_mask]).unsqueeze(-1)
            d2_mahalanobis = torch.bmm(torch.bmm(diff.transpose(1, 2), S_inv), diff).squeeze()
            d2[non_single_sample_mask] = d2_mahalanobis

        if (d2 < 0).any():
            d2[d2<0]= inf
            print("Critical error! Negative distance detected in Gamma computation, which should be impossible")

        # Compute activations for the candidate clusters
        Gamma = torch.exp(-d2)#*scaling_factor

        if torch.isnan(Gamma).any().item():
            print("Critical error! NaN detected in Gamma computation")

        # Expand activations and distances to the full set of clusters
        full_Gamma = torch.zeros(self.parent.c, dtype=torch.float32, device=self.parent.device)
        full_Gamma[self.parent.matching_clusters] = Gamma

        return full_Gamma

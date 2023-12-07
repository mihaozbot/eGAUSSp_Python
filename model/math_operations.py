import torch

# Attempt to load the line_profiler extension
try:
    from line_profiler import LineProfiler
    profile = LineProfiler()  # If line_profiler is available, use it
except ImportError:
    # If line_profiler is not available, define a dummy profile decorator
    def profile(func): 
        return func
    
class MathOps():
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim

    @profile
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
        
        # Compute distances for clusters with a single sample
        if single_sample_mask.sum() > 0:
            diff_single_sample = z_expanded[single_sample_mask] - mu[single_sample_mask]
            inv_cov_diag = torch.linalg.pinv(self.parent.S_0).repeat(single_sample_mask.sum(), 1, 1).diagonal(dim1=-2, dim2=-1)
            d2_single_sample = torch.sum(diff_single_sample**2 * inv_cov_diag, dim=1)
            d2[single_sample_mask] = d2_single_sample


        # Compute Mahalanobis distances for other clusters
        non_single_sample_mask = ~single_sample_mask
        if non_single_sample_mask.sum() > 0:
            S_inv = torch.linalg.pinv(Sigma[non_single_sample_mask])
            diff = (z_expanded[non_single_sample_mask] - mu[non_single_sample_mask]).unsqueeze(-1)
            d2_mahalanobis = torch.bmm(torch.bmm(diff.transpose(1, 2), S_inv), diff).squeeze()
            d2[non_single_sample_mask] = d2_mahalanobis

        # Compute activations for the candidate clusters
        Gamma = torch.exp(-d2) #*scaling_factor

        # Expand activations and distances to the full set of clusters
        full_Gamma = torch.zeros(self.parent.c, dtype=torch.float32, device=self.parent.device)
        full_Gamma[self.parent.matching_clusters] = Gamma

        return full_Gamma

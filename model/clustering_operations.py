
import torch
import numpy as np

# Attempt to load the line_profiler extension
try:
    from line_profiler import LineProfiler
    profile = LineProfiler()  # If line_profiler is available, use it
except ImportError:
    # If line_profiler is not available, define a dummy profile decorator
    def profile(func): 
        return func
    
class ClusteringOps:
    def __init__(self, parent):
        self.parent = parent # Reference to the parent class
        self.feature_dim = parent.feature_dim # Number of features in the dataset
        self.Gamma_max = np.exp(-parent.num_sigma**2) # Maximum value of Gamma (used to determine if a new cluster should be added)

    @profile             
    def _add_new_cluster(self, z, label):
        ''' Add a new cluster to the model. This is called when no matching clusters are found or when the Gamma value is too low.'''
        
        self.update_S_0() # Update S_0 based on the new sample
        
        # Ensure the parameters have enough space
        self.parent.overseer.ensure_capacity(self.parent.c+2)

        # Perform the updates in place
        with torch.no_grad():  # Temporarily disable gradient tracking
            self.parent.mu.data[self.parent.c] = z
            self.parent.S.data[self.parent.c] = self.parent.S_0
            self.parent.n.data[self.parent.c] = 1.0

        # Update cluster_labels
        # If cluster_labels is not a Parameter and does not require gradients, update as a regular tensor
        self.parent.cluster_labels[self.parent.c] = label
    
        #Add a new Gamma value for the new cluster equal to 1 (Gamma is the weight of the cluster) 
        self.parent.Gamma = torch.cat((self.parent.Gamma, torch.tensor([1.0], dtype=torch.float32, device=self.parent.device)))
        
        self.parent.matching_clusters = torch.cat((self.parent.matching_clusters, torch.tensor([self.parent.c], device=self.parent.device)))

        self.parent.c += 1# Increment the number of clusters
    
    @profile 
    def update_S_0(self):
        ''' Update the smallest cluster covariance matrix based on global statistics, before adding a new cluster. '''
        
        #Compute the parameter spread from the global statistics
        S_0 = (self.parent.s_glo)**2/(self.parent.c_max)

        #Update smallest cluster covariance matrix
        self.parent.S_0 = torch.diag(S_0)
        self.parent.S_0 = torch.max(self.parent.S_0, self.parent.S_0_initial)

    @profile      
    def _increment_cluster(self, z):
        ''' Decide whether to increment an existing cluster or add a new cluster based on the current state. '''
        
        #if self.parent.enable_debugging and (j >= len(self.parent.mu) or j < 0):
        #    logging.warning(f"Warning rule increment! Invalid cluster index: {j}. Valid indices are between 0 and {len(self.parent.mu)-1}.")
           
        #Normalize membership functions 
        NGamma = self.parent.Gamma/torch.sum(self.parent.Gamma)  
        
        # Calculate the error for the data point for each cluster
        z_expanded = z.unsqueeze(0).expand(self.parent.c, self.parent.feature_dim)
        e_c = z_expanded - self.parent.mu[0:self.parent.c]  # shape [self.parent.current_capacity, feature_dim]

        # Update cluster means
        self.parent.mu[0:self.parent.c] += NGamma.unsqueeze(1) / (1 + self.parent.n[0:self.parent.c].unsqueeze(1)) * e_c

        # e_c_transposed for matrix multiplication, shape [self.parent.current_capacity, feature_dim, 1]
        e_c_transposed = e_c.unsqueeze(-1)  # shape [self.parent.current_capacity, feature_dim, 1]
        self.parent.S[0:self.parent.c] += NGamma.unsqueeze(-1).unsqueeze(-1) * torch.bmm(e_c_transposed, e_c_transposed.transpose(1, 2))

        # Update number of samples in each cluster
        self.parent.n[0:self.parent.c] += NGamma

    @profile      
    def increment_or_add_cluster(self, z, label):
        ''' Increment an existing cluster if a cluster is activated enough, else add a new one'''
        
        if len(self.parent.matching_clusters) == 0:
            self._add_new_cluster(z, label)
            #logging.info(f"Info. Added new cluster for label {label} due to no matching clusters. Total clusters now: {self.parent.c}")
            return torch.tensor([1.0], device=self.parent.device)
        
        _, j_rel = torch.max(self.parent.Gamma[self.parent.matching_clusters], dim=0)
        j_abs = self.parent.matching_clusters[j_rel].item()  # Map relative index back to full list of clusters
        
        if self.parent.enable_adding and (self.parent.Gamma[j_abs] <= self.Gamma_max):
            self._add_new_cluster(z, label)
            #logging.info(f"Info. Added new cluster for label {label} due to low Gamma value. Total clusters now: {self.parent.c}")
        else:
            self._increment_cluster(z)
            
    def update_global_statistics(self, z):
        ''' Update the global mean, covariance, and count based on the new data point. '''
        
        e_glo = z - self.parent.mu_glo  # Error between the sample and the global mean
        self.parent.mu_glo += e_glo / (self.parent.n_glo + 1)  # Mean
        self.parent.S_glo += self.parent.n_glo / (self.parent.n_glo + 1) * e_glo * e_glo # Variance, (not normalized to reduce computation)
        self.parent.n_glo += 1 #Number of samples
        self.parent.s_glo = torch.sqrt(self.parent.S_glo / self.parent.n_glo) #Standard deviation
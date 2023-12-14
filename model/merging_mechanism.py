
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
import logging
import os

class MergingMechanism:
    def __init__(self, parent):
        self.parent = parent
        self.feature_dim = parent.feature_dim
        self.V_factor = (2 * torch.pi ** (self.feature_dim/2) / 
                        (self.feature_dim * torch.exp(torch.lgamma(torch.tensor(float(self.feature_dim) / 2, device=self.parent.device)))))
             
    def perform_merge(self, i_all, j_all):
        
        # Start plotting BEFORE the merge, for debugging
        if self.parent.enable_debugging: 
            plt.figure(figsize = (6, 6))
            self.plot_cluster(i_all, 'Cluster i (Before)', 'blue')
            self.plot_cluster(j_all, 'Cluster j (Before)', 'red')

        #Compute combined number of samples and new center
        n_ij = self.parent.n[i_all] + self.parent.n[j_all]
        mu_ij = (self.parent.n[i_all] * self.parent.mu[i_all]+ self.parent.n[j_all] * self.parent.mu[j_all]) / n_ij
        
        #Compute mean offset
        mu_diff = self.parent.mu[i_all] - self.parent.mu[j_all]

        # Determine the correct S values to use based on the number of samples in each cluster
        # Check if either cluster has only one sample
        S_i = self.parent.S_0 if self.parent.n[i_all] == 1 else self.parent.S[i_all]
        S_j = self.parent.S_0 if self.parent.n[j_all] == 1 else self.parent.S[j_all]

        # Calculate the new covariance matrix for the merged cluster
        S_ij = (S_i + S_j) + (self.parent.n[i_all] * self.parent.n[j_all] / n_ij * torch.outer(mu_diff, mu_diff))

        # Perform the merging operation
        self.parent.mu[i_all] = mu_ij
        self.parent.S[i_all] = S_ij
        self.parent.n[i_all] = n_ij

        # Update Gamma values 
        self.parent.Gamma[i_all] = 0 #self.parent.Gamma[i_all]
                
        # Use RemovalMechanism to remove the j-th cluster
        self.parent.removal_mech.remove_cluster(j_all)
        
        # Visualize the clusters after merging, for debugging
        if self.parent.enable_debugging:
            self.plot_cluster(i_all, 'Merged Cluster (After)', 'green')

            # Set the title
            plt.title(f"Clusters Before & After Merging: {i_all} and {j_all}")

            # Save the figure
            output_dir = 'Merging'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filename = f"merge_plot_{int(i_all)}_and_{int(j_all)}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)

            # Close the figure
            plt.close()
          
    def plot_cluster(self, index, label, color, alpha=1):
        """Helper function to plot a cluster given its index. Mainly for debugging the merging procedure."""

        max_values = self.parent.mu_glo.cpu() + 3*self.parent.s_glo.cpu()
        min_values = self.parent.mu_glo.cpu() - 3*self.parent.s_glo.cpu()

        # Set plot limits
        plt.xlim(min_values[0], max_values[0])
        plt.ylim(min_values[1], max_values[1])
        
        mu = self.parent.mu[index].clone().cpu().detach().numpy()
        S = self.parent.S[index].clone().cpu().detach().numpy()/(self.parent.n[index].cpu().detach().numpy()-1)

        # Only use the first two dimensions of mu and S
        mu_2d = mu[:2]
        S_2d = S[:2, :2]

        plt.scatter(mu_2d[0], mu_2d[1], s=100, marker=MarkerStyle('x'), color=color, label=label, alpha=alpha)

        # Assuming 2D data, plot ellipse for the covariance matrix
        vals, vecs = np.linalg.eigh(S_2d)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 3 * np.sqrt(vals)
        ell = Ellipse(xy=mu_2d, width=width, height=height, angle=angle, edgecolor=color, lw=2, facecolor='none', alpha=alpha)
        plt.gca().add_patch(ell)

    def compute_merging_condition(self):
        """
        Compute the volume 'V' between pairs of clusters in the provided set of matching clusters. Merge clusters that meet the specified condition.
        """
        
        # Prepare necessary tensors for valid clusters
        n = self.parent.n[self.valid_clusters]
        mu = self.parent.mu[self.valid_clusters]
        S = self.parent.S[self.valid_clusters]

        # Compute Sigma_ij only for valid clusters (vectorized computation)
        mu_diff = mu[:, None, :] - mu[None, :, :]
        mu_outer_product = mu_diff[..., None] * mu_diff[:, :, None, :]
        n_ij_matrix = n[:, None] + n[None, :]
        Sigma_ij = (S[None, :, :, :] + S[:, None, :, :]) + (n[:, None, None, None] * n[None, :, None, None] / n_ij_matrix[:, :, None, None]) * mu_outer_product
        Sigma_ij = Sigma_ij/(n_ij_matrix[:, :, None, None]-1)

        # Compute log-determinant for numerical stability
        det_matrix = torch.exp(torch.linalg.slogdet(Sigma_ij)[1]) # [1] is the log determinant

        # Vectorized computation of volume V for upper triangle
        self.V = torch.sqrt(det_matrix)

        #Extract the upper triangle
        self.V = torch.triu(self.V , diagonal=0)

        # Compute merging condition kappa
        kappa = self.compute_kappa_matrix()
    
        return kappa


    def update_merging_condition(self, i, j):
        
        j_all = self.valid_clusters[j]
        #Note due to how torch.min() works we assume that i_all < j_all always holds

        #If j_all happends to be the last active cluster just remove it from self.valid_clusters and self.kappa
        #If self.parent.c-1 is not on the list, we need to remove j
         #Note the -1 is because self.parent.c starts at 1
        #We want to know if self.parent.c-1 is on the list...If self.valid_clusters[-1] is sorted, and it should be, the last element is the largest index.
        if (j_all == (self.parent.c-1)) or (self.valid_clusters[-1] != (self.parent.c-1)):
            # Remove the j-th element from valid_clusters
            self.valid_clusters = torch.cat((self.valid_clusters[:j], self.valid_clusters[j + 1:]))

            # Remove the j-th row and column from V
            self.V = torch.cat((self.V[:j], self.V[j + 1:]), dim=0)  # Remove j-th row
            self.V = torch.cat((self.V[:, :j], self.V[:, j + 1:]), dim=1)  # Remove j-th column

        else:
            # (self.parent.c - 1) is on the list, and the last cluster was copied to the j-th place
            # Remove the last element from valid_clusters
            self.valid_clusters = self.valid_clusters[:-1] 

            # Move the last row and column of V to the j-th position, then remove the last row and column
            self.V[j] = self.V[-1]  # Move last row to j-th position
            self.V[:, j] = self.V[:, -1]  # Move last column to j-th position
            self.V = self.V[:-1, :-1]  # Remove last row and column

        if len(self.valid_clusters) < 2:
            return #Further computation does not matter in this case
            
        # Prepare necessary tensors for valid clusters
        n = self.parent.n[self.valid_clusters]
        mu = self.parent.mu[self.valid_clusters]
        S = self.parent.S[self.valid_clusters]

        # Compute Sigma_ij only for the i-th row and column
        mu_diff = mu[i, None, :] - mu
        mu_outer_product = mu_diff[..., None] * mu_diff[:, None, :]
        n_matrix = n[i, None] + n
        Sigma = (S[i, None, :, :] + S) + (n[i, None, None, None] * n[:, None, None] / n_matrix[:, None, None]) * mu_outer_product
        Sigma = Sigma / (n_matrix[:, None, None] - 1)

        # Compute log-determinant for numerical stability
        det_matrix = torch.exp(torch.linalg.slogdet(Sigma)[1])  # [1] is the log determinant

        # Vectorized computation of volume V for upper triangle
        V_i = torch.sqrt(det_matrix)
        self.V[i,:], self.V[:,i]  = V_i, V_i
        self.V = torch.triu(self.V , diagonal=0)

        # Update kappa for the i-th row and column
        self.compute_kappa_matrix()
    
            # Debugging checks
        if self.parent.enable_debugging:

            # Convert tensors to sets for set operation (if they are not large)
            set_valid_clusters = set(self.valid_clusters.tolist())
            set_matching_clusters = set(self.parent.matching_clusters.tolist())

            # Check if valid_clusters is a subset of matching_clusters
            is_subset = set_valid_clusters.issubset(set_matching_clusters)
            if not is_subset:
                print("Is valid_clusters a subset of matching_clusters:", is_subset)

    def merge_clusters(self):
        
        # Track if any merge has occurred
        merge_occurred = False
        
        kappa_min = torch.min(self.kappa)
        
        if kappa_min < self.parent.kappa_join:

            # Find the indices with the minimum kappa value
            i_valid, j_valid = (self.kappa == kappa_min).nonzero(as_tuple=True)
            i_valid, j_valid = i_valid[0].item(), j_valid[0].item()  # Convert tensor indices to integers

            # Map local indices i_valid, j_valid to global indices i_all, j_all
            i_all = self.valid_clusters[i_valid]
            j_all = self.valid_clusters[j_valid]

            #Recompute condition for i and potenitionally j
            self.update_merging_condition(i_valid, j_valid)
            
            #Actual merging of clusters
            with torch.no_grad():
                self.perform_merge(i_all, j_all)

            merge_occurred = True
            
        return merge_occurred  # Return True if any merge happened, otherwise False
  
    def merging_mechanism(self, max_iterations=100):
    
        iteration = 0 # Iteration counter
        
        #Check which clusters have the necesary conditions to allow them to merge
        #The point is that we do not want to check all the clusters at every time step, but only the relevant ones
        if self.parent.c > 10*np.sqrt(self.parent.feature_dim):
            self.valid_clusters = self.parent.matching_clusters
        else:
            threshold = np.exp(-(2*self.parent.num_sigma) ** 2)
            self.valid_clusters = self.parent.matching_clusters[(self.parent.Gamma[self.parent.matching_clusters] > threshold)*
                                                                (self.parent.n[self.parent.matching_clusters] >= self.parent.feature_dim)] #np.sqrt(
        
        #Compute the initial merging candidates
        self.compute_merging_condition()

        #Merge until you can not merge no mo
        merge = True  # initial condition to enter the loop
        while merge and iteration < max_iterations:
            
            if len(self.valid_clusters) < 2:
                break

            #Check merging condition, merge rules, and return True if merge happened
            merge = self.merge_clusters()
            iteration += 1

    def compute_kappa_matrix(self):
        # Create diagonal matrix with shape (c, c) containing V[i, i] + V[j, j] for all i and j
        diag_sum = self.V.diag().unsqueeze(0) + self.V.diag().unsqueeze(1)

        # Create the upper triangular part of kappa matrix
        self.kappa = torch.triu(self.V / diag_sum, diagonal=1)

        #Compute volume of default cluster covariance matrix
        V_S_0 = torch.sqrt(torch.prod(torch.diag(self.parent.S_0)))
       
        #Compare cluster volume to standard volume
        V_ratio = self.V/V_S_0
        
        # Filtering kappa based on conditions
        kappa_filter = (self.kappa == 0) + (V_ratio > np.sqrt(self.feature_dim))
        self.kappa[kappa_filter] = float("inf")

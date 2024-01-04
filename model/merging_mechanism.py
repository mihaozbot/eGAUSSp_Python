
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
        S_i = self.parent.S_0 if self.parent.n[i_all] < 2 else self.parent.S[i_all]
        S_j = self.parent.S_0 if self.parent.n[j_all] < 2 else self.parent.S[j_all]

        # Calculate the new covariance matrix for the merged cluster
        S_ij = (S_i + S_j) + (self.parent.n[i_all] * self.parent.n[j_all] / n_ij * torch.outer(mu_diff, mu_diff))

        score_ij = (self.parent.n[i_all] * self.parent.score[i_all]+ self.parent.n[j_all] * self.parent.score[j_all]) / n_ij
        num_pred_ij = self.parent.num_pred[i_all] + self.parent.num_pred[j_all]

        # Perform the merging operation
        self.parent.mu[i_all] = mu_ij
        self.parent.S[i_all] = S_ij
        self.parent.n[i_all] = n_ij

        #Merge score and number of predictions
        self.parent.score[i_all] = score_ij
        self.parent.num_pred[i_all] = num_pred_ij

        # Update Gamma values 
        self.parent.Gamma[i_all] = 0 #self.parent.Gamma[i_all]
        
        self.parent.S_inv[i_all] = torch.linalg.inv((self.parent.S[i_all] / self.parent.n[i_all]) * self.parent.feature_dim)

         # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(self.parent.S_inv[i_all])
        
        # Check if all eigenvalues are positive (matrix is positive definite)
        if not torch.all(eigenvalues > 0):
            # Handle the case where the matrix is not positive definite
            # Depending on your requirements, you might set a default value or handle it differently
            print("Matrix is not positive definite for index", i_all)
            # Example: set S_inv[j] to a matrix of zeros or some other default value
            # Adjust the dimensions as needed
            self.parent.S_inv[i_all] = torch.zeros_like(self.parent.S[i_all])

        # Use RemovalMechanism to remove the j-th cluster
        self.parent.removal_mech.remove_cluster(j_all)
        
        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvalsh(self.parent.S_inv[j_all])
        
        # Check if all eigenvalues are positive (matrix is positive definite)
        if not torch.all(eigenvalues > 0):
            # Handle the case where the matrix is not positive definite
            # Depending on your requirements, you might set a default value or handle it differently
            print("Matrix is not positive definite for index", i_all)
            # Example: set S_inv[j] to a matrix of zeros or some other default value
            # Adjust the dimensions as needed
            self.parent.S_inv[i_all] = torch.zeros_like(self.parent.S[i_all])

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
        S = self.parent.S[index].clone().cpu().detach().numpy()/(self.parent.n[index].cpu().detach().numpy())

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

    def compute_volume(self):
        
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
        n_matrix = n[:, None] + n[None, :]
        Sigma = (S[None, :, :, :] + S[:, None, :, :]) + (n[:, None, None, None] * n[None, :, None, None] / n_matrix[:, :, None, None]) * mu_outer_product
        Sigma = Sigma/(n_matrix[:, :, None, None]-1)

        # Compute log-determinant for numerical stability
        # Compute log-determinant for numerical stability
        #L = torch.linalg.cholesky(Sigma)
        #det_matrix = torch.prod(torch.diag(L))**2
        det_matrix = torch.exp(torch.linalg.slogdet(Sigma)[1]) # [1] is the log determinant

        # Vectorized computation of volume V for upper triangle
        self.V = det_matrix**(1/self.parent.feature_dim)

        #Extract the upper triangle
        #self.V = torch.triu(self.V , diagonal=0)

    def update_volume(self, i):
        # i is the index in self.valid_clusters for the cluster that has just merged

        # Extracting the i-th cluster's parameters
        n_i = self.parent.n[self.valid_clusters[i]]
        mu_i = self.parent.mu[self.valid_clusters[i]]
        S_i = self.parent.S[self.valid_clusters[i]]

        # Prepare necessary tensors for all other valid clusters
        n_other = self.parent.n[self.valid_clusters]
        mu_other = self.parent.mu[self.valid_clusters]
        S_other = self.parent.S[self.valid_clusters]

        # Compute Sigma_ij for the i-th cluster against all others
        mu_diff = mu_i[None, :] - mu_other
        mu_outer_product = mu_diff[..., None] * mu_diff[:, None, :]
        n_matrix = n_i + n_other
        Sigma = (S_i[None, :, :] + S_other) + (n_i * n_other[:, None, None] / n_matrix[:, None, None]) * mu_outer_product
        Sigma = Sigma / (n_matrix[:, None, None] - 1)

        # Compute log-determinant for numerical stability
        det_matrix = torch.exp(torch.linalg.slogdet(Sigma)[1])  # [1] is the log determinant

        # Compute the volume V for the i-th row/column
        V_i = det_matrix**(1/self.parent.feature_dim)

        # Update the i-th row and column of the volume matrix
        self.V[i, :] = V_i
        self.V[:, i] = V_i

    def update_merging_condition(self, i, j):
        # i and j are local to self.valid_clusters
        #i_all = self.valid_clusters[i]
        j_all = self.valid_clusters[j]

        # Update V for the i-th row and column
        self.update_volume(i)

        # Update kappa for the i-th row and column
        self.update_kappa(j)

        # Test the partial updates
        #self.test_update_volume_kappa(i)

        # Handle removal of the j-th cluster
        if j_all == (self.parent.c-1) or (self.valid_clusters[-1] != (self.parent.c-1)):
            self.valid_clusters = self.valid_clusters[self.valid_clusters != j_all]
            # Adjust V and kappa matrices
            self.V = torch.cat((self.V[:j], self.V[j + 1:]), dim=0)  # Remove j-th row
            self.V = torch.cat((self.V[:, :j], self.V[:, j + 1:]), dim=1)  # Remove j-th column
            self.kappa = torch.cat((self.kappa[:j], self.kappa[j + 1:]), dim=0)
            self.kappa = torch.cat((self.kappa[:, :j], self.kappa[:, j + 1:]), dim=1)
        else:
            self.valid_clusters = self.valid_clusters[:-1]  # Remove last element
            # Adjust V and kappa matrices
            self.V = self.V[:-1, :-1]  # Remove last row and column
            self.kappa = self.kappa[:-1, :-1]

    def merge_clusters(self):
        
        # Track if any merge has occurred
        merge_occurred = False
        
        kappa_min = torch.min(self.kappa[self.kappa==self.kappa])
        
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
    
    def merging_mechanism(self, max_iterations=1000):
    
        iteration = 0 # Iteration counter
        
        #Check which clusters have the necesary conditions to allow them to merge
        #The point is that we do not want to check all the clusters at every time step, but only the relevant ones
        #if self.parent.c > 10*np.sqrt(self.parent.feature_dim):
        #    self.valid_clusters = self.parent.matching_clusters
        #else:
        threshold = np.exp(-(self.parent.num_sigma) ** 2)
        self.valid_clusters = self.parent.matching_clusters[(self.parent.Gamma[self.parent.matching_clusters] > threshold)*
                                                            (self.parent.n[self.parent.matching_clusters] >= self.parent.kappa_n)] #np.sqrt(
        if len(self.valid_clusters) < 2:
            return

        #Compute the volume of the combined clusters
        self.compute_volume()

        # Compute merging condition kappa
        self.compute_kappa()
        
        #Merge until you can not merge no mo
        merge = True  # initial condition to enter the loop
        while merge and iteration < max_iterations:

            if len(self.valid_clusters) < 2:
                break
            
            # Check if all elements in self.parent.cluster_labels[self.parent.matching_clusters] are the same
            labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) == 1
            if not labels_consistency_check:
                print("Critical error: Labels consistency in matching clusters in merging mechanism:", labels_consistency_check)

            #Check merging condition, merge rules, and return True if merge happened
            merge = self.merge_clusters()
            iteration += 1

    def compute_kappa(self):
        # Create diagonal matrix with shape (c, c) containing V[i, i] + V[j, j] for all i and j
        diag_sum = self.V.diag().unsqueeze(0) + self.V.diag().unsqueeze(1)

        # Create the upper triangular part of kappa matrix
        self.kappa = (self.V / diag_sum)

        # Replace NaN values in kappa with zeros
        nan_mask = torch.isnan(self.kappa)
        self.kappa[nan_mask] = 0

        # Compute volume of default cluster covariance matrix
        V_S_0 = torch.prod(torch.diag(self.parent.S_0)**(1/self.parent.feature_dim))
    
        # Compare cluster volume to standard volume
        V_ratio = (self.V/V_S_0)
        
        # Filtering kappa based on conditions
        kappa_filter = (self.kappa == 0) + (V_ratio > 3)
        self.kappa[kappa_filter] = float("inf")
        self.kappa.fill_diagonal_(float("inf"))


    def update_kappa(self, i):
        # i is the index in self.valid_clusters for the cluster that has just merged

        # Compute the diagonal sum for the i-th row and column
        diag_sum_i = self.V[i, i] + self.V.diag()

        # Update kappa for the i-th row
        self.kappa[i, :] = (self.V[i, :] / diag_sum_i)

        # Update kappa for the i-th column
        # Since kappa matrix is symmetric, we can copy the i-th row to the i-th column
        self.kappa[:, i] = self.kappa[i, :]

        # Replace NaN values in kappa with zeros
        nan_mask_row = torch.isnan(self.kappa[i, :])
        nan_mask_col = torch.isnan(self.kappa[:, i])
        self.kappa[i, nan_mask_row] = 0
        self.kappa[nan_mask_col, i] = 0

        # Compute volume of default cluster covariance matrix
        V_S_0 = torch.prod(torch.diag(self.parent.S_0)**(1/self.parent.feature_dim))

        # Compare cluster volume to standard volume for the i-th row and column
        V_ratio_i = (self.V[i, :] / V_S_0)
        V_ratio_col = (self.V[:, i] / V_S_0)

        # Filtering kappa based on conditions for the i-th row and column
        kappa_filter_row = (self.kappa[i, :] == 0) + (V_ratio_i > self.parent.N_r)
        kappa_filter_col = (self.kappa[:, i] == 0) + (V_ratio_col > self.parent.N_r)
        self.kappa[i, kappa_filter_row] = float("inf")
        self.kappa[kappa_filter_col, i] = float("inf")

        # Ensure the diagonal of kappa remains infinity
        self.kappa.fill_diagonal_(float("inf"))

    def test_update_volume_kappa(self, i):
        # Perform full computations
        self.compute_volume()
        self.compute_kappa()

        # Store the results from the full computations
        full_V = self.V.clone()
        full_kappa = self.kappa.clone()

        # Perform partial updates
        self.update_volume(i)
        self.update_kappa(i)

        # Compare the results
        volume_match = torch.allclose(full_V, self.V, atol=1e-6)
        kappa_match = torch.allclose(full_kappa, self.kappa, atol=1e-6)

        # Print the result
        if volume_match and kappa_match:
            print("Test Passed: Partial updates to volume and kappa are correct.")
        else:
            print("Test Failed: Discrepancy found in partial updates to volume or kappa.")

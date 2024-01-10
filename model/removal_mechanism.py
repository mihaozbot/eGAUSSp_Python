from re import S
import torch
import numpy as np
import torch.nn.functional as F

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent

    def update_score(self, label):
        if len(torch.unique(self.parent.cluster_labels[:self.parent.c],dim= 0)) < 2:
            return 
        
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Masks for correct and incorrect clusters
        correct_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 1)
        incorrect_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 0)

        # Sorted indices by activation
        sorted_indices = torch.argsort(normalized_gamma, descending=True)

        # Lists to keep track of indices to update
        correct_to_update = []
        wrong_to_update = []
        first_correct_index = None
        first_wrong_index = None

        
        for index in sorted_indices:
            if correct_clusters_mask[index] and first_wrong_index is None:
                correct_to_update.append(index)
                if first_correct_index is None:
                    first_correct_index = index
            elif incorrect_clusters_mask[index] and first_correct_index is None:
                wrong_to_update.append(index)
                if first_wrong_index is None:
                    first_wrong_index = index

        # Update correct clusters
        if correct_to_update:
            self.update_cluster_scores(correct_to_update, label, correct=True)

        # Update wrong clusters
        if wrong_to_update:
            self.update_cluster_scores(wrong_to_update, label, correct=False)

        # Extra update for the first correct cluster
        if first_correct_index is not None:
            self.update_cluster_scores([first_correct_index], label, correct=True)

        # Extra update for the first wrong cluster
        if first_wrong_index is not None:
            self.update_cluster_scores([first_wrong_index], label, correct=False)


    def update_cluster_scores(self, cluster_indices, label, correct=True):
        cluster_indices = torch.tensor(cluster_indices)

        score_increment = 1.0 if correct else 0.0

        # Create a tensor of score increments with the same length as cluster_indices
        score_increment_tensor = torch.full((len(cluster_indices),), score_increment, dtype=torch.float32, device=self.parent.device)

        self.parent.num_pred[cluster_indices] += 1

        # Calculate new scores
        prev_scores = self.parent.score[cluster_indices]
        num_preds = self.parent.num_pred[cluster_indices]
        new_scores = ((num_preds - 1) * prev_scores + score_increment_tensor) / num_preds

        self.parent.score[cluster_indices] = new_scores

    def remove_overlapping(self):
        # Compute the volume and kappa initially
        self.parent.merging_mech.compute_volume()
        self.parent.merging_mech.compute_kappa()

        with torch.no_grad():
            while len(self.parent.merging_mech.valid_clusters) > self.parent.c_max:
                # Identify the smallest kappa value

                if len(self.parent.merging_mech.valid_clusters) < 2:
                    break

                # Find the pair of clusters with the smallest kappa value
                rows, cols = torch.where(self.parent.merging_mech.kappa)
                if rows.nelement() == 0 or cols.nelement() == 0:
                    break
            
                # Flatten the kappa matrix and find the global minimum
                kappa_flat = self.parent.merging_mech.kappa.view(-1)
                min_kappa_value, min_kappa_flat_idx = torch.min(kappa_flat, 0)


                # Check if the minimum kappa value is infinite
                if torch.isinf(min_kappa_value):
                    break
                
                # Convert the flat index back to a 2D index
                row, col = divmod(min_kappa_flat_idx.item(), self.parent.merging_mech.kappa.shape[1])

                # Get the scores of the clusters in the pair
                score_row = self.parent.score[self.parent.merging_mech.valid_clusters[row]].item() #*self.parent.num_pred[self.parent.merging_mech.valid_clusters[row]].item()
                score_col = self.parent.score[self.parent.merging_mech.valid_clusters[col]].item() #*self.parent.num_pred[self.parent.merging_mech.valid_clusters[col]].item()

                cluster_to_remove_idx = []
                # Determine which cluster of the pair to remove
                if score_row < score_col:
                    cluster_to_remove_idx = row
                elif score_row >= score_col:
                    cluster_to_remove_idx = col

                # Determine which cluster of the pair to remove based on the smallest score
                cluster_to_remove = self.parent.merging_mech.valid_clusters[cluster_to_remove_idx]

                # Remove the selected cluster
                self.remove_cluster(cluster_to_remove)

                # Update kappa and valid_clusters
                self.parent.merging_mech.valid_clusters = torch.cat([
                    self.parent.merging_mech.valid_clusters[:cluster_to_remove_idx],
                    self.parent.merging_mech.valid_clusters[cluster_to_remove_idx + 1:]
                ])
                self.parent.merging_mech.kappa = torch.cat([
                    self.parent.merging_mech.kappa[:cluster_to_remove_idx],
                    self.parent.merging_mech.kappa[cluster_to_remove_idx + 1:]
                ], dim=0)
                self.parent.merging_mech.kappa = torch.cat([
                    self.parent.merging_mech.kappa[:, :cluster_to_remove_idx],
                    self.parent.merging_mech.kappa[:, cluster_to_remove_idx + 1:]
                ], dim=1)


    def remove_aged(self, c_max):
        if self.parent.c < 2:
            return

        # Determine how many clusters to remove to meet the desired count
        num_clusters_to_remove = len(self.parent.matching_clusters) - c_max

        if num_clusters_to_remove > 0:
            # Sort the clusters based on age, highest (oldest) first
            sorted_clusters_by_age = sorted(self.parent.matching_clusters, 
                                            key=lambda idx: self.parent.age[idx], 
                                            reverse=True)

            # Get the indices of clusters to remove, oldest first
            indices_to_remove = sorted_clusters_by_age[:num_clusters_to_remove]

            with torch.no_grad():
                # Remove the selected clusters
                for index in indices_to_remove:
                    self.remove_cluster(index)

    def remove_score(self,c_max):
        if self.parent.c < 2:
            return

        # Determine how many clusters to remove to meet the desired count
        num_clusters_to_remove = len(self.parent.matching_clusters) - c_max

        if num_clusters_to_remove > 0:
            # Create a composite score for each cluster
            composite_scores = [(self.parent.score[i], -self.parent.num_pred[i], i) for i in self.parent.matching_clusters]

            # Sort the clusters by composite score, excluding those with a score of 1
            sorted_indices = sorted([idx for score, _, idx in composite_scores if score < 1], 
                                    key=lambda idx: (self.parent.score[idx], -self.parent.num_pred[idx]), 
                                    reverse=True)

            # Get the indices of clusters to remove in descending order, up to the number needed
            indices_to_remove = sorted(sorted_indices[:num_clusters_to_remove], reverse=True)

            with torch.no_grad():
                # Remove the selected clusters
                for index in indices_to_remove:
                    self.remove_cluster(index)

    def remove_irrelevant(self, c_max):
        if self.parent.c < 2:
            return
        
        # Calculate how many clusters to remove
        num_clusters_to_remove = len(self.parent.matching_clusters) - c_max
        if num_clusters_to_remove > 0:
            # Filter out clusters with num_pred == 1
            clusters_eligible_for_removal = [i for i in self.parent.matching_clusters if self.parent.num_pred[i] > 1]

            # Sort the eligible clusters first by num_pred (ascending) and then by score (descending)
            sorted_indices = sorted(clusters_eligible_for_removal, key=lambda i: (self.parent.num_pred[i], -self.parent.score[i]))

            # Indices of clusters to remove in descending order
            indices_to_remove = sorted(sorted_indices[:num_clusters_to_remove], reverse=True)

            with torch.no_grad():
                # Remove selected clusters
                for index in indices_to_remove:
                    self.remove_cluster(index)
                    

    def removal_mechanism(self, c_max):

        #self.remove_score(c_max)
        self.remove_aged(c_max)


    def remove_cluster(self, cluster_index):
        '''Remove a specified cluster by replacing it with the last active cluster and updating relevant parameters. '''
        #Copy everhing from the last active cluster to the remove cluster index
        #Then if the last active index was on the matching clusters list it needs to be removed, 
        # else the index of the memoved cluster needs to be removed
        
        last_active_index = self.parent.c - 1

        #Instead of removing just copy last cluster in the place of the removed on
        if cluster_index != last_active_index: #The target cluster is not the last cluster

             # Update matching_clusters list
            #The last index was moved to j, 
            #we need to check if this new j is the right label
            if not torch.equal(self.parent.cluster_labels[cluster_index], self.parent.cluster_labels[last_active_index]):
                self.parent.matching_clusters = self.parent.matching_clusters[self.parent.matching_clusters != cluster_index]

            #If the last_active_index is on the matching cluster list, it has to be removed as it will not exist after this:
            elif self.parent.matching_clusters[-1] == last_active_index:
                self.parent.matching_clusters = self.parent.matching_clusters[:-1]

            # Move the last active cluster to the position of the cluster to be removed
            self.parent.mu[cluster_index] = self.parent.mu[last_active_index]
            self.parent.S[cluster_index] = self.parent.S[last_active_index]
            self.parent.n[cluster_index] = self.parent.n[last_active_index]

            self.parent.S_inv[cluster_index] = self.parent.S_inv[last_active_index]
            
            self.parent.age[cluster_index] = self.parent.age[last_active_index]

            self.parent.score[cluster_index] = self.parent.score[last_active_index]
            self.parent.num_pred[cluster_index] = self.parent.num_pred[last_active_index]

            # Update the label of the cluster that is moved
            self.parent.cluster_labels[cluster_index] = self.parent.cluster_labels[last_active_index]

            # Update the Gamma value for the cluster 
            self.parent.Gamma[cluster_index] = self.parent.Gamma[last_active_index]

            #if last_active_index == self.parent.matching_clusters[-1]: #if the last cluster is on the list, it is the last elements as the list is ordered
                # The last cluster was moved to the where the jth index is pointing so just remove the last cluster from the list
                # Remove the last index if last_active_index is in matching_clusters
            #    self.parent.matching_clusters = self.parent.matching_clusters[:-1]
            #else: #The last cluster is not a matching cluster so we can remove the cluster index
            # Else, remove the cluster_index from matching_clusters
            
            #We need to remove j from the list

        else: #The cluster_index is the last index, so we just need to remove it
            self.parent.matching_clusters = self.parent.matching_clusters[:-1]
         
        #If the cluster was the last cluster just reduce the number of clusters 
        # Decrement the count of active clusters
        self.parent.c -= 1

        # Check if all elements in self.parent.cluster_labels[self.parent.matching_clusters] are the same
        #labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) < 2
        #if not labels_consistency_check:
        #    print("Critical error: Labels consistency in matching clusters after removal:", labels_consistency_check)

        # # Compute eigenvalues
        # eigenvalues = torch.linalg.eigvalsh(self.parent.S_inv[cluster_index])

        # # Check if all eigenvalues are positive (matrix is positive definite)
        # if not torch.all(eigenvalues > 0):
        #     # Handle the case where the matrix is not positive definite
        #     # Depending on your requirements, you might set a default value or handle it differently
        #     print("Matrix is not positive definite for index", cluster_index)
        #     # Example: set S_inv[j] to a matrix of zeros or some other default value
        #     # Adjust the dimensions as needed
        #     self.parent.S_inv[j] = torch.zeros_like(self.parent.S[cluster_index])


        # Debugging checks
        if self.parent.enable_debugging:

            # Check if cluster_index is in matching_clusters and points to the same data as the old last_active_index
            label_match_check = True
            if cluster_index not in self.parent.matching_clusters:
                # Check if the label of cluster_index is among the labels of clusters in matching_clusters
                label_match_check = self.parent.cluster_labels[cluster_index] in self.parent.cluster_labels[self.parent.matching_clusters]
                print("Label of cluster index is in labels of matching clusters:", label_match_check)
            

    def select_clusters_nearmiss(self):
        all_class_distances = self.vectorized_bhattacharyya_distance()

        selected_clusters = {}
        for class_label, distances in all_class_distances.items():
            # Find the closest clusters for each class
            min_distances, indices = torch.min(distances, dim=1)
            sorted_indices = torch.argsort(min_distances)

            # Select up to c_max clusters for this class
            selected_clusters_for_class = sorted_indices[:self.parent.c_max].tolist()
            selected_clusters[class_label] = selected_clusters_for_class

        selected_clusters_tensor = torch.tensor(selected_clusters[0], dtype=torch.int32, device=self.parent.device)
        all_clusters = torch.arange(self.parent.c, dtype=torch.int32, device=self.parent.device)

        # Use the mask to filter out the elements
        indices_to_remove = all_clusters[~torch.isin(all_clusters, selected_clusters_tensor)]
                
        with torch.no_grad():
            # Remove additional low scoring clusters
            for index in indices_to_remove:
                self.remove_cluster(index)
        #return selected_clusters


    def vectorized_bhattacharyya_distance(self):
        # Extract means (mu) and covariance matrices (sigma) for all valid clusters
        mu = self.parent.mu[:self.parent.c]
        sigma = (self.parent.S[:self.parent.c] / self.parent.n[:self.parent.c].unsqueeze(1).unsqueeze(2))

        # Get unique class labels
        unique_classes = torch.unique(self.parent.cluster_labels[:self.parent.c])

        # Dictionary to store Bhattacharyya distances for each class against others
        class_distances = {}

        for cls in unique_classes:
            # Indices for the current class and other classes
            current_class_indices = torch.where(self.parent.cluster_labels[:self.parent.c] == cls)[0]
            other_class_indices = torch.where(self.parent.cluster_labels[:self.parent.c] != cls)[0]

            # Extract mu and sigma for current and other classes
            mu_1 = mu[current_class_indices]
            sigma_1 = sigma[current_class_indices]
            mu_2 = mu[other_class_indices]
            sigma_2 = sigma[other_class_indices]

            # Compute the vectorized Bhattacharyya distance
            distances = self._compute_bhattacharyya(mu_1, sigma_1, mu_2, sigma_2)
            class_distances[cls.item()] = distances

        return class_distances

    def _compute_bhattacharyya(self, mu_1, sigma_1, mu_2, sigma_2):
        n1, d = mu_1.shape[0], mu_1.shape[1]
        n2 = mu_2.shape[0]

        # Expand dimensions to enable broadcasting
        expanded_mu_1 = mu_1.unsqueeze(1).expand(n1, n2, d)
        expanded_sigma_1 = sigma_1.unsqueeze(1).expand(n1, n2, d, d)
        expanded_mu_2 = mu_2.unsqueeze(0).expand(n1, n2, d)
        expanded_sigma_2 = sigma_2.unsqueeze(0).expand(n1, n2, d, d)

        # Mean difference
        mean_diff = expanded_mu_2 - expanded_mu_1

        # Average covariance
        avg_sigma = (expanded_sigma_1 + expanded_sigma_2) / 2

        # Inverse of average covariance
        inv_avg_sigma = torch.linalg.inv(avg_sigma)

        # Term 1: Mahalanobis distance
        term1 = 0.125 * torch.torch.einsum('ijk,ijkl,ijl->ij', mean_diff, inv_avg_sigma, mean_diff)

        # Term 2: Log determinant term
        det_expanded_sigma_1 = torch.linalg.det(expanded_sigma_1)**(1/self.parent.feature_dim)
        det_expanded_sigma_2 = torch.linalg.det(expanded_sigma_2)**(1/self.parent.feature_dim)
        det_avg_sigma = torch.linalg.det(avg_sigma)**(1/self.parent.feature_dim)
        term2 = 0.5 * torch.log((det_avg_sigma / torch.sqrt(det_expanded_sigma_1 * det_expanded_sigma_2)))

        return term1 + term2
    
import torch
import numpy as np
import torch.nn.functional as F

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent


    '''     
    def update_score(self, label):
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Step 1: Find the closest correct cluster
        correct_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 1)
        if not correct_clusters_mask.any():
            # Handle the case where no correct clusters are found
            return

        correct_activations = normalized_gamma * correct_clusters_mask
        best_correct_index = torch.argmax(correct_activations)

        # Step 2: Update the score of the closest correct cluster
        self.update_cluster_score(best_correct_index, label, correct = self.parent.cluster_labels[best_correct_index][label] == 1)

        # Step 3: Identify incorrect clusters with higher activation than the best correct cluster
        incorrect_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 0)
        incorrect_clusters_higher_activation = incorrect_clusters_mask & (normalized_gamma > normalized_gamma[best_correct_index])
        incorrect_higher_indices = torch.where(incorrect_clusters_higher_activation)[0]

        # Step 4: Update the scores of identified incorrect clusters
        for index in incorrect_higher_indices:
            self.update_cluster_score(index, label, correct=False)

        # Step 5: Identify the first incorrect cluster with lower activation than the best correct cluster
        incorrect_clusters_lower_activation = incorrect_clusters_mask & (normalized_gamma <= normalized_gamma[best_correct_index])
        if incorrect_clusters_lower_activation.any():
            first_incorrect_lower_index = torch.where(incorrect_clusters_lower_activation)[0][0]  # Get the first such index
            self.update_cluster_score(first_incorrect_lower_index, label, correct=True)
    '''

    def update_score(self, label):
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Masks for correct and incorrect clusters
        correct_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 1)
        incorrect_clusters_mask = (self.parent.cluster_labels[:self.parent.c, label] == 0)

        # Sorted indices by activation
        sorted_indices = torch.argsort(normalized_gamma, descending=True)

        # Find the first cluster that should be updated
        first_update_index = None
        update_correct = True  # Flag to indicate if we are updating correct or incorrect clusters

        for index in sorted_indices:
            if correct_clusters_mask[index]:
                first_update_index = index
                update_correct = True
                break
            elif incorrect_clusters_mask[index]:
                first_update_index = index
                update_correct = False
                break

        if first_update_index is not None:
            # Determine the range of indices to update
            # +1 because we include the first cluster of the opposite type
            update_indices = sorted_indices[:sorted_indices.tolist().index(first_update_index) + 1]

            # Update the scores
            self.update_cluster_scores(update_indices, label, correct=update_correct)


    def update_cluster_scores(self, cluster_indices, label, correct=True):
        score_increment = 1.0 if correct else 0.0
        score_increment_tensor = torch.tensor(score_increment, dtype=torch.float32)

        # Update num_pred for all clusters
        self.parent.num_pred[cluster_indices] += 1

        # Calculate new scores
        prev_scores = self.parent.score[cluster_indices]
        num_preds = self.parent.num_pred[cluster_indices]
        new_scores = ((num_preds - 1) * prev_scores + score_increment_tensor) / num_preds

        self.parent.score[cluster_indices] = new_scores

    '''
    def update_score(self, label):
        # Normalize Gamma (this is used as the weight)
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Identify the winning cluster for this sample (the one with the highest normalized_gamma)
        j = torch.argmax(normalized_gamma)
        # The weight for this prediction is the normalized gamma value of the winning cluster
        weight = normalized_gamma[j]

        # Check if the winning cluster's prediction matches the true class (label)
        correct = self.parent.cluster_labels[j][label] == 1

        # Update the sum of weights for the winning cluster
        #old_weight_sum = self.parent.num_pred[j]
        self.parent.num_pred[j] += 1

        # Compute the new score (weighted accuracy)
        if self.parent.num_pred[j] == 1:
            # First prediction for this cluster, the score is the weight if correct, and 0 if incorrect
            new_score = torch.tensor(1) if correct else torch.tensor(0.0)
        else:
            # Update the score based on the accumulated weighted accuracy formula
            weighted_correct = torch.tensor(1) if correct else torch.tensor(0.0)
            new_score = ((self.parent.num_pred[j] -1) * self.parent.score[j] + weighted_correct) / self.parent.num_pred[j]

        # Check for NaN in the new score
        if torch.isnan(new_score):
            print("Warning: Computed score is NaN. Setting score to 0.")
            new_score = torch.tensor(0.0)

        self.parent.score[j] = new_score

    '''
    '''
    def update_score(self, label):
        # Normalize Gamma (used as the weights for each cluster's prediction)
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Check if each cluster's prediction matches the true class (label)
        correct_predictions = self.parent.cluster_labels[:self.parent.c, label] == 1

        # Convert correct_predictions to float for calculations
        correct_predictions = correct_predictions.float()

        # Update the sum of weights for each cluster
        old_weight_sum = self.parent.num_pred[:self.parent.c].clone()
        self.parent.num_pred[:self.parent.c] += normalized_gamma

        # Compute the new scores (weighted accuracy) for all clusters
        weighted_correct = normalized_gamma * correct_predictions
        new_scores = torch.where(
            old_weight_sum == 0,
            weighted_correct,
            (old_weight_sum * self.parent.score[:self.parent.c] + weighted_correct) / self.parent.num_pred[:self.parent.c]
        )

        # Check for NaN in the new scores and set them to 0
        nan_mask = torch.isnan(new_scores)
        if nan_mask.any():
            print("Warning: Computed scores contain NaN. Setting those scores to 0.")
            new_scores[nan_mask] = 0.0

        self.parent.score[:self.parent.c] = new_scores
    '''
    
    
    '''
    def update_score(self, label):
    
        # Normalize Gamma
        normalized_gamma = self.parent.consequence.compute_normalized_gamma() # self.parent.Gamma[:self.parent.c] #
        
        # Assuming target_label is defined (the label you're comparing against)
        matching_clusters = self.parent.cluster_labels[:self.parent.c][:, label] == 1
        label_adjustment = torch.where(matching_clusters, 1, 0)

        number_of_samples = self.parent.n_glo[label]
        
        # Apply the label adjustment to normalized_gamma
        self.parent.score[0:self.parent.c] = (self.parent.score[0:self.parent.c] + normalized_gamma*label_adjustment/number_of_samples)/sum( self.parent.n_glo)
    '''

    '''
    def update_score(self, label):
        # Normalize Gamma
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Ensure there are at least two clusters
        if normalized_gamma.numel() < 2:
            return

        top_two_indices = torch.topk(normalized_gamma, 2).indices
        best_cluster_idx = top_two_indices[0]
        second_best_cluster_idx = top_two_indices[1]

        # Check if the best and second-best clusters' predictions match the true class
        best_correct = self.parent.cluster_labels[best_cluster_idx][label] == 1
        second_best_correct = self.parent.cluster_labels[second_best_cluster_idx][label] == 1

        # Set 'correct' to False if the second-best cluster is not of the opposite label
        if best_correct == second_best_correct:
            correct = False
        else:
            correct = best_correct

        # Update the score for the best cluster
        self.update_individual_score(best_cluster_idx, correct)

        # Update the score for the second-best cluster
        # Increase if they are different, decrease if they are the same
        self.update_individual_score(second_best_cluster_idx, not correct)

    def update_individual_score(self, cluster_idx, correct):
        # Increment the number of predictions for the cluster
        self.parent.num_pred[cluster_idx] += 1

        # Compute the new score
        if self.parent.num_pred[cluster_idx] == 1:
            new_score = torch.tensor(float(correct))
        else:
            N = self.parent.num_pred[cluster_idx]
            new_score = ((N - 1) * self.parent.score[cluster_idx] + float(correct)) / N

        # Check for NaN in the new score and update
        if torch.isnan(new_score):
            print(f"Warning: Computed score is NaN for cluster {cluster_idx}. Setting score to 0.")
            self.parent.score[cluster_idx] = torch.tensor(0.0)
        else:
            self.parent.score[cluster_idx] = new_score
        '''

    '''
    def update_score(self, label):
        # Normalize Gamma
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Identify the winning cluster for this sample (the one with the highest normalized_gamma)
        j = torch.argmax(normalized_gamma)

        # Check if the winning cluster's prediction matches the true class (label)
        correct = self.parent.cluster_labels[j][label] == 1
        
        # Update the error rate for the winning cluster

        N = torch.sum(self.parent.n_glo) # number of samples
        n = self.parent.n_glo[label] # number of class samples
        if (N != n):
            T = (self.parent.score[j] *self.parent.num_pred[j] * n / (N - n)) / (1 - self.parent.score[j] + self.parent.score[j] * n / (N - n))

            self.parent.num_pred[j] = self.parent.num_pred[j]+1# number of classifications

            # Calculate new score
            if correct:
                new_score = ((T + 1) / (n + 1)) / ((T + 1) / (n + 1) + (self.parent.num_pred[j] - T) / (N - n))
            else:
                new_score = (T / n) / (T / n + (self.parent.num_pred[j] - T) / (N - n))
            
            # Check for NaN in the new score
            if torch.isnan(new_score):
                print("Warning: Computed score is NaN. Setting score to 0.")
                self.parent.score[j] = torch.tensor(0.0)
            else:
                self.parent.score[j] = new_score
    '''
    ''' 
    def update_score(self, label):

        # Compute normalized gamma values
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Find the index of the cluster with the maximal gamma value
        max_gamma_index = torch.argmax(normalized_gamma)

        # Check if the label matches for the cluster with the maximal gamma
        is_label_match = self.parent.cluster_labels[max_gamma_index, label] == 1

        # Calculate the adjustment based on the label correctness
        label_adjustment = 1 if is_label_match else -1

        # Calculate the number of samples for the given label
        number_of_samples = torch.sum(self.parent.n_glo[label])

        # Update the score only for the cluster with the maximal gamma
        self.parent.score[max_gamma_index] = self.parent.score[max_gamma_index] + label_adjustment / number_of_samples
    ''' 
    '''
    def update_score(self, label):
        """
        Update the running log loss (score) for each cluster in a vectorized manner.

        Args:
        true_label (int): The true label of the sample.
        """
        # Compute normalized gamma values (predicted probabilities for each cluster)
        normalized_gamma = self.parent.consequence.compute_normalized_gamma()

        # Get the one-hot encoded class labels for each cluster
        one_hot_labels = normalized_gamma.unsqueeze(-1) *self.parent.cluster_labels[:self.parent.c]

        # Compute the predicted probability for the true label
        predicted_probabilities = one_hot_labels

        # Create a tensor of the true label in float and match the shape
        true_label_tensor = torch.zeros_like(predicted_probabilities)
        true_label_tensor[:,label] = 1

        # Compute log loss for the predicted probabilities
        current_log_losses = F.binary_cross_entropy(predicted_probabilities, 
                                                    true_label_tensor, 
                                                    reduction='none')

        # Normalize the log loss by the number of samples for the true label and update the score
        self.parent.score[0:self.parent.c] += torch.sum(current_log_losses /  self.parent.n_glo[label], dim = 1)
        '''
        
    def remove_small(self):
        if self.parent.c < 2:
            return

        V = torch.exp(torch.linalg.slogdet(self.parent.S[self.parent.matching_clusters])[1]) # [1] is the log determinant
        V_S_0 = torch.prod(torch.diag(self.parent.S_0))
        V_ratio = (V / V_S_0)**(1/self.parent.feature_dim)

        clusters_to_remove = self.parent.matching_clusters[V_ratio < 1/(10*self.parent.N_r)]
        #num_clusters_to_remove = len(self.parent.matching_clusters) - self.parent.c_max

        #if num_clusters_to_remove > 0:
            # Remove the clusters
        with torch.no_grad():
            for cluster_id in clusters_to_remove:
                self.remove_cluster(cluster_id)

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
                score_row = self.parent.score[self.parent.merging_mech.valid_clusters[row]].item()*self.parent.num_pred[self.parent.merging_mech.valid_clusters[row]].item()
                score_col = self.parent.score[self.parent.merging_mech.valid_clusters[col]].item()*self.parent.num_pred[self.parent.merging_mech.valid_clusters[col]].item()

                # Determine which cluster of the pair to remove based on the smallest score
                cluster_to_remove_idx = row if score_row < score_col else col
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


    def remove_score(self):
        if self.parent.c < 2:
            return

        # Determine how many clusters to remove to meet the desired count
        num_clusters_to_remove = len(self.parent.matching_clusters) - self.parent.c_max

        if num_clusters_to_remove > 0:
            # Create a composite score for each cluster, lower score is better
            # In case of a tie in score, lower num_pred is better
            composite_scores = [(self.parent.score[i], -self.parent.num_pred[i]) for i in self.parent.matching_clusters]

            # Sort the clusters by composite score
            sorted_indices = sorted(range(len(composite_scores)), key=lambda i: composite_scores[i], reverse=True)

            # Get the indices of clusters to remove in descending order
            indices_to_remove = sorted(sorted_indices[:num_clusters_to_remove], reverse=True)

            with torch.no_grad():
                # Remove the selected clusters
                for index in indices_to_remove:
                    self.remove_cluster(self.parent.matching_clusters[index])

    def removal_mechanism(self):
        ''' Remove clusters with negative scores and additional low scoring clusters if necessary. '''

        '''
        # Identify and sort indices of clusters with negative scores
        negative_score_indices = torch.where(self.parent.score[self.parent.matching_clusters] < 0)[0]
        negative_score_indices = negative_score_indices.sort(descending=True)[0]

        # Remove clusters with negative scores
        with torch.no_grad():
            for index in negative_score_indices:
                self.remove_cluster(self.parent.matching_clusters[index])
        '''
        # Determine how many clusters to remove to meet the desired count

        #if len(self.parent.matching_clusters) > self.parent.c_max:
            #self.remove_small()
            
            # Print the number of samples after removal
            #print("Number of samples after remove_small:", len(self.parent.matching_clusters))
            
            # Labels consistency check
            #labels_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) < 2
            #if not labels_check:
            #    print("Critical error: Labels consistency in matching clusters after remove_small:", labels_check)

        #self.remove_overlapping()

            # Print the number of samples after removal
            #print("Number of samples after remove_overlapping:", len(self.parent.matching_clusters))
            
            # Labels consistency check
           # labels_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) < 2
            #if not labels_check:
            #    print("Critical error: Labels consistency in matching clusters after remove_overlapping:", labels_check)
            
        self.remove_score()

            # Print the number of samples after removal
            #print("Number of samples after remove_score:", len(self.parent.matching_clusters))

            # Labels consistency check
            #labels_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) < 2
            #if not labels_check:
            #    print("Critical error: Labels consistency in matching clusters after remove_score:", labels_check)


    '''      
    def removal_mechanism(self):
    #Compute the initial merging candidates
        if len(self.parent.matching_clusters) < self.parent.c_max:
            return
        
        # Continue removing the smallest clusters while the condition is not met
        while len(self.parent.matching_clusters) > self.parent.c_max:
            
            self.parent.merging_mech.valid_candidate = torch.arange(self.parent.c, dtype=torch.int32, device=self.parent.device)

            self.parent.merging_mech.compute_merging_condition()

            kappa = self.parent.merging_mech.kappa[self.parent.merging_mech.kappa==self.parent.merging_mech.kappa]
            
            i_smallest_n = torch.argmin(self.parent.n[kappa < self.parent.kappa_join])
            
            # Remove the smallest overlapping luster
            with torch.no_grad():
                self.remove_cluster(self.parent.matching_clusters[i_smallest_n])
    '''
    
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
        labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) < 2
        if not labels_consistency_check:
            print("Critical error: Labels consistency in matching clusters after removal:", labels_consistency_check)

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
    
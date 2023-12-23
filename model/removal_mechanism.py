import torch
import numpy as np
import torch.nn.functional as F

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent
      
    def update_score(self, label):
        
            #det_matrix = torch.exp(torch.linalg.slogdet(self.parent.S[0:self.parent.c])[1]) # [1] is the log determinant
            #V = torch.sqrt(det_matrix)
        
            # Normalize Gamma
            normalized_gamma = self.parent.consequence.compute_normalized_gamma() # self.parent.Gamma[:self.parent.c] #
            
            # Assuming target_label is defined (the label you're comparing against)
            # Adjust normalized_gamma based on label correctness
            # Create a tensor that is 1 where the label is correct and -1 where it is not
            matching_clusters = self.parent.cluster_labels[:self.parent.c][:, label] == 1
            label_adjustment = torch.where(matching_clusters, 0, -1)
    
            #n = self.parent.n[0:self.parent.c]
            #number_of_samples = sum(n[matching_clusters])

            number_of_samples = torch.sum(self.parent.n_glo[label])
            
            # Apply the label adjustment to normalized_gamma
            self.parent.score[0:self.parent.c] = self.parent.score[0:self.parent.c] + normalized_gamma*label_adjustment/number_of_samples#
    
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
    

    def removal_mechanism(self):
        ''' Remove smallest clusters until the number of clusters is less than 10 times the square root of the feature dimension. '''

        # Compute volume self.parent.merging_mech.V
        #det_matrix = torch.exp(torch.linalg.slogdet(self.parent.S[self.parent.matching_clusters])[1]) # [1] is the log determinant

        # Vectorized computation of volume V for upper triangle
        #V = torch.sqrt(det_matrix)
        #V_S_0 = torch.sqrt(torch.prod(torch.diag(self.parent.S_0)))
        #V_ratio = V  / (V_S_0 + 1e-30)

        # Continue removing the smallest clusters while the condition is not met
        while (len(self.parent.matching_clusters) > self.parent.c_max):

            # Here you should implement your new logic for determining which cluster to remove
            # For instance, removing based on another metric
            index_to_remove = torch.argmin(self.parent.score[self.parent.matching_clusters]) # This is a placeholder for your actual removal logic
            #index_to_remove = torch.argmin(self.parent.score[self.parent.matching_clusters]*self.parent.n[self.parent.matching_clusters]) # This is a placeholder for your actual removal logic
            # Update V_ratio by removing the index_to_remove element
            
            #V = torch.cat((V[:index_to_remove], V[index_to_remove + 1:]))
            
            #self.parent.matching_clusters = torch.cat((self.parent.matching_clusters[:index_to_remove], self.parent.matching_clusters[index_to_remove + 1:]))
            
            # Remove the cluster
            with torch.no_grad():
                self.remove_cluster(self.parent.matching_clusters[index_to_remove])
            
               #self.parent.n[self.parent.matching_clusters]*self.parent.score[self.parent.matching_clusters])              
                #highest_error = torch.argmin(self.parent.score[self.parent.matching_clusters])
                #self.remove_cluster(self.parent.matching_clusters[highest_error])
    
    
    def federated_removal_mechanism(self):
        ''' Remove smallest clusters until the number of clusters is less than 10 times the square root of the feature dimension. '''

        # Continue removing the smallest clusters while the condition is not met
        while (len(self.parent.matching_clusters) > self.parent.c_max):

            # Here you should implement your new logic for determining which cluster to remove
            # For instance, removing based on another metric
            index_to_remove = torch.argmin(self.parent.n[self.parent.matching_clusters]) # This is a placeholder for your actual removal logic
            #index_to_remove = torch.argmin(self.parent.score[self.parent.matching_clusters]*self.parent.n[self.parent.matching_clusters]) # This is a placeholder for your actual removal logic
            # Update V_ratio by removing the index_to_remove element
            
            #V = torch.cat((V[:index_to_remove], V[index_to_remove + 1:]))
            
            #self.parent.matching_clusters = torch.cat((self.parent.matching_clusters[:index_to_remove], self.parent.matching_clusters[index_to_remove + 1:]))
            
            # Remove the cluster
            with torch.no_grad():
                self.remove_cluster(self.parent.matching_clusters[index_to_remove])

    '''      
    def removal_mechanism(self):
    #Compute the initial merging candidates
        if len(self.parent.matching_clusters) < self.parent.c_max:
            return
        
        # Continue removing the smallest clusters while the condition is not met
        while len(self.parent.matching_clusters) > self.parent.c_max:
            
            self.parent.merging_mech.valid_candidate = torch.arange(self.parent.c, dtype=torch.int64, device=self.parent.device)

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
            self.parent.score[cluster_index] = self.parent.score[last_active_index]
                    
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
        labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) == 1
        if not labels_consistency_check:
            print("Critical error: Labels consistency in matching clusters after removal:", labels_consistency_check)

        # Debugging checks
        if self.parent.enable_debugging:

            # Check if cluster_index is in matching_clusters and points to the same data as the old last_active_index
            label_match_check = True
            if cluster_index not in self.parent.matching_clusters:
                # Check if the label of cluster_index is among the labels of clusters in matching_clusters
                label_match_check = self.parent.cluster_labels[cluster_index] in self.parent.cluster_labels[self.parent.matching_clusters]
                print("Label of cluster index is in labels of matching clusters:", label_match_check)
            

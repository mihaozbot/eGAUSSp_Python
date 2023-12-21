import torch
import numpy as np

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent
    
    def update_score(self, label):
        
            # Normalize Gamma
            normalized_gamma = self.parent.Gamma[0:self.parent.c]/ self.parent.Gamma[0:self.parent.c].sum()

            # Assuming target_label is defined (the label you're comparing against)
            # Adjust normalized_gamma based on label correctness
            # Create a tensor that is 1 where the label is correct and -1 where it is not
            matching_clusters = self.parent.cluster_labels[:self.parent.c][:, label] == 1
            label_adjustment = torch.where(matching_clusters,0, -1)
    
            n = self.parent.n[0:self.parent.c]
            number_of_samples = sum(n[matching_clusters])

            # Apply the label adjustment to normalized_gamma
            self.parent.score[0:self.parent.c] = self.parent.score[0:self.parent.c] + normalized_gamma * label_adjustment/number_of_samples
          
    def removal_mechanism(self):

        ''' Remove smallest clusters until the number of clusters is less than 10 times the square root of the feature dimension. '''
        #if len(self.parent.matching_clusters) < self.parent.c_max:
        #    return

        # Continue removing the smallest clusters while the condition is not met
        while (len(self.parent.matching_clusters) > self.parent.c_max):
            
            # Identify the smallest cluster
            # Assuming 'n' holds the size of each cluster, find the index of the smallest cluster
            smallest_cluster_index = torch.argmin(self.parent.score[self.parent.matching_clusters])

 
            # Remove the smallest cluster
            with torch.no_grad():
                self.remove_cluster(self.parent.matching_clusters[smallest_cluster_index])
                            
                #highest_error = torch.argmin(self.parent.score[self.parent.matching_clusters])
                #self.remove_cluster(self.parent.matching_clusters[highest_error])
            
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
            if self.parent.matching_clusters[-1] == last_active_index:
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
            

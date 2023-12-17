import torch
import numpy as np

class RemovalMechanism:
    def __init__(self, parent):
        self.parent = parent

    def removal_mechanism(self):

        ''' Remove smallest clusters until the number of clusters is less than 10 times the square root of the feature dimension. '''
        if len(self.parent.matching_clusters) < 10*np.sqrt(self.parent.feature_dim):
            return
        
        # Continue removing the smallest clusters while the condition is not met
        while len(self.parent.matching_clusters) >= 10*np.sqrt(self.parent.feature_dim):
            # Identify the smallest cluster
            # Assuming 'n' holds the size of each cluster, find the index of the smallest cluster
            smallest_cluster_index = torch.argmin(self.parent.n[self.parent.matching_clusters])

            # Remove the smallest cluster
            with torch.no_grad():
                self.remove_cluster(self.parent.matching_clusters[smallest_cluster_index])

    
    def remove_cluster(self, cluster_index):
        '''Remove a specified cluster by replacing it with the last active cluster and updating relevant parameters. '''
        #Copy everhing from the last active cluster to the remove cluster index
        #Then if the last active index was on the matching clusters list it needs to be removed, 
        # else the index of the memoved cluster needs to be removed
        
        last_active_index = self.parent.c - 1

        #Instead of removing just copy last cluster in the place of the removed on
        if cluster_index != last_active_index: #The target cluster is not the last cluster
        
            # Move the last active cluster to the position of the cluster to be removed
            self.parent.mu[cluster_index] = self.parent.mu[last_active_index]
            self.parent.S[cluster_index] = self.parent.S[last_active_index]
            self.parent.n[cluster_index] = self.parent.n[last_active_index]
            
            # Update the label of the cluster that is moved
            self.parent.cluster_labels[cluster_index] = self.parent.cluster_labels[last_active_index]

            # Update the Gamma value for the cluster 
            self.parent.Gamma[cluster_index] = self.parent.Gamma[last_active_index]

            # Update matching_clusters list
            if last_active_index == self.parent.matching_clusters[-1]: #if the last cluster is on the list, it is the last elements as the list is ordered
                # The last cluster was moved to the where the jth index is pointing so just remove the last cluster from the list
                # Remove the last index if last_active_index is in matching_clusters
                self.parent.matching_clusters = self.parent.matching_clusters[:-1]
            else: #The last cluster is not a matching cluster so we can remove the cluster index
                # Else, remove the cluster_index from matching_clusters
                self.parent.matching_clusters = self.parent.matching_clusters[self.parent.matching_clusters  != cluster_index]
                
        #If the cluster was the last cluster just reduce the number of clusters 
        # Decrement the count of active clusters
        self.parent.c -= 1

        # Debugging checks
        if self.parent.enable_debugging:

            # Check if cluster_index is in matching_clusters and points to the same data as the old last_active_index
            label_match_check = True
            if cluster_index not in self.parent.matching_clusters:
                # Check if the label of cluster_index is among the labels of clusters in matching_clusters
                label_match_check = self.parent.cluster_labels[cluster_index] in self.parent.cluster_labels[self.parent.matching_clusters]
                print("Label of cluster index is in labels of matching clusters:", label_match_check)
            
            # Check if all elements in self.parent.cluster_labels[self.parent.matching_clusters] are the same
            labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) == 1
            if not labels_consistency_check:
                print("Labels consistency in matching clusters:", labels_consistency_check)
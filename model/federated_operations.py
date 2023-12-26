import torch
import numpy as np
import torch.nn as nn

    
class FederalOps:
    def __init__(self, parent):
        ''' Initialize the FederalOps with a reference to the parent class. '''
        self.parent = parent
        
    def federated_merging(self):
        ''' Perform federated merging of clusters based on labels within a specified number of iterations. '''
        
        # Iterate over unique labels in the cluster labels
        for label in range(0, self.parent.num_classes):
        
            # Identify clusters with the current label
            # In training mode, match clusters based on the label
            self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c][:, label])[0]
            random_indices = torch.randperm(self.parent.matching_clusters.size(0))
            centers = self.parent.mu[self.parent.matching_clusters[random_indices]].detach().clone()
            
            if len(centers) == 0:
                continue

            for i, center in enumerate(centers):

                #self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c][:, label])[0]
                if self.parent.matching_clusters[-1] == self.parent.c:
                    print(f"matching_clusters is not matching! There must be some error in the code logic.")

            # Check if all elements in self.parent.cluster_labels[self.parent.matching_clusters] are the same
                labels_consistency_check = len(torch.unique(self.parent.cluster_labels[self.parent.matching_clusters], dim=0)) == 1
                if not labels_consistency_check:
                    print("Critical error: Labels consistency in matching clusters in federated merging:", labels_consistency_check)

                #Compute distance to the current cluster center
                self.parent.Gamma = self.parent.mathematician.compute_activation(center)  

                #Use the merging mechanism 
                self.parent.merging_mech.merging_mechanism()

                '''
                self.parent.merging_mech.valid_clusters = self.parent.matching_clusters[(self.parent.n[self.parent.matching_clusters] >= self.parent.kappa_n)]

                # Continue merging while the flag is True and iterations are below the maximum
                merge = True  # Flag to control the merging process
                iteration = 0 # Counter to track the number of iterations
                while merge and iteration < max_iterations:
                
                    if len(self.parent.merging_mech.valid_clusters) < 2:
                        break

                    #Compute the initial merging candidates
                    self.parent.merging_mech.compute_merging_condition()
                
                    #Check merging condition, merge rules, and return True if merge happened
                    merge = self.parent.merging_mech.merge_clusters()
                    iteration += 1  # Increment the iteration counter
                '''

            #Remove small clusters 
            if self.parent.c>1:
                self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c][:, label])[0]
                self.parent.merging_mech.valid_clusters = self.parent.matching_clusters
                self.parent.removal_mech.removal_mechanism()

                self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c][:, label])[0]
                self.parent.merging_mech.valid_clusters = self.parent.matching_clusters
                self.parent.removal_mech.federated_removal_mechanism()

    def merge_model_statistics(self, model):
        ''' Merge the global statistical parameters of another model into the current federated model. '''

        # Calculate total samples in both models
        n_fed = torch.sum(self.parent.n_glo)
        n_local = torch.sum(model.n_glo)

        # Merge global parameters
        self.parent.S_glo = (self.parent.S_glo * n_fed + model.S_glo * n_local) / (n_fed + n_local) + \
                            (n_fed * n_local / (n_fed + n_local)) * (self.parent.mu_glo - model.mu_glo) ** 2
        self.parent.mu_glo = (self.parent.mu_glo * n_fed + model.mu_glo * n_local) / (n_fed + n_local)
        self.parent.var_glo = self.parent.S_glo / (n_fed + n_local)

        # Merge class counts
        self.parent.n_glo += model.n_glo

        # Update minimum cluster size
        self.parent.clusterer.update_S_0()

    def merge_model_privately(self, model, n_min):
        ''' Merge the parameters of another model into the current federated model. '''

        # Filter out clusters where model.n > 0
        valid_clusters = (model.n[:model.c] > n_min) #*(model.score[:model.c] > 0)
        num_valid_clusters = valid_clusters.sum()

        # First, merge the global statistical parameters
        self.merge_model_statistics(model)
        
        # Ensure the federated model has enough capacity for the new clusters
        before_size = self.parent.c  # Current size of the model
        after_size = self.parent.c + num_valid_clusters # Size after merging
        self.parent.overseer.ensure_capacity(after_size)  # Ensure there's enough space

        # Merge the parameters of the models
        with torch.no_grad():  # Temporarily disable gradient tracking
            # Identify the indices of valid clusters
            valid_indices = torch.where(valid_clusters)[0]

            # Calculate the new indices in the parent model for these valid clusters
            new_indices = torch.arange(before_size, before_size + len(valid_indices), device=model.mu.device)

            # Perform the parameter copying using advanced indexing
            self.parent.mu.data[new_indices] = model.mu.data[valid_indices]
            self.parent.S.data[new_indices] = model.S.data[valid_indices]
            self.parent.n.data[new_indices] = model.n.data[valid_indices]

            # Update cluster labels
            self.parent.cluster_labels[new_indices] = model.cluster_labels[valid_indices]

            # Update scores
            self.parent.score[new_indices] = model.score[valid_indices]
                    
        # Reset the Gamma values for all clusters
        self.parent.Gamma = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=True)
                
        # Update the total count of clusters in the federated model
        self.parent.c = after_size  # Update to the new index, which reflects the actual number of clusters after merging

    
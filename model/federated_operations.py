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
            self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c][:, label])[0]
            self.parent.removal_mech.removal_mechanism()

    def merge_model_statistics(self, model):
        ''' Merge the global statistical parameters of another model into the current federated model. '''

        # Calculate total samples in both models
        n_fed = torch.sum(self.parent.n_glo)
        n_local = torch.sum(model.n_glo)

        # Merge global parameters
        self.parent.S_glo = (self.parent.S_glo * n_fed + model.S_glo * n_local) / (n_fed + n_local) + \
                            (n_fed * n_local / (n_fed + n_local)) * (self.parent.mu_glo - model.mu_glo) ** 2
        self.parent.mu_glo = (self.parent.mu_glo * n_fed + model.mu_glo * n_local) / (n_fed + n_local)
        self.parent.s_glo = torch.sqrt(self.parent.S_glo / (n_fed + n_local))

        # Merge class counts
        self.parent.n_glo += model.n_glo

        # Update minimum cluster size
        self.parent.clusterer.update_S_0()

    def merge_model(self, model, num_clients):
        ''' Merge the parameters of another model into the current federated model. '''

        # First, merge the global statistical parameters
        self.merge_model_statistics(model)
        
        # Ensure the federated model has enough capacity for the new clusters
        before_size = self.parent.c  # Current size of the model
        after_size = self.parent.c + model.c  # Size after merging
        self.parent.overseer.ensure_capacity(after_size + 1)  # Ensure there's enough space

        # Merge the parameters of the models
        with torch.no_grad():  # Temporarily disable gradient tracking
            self.parent.mu.data[before_size:after_size] = model.mu.data[:model.c]
            self.parent.S.data[before_size:after_size] = model.S.data[:model.c]
            self.parent.n.data[before_size:after_size] = model.n.data[:model.c]

            # Update the cluster labels and the label-to-cluster mapping
            for i in range(model.c):
                cluster_label = model.cluster_labels[i]
                self.parent.cluster_labels[before_size + i] = cluster_label

            '''
                # Update or create the label_to_clusters entry for the new label
                if cluster_label not in self.parent.label_to_clusters:
                    self.parent.label_to_clusters[cluster_label] = torch.empty(0, dtype=torch.int32, device=self.parent.device)

                self.parent.label_to_clusters[cluster_label] = torch.cat(
                    (self.parent.label_to_clusters[cluster_label], torch.tensor([before_size + i], dtype=torch.int32, device=self.parent.device))
                )
            '''      
            # Reset the Gamma values for all clusters
            self.parent.Gamma = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=True)
                
        # Update the total count of clusters in the federated model
        self.parent.c = after_size

    def merge_model_privately(self, model, c_min):
        ''' Merge the parameters of another model into the current federated model. '''

        # Filter out clusters where model.n > 0
        valid_clusters = (model.n[:model.c] > c_min) #*(model.score[:model.c] > 0)
        num_valid_clusters = valid_clusters.sum()

        # First, merge the global statistical parameters
        self.merge_model_statistics(model)
        
        # Ensure the federated model has enough capacity for the new clusters
        before_size = self.parent.c  # Current size of the model
        after_size = self.parent.c + num_valid_clusters  # Size after merging
        self.parent.overseer.ensure_capacity(after_size)  # Ensure there's enough space

        # Initialize a counter for the new index in self.parent
        new_index = before_size

        # Merge the parameters of the models
        with torch.no_grad():  # Temporarily disable gradient tracking
            for i in range(model.c):
                if valid_clusters[i]:
                    self.parent.mu.data[new_index] = model.mu.data[i]
                    self.parent.S.data[new_index] = model.S.data[i] #/num_clients
                    self.parent.n.data[new_index] = model.n.data[i] #/num_clients
                        
                    # Update cluster labels and the label-to-cluster mapping
                    cluster_label = model.cluster_labels[i]
                    self.parent.cluster_labels[new_index] = cluster_label

                    self.parent.score[new_index] = model.score[i]
                    
                    '''
                    # Update or create the label_to_clusters entry for the new label
                    if cluster_label not in self.parent.label_to_clusters:
                        self.parent.label_to_clusters[cluster_label] = torch.empty(0, dtype=torch.int32, device=self.parent.device)

                    self.parent.label_to_clusters[cluster_label] = torch.cat(
                        (self.parent.label_to_clusters[cluster_label], torch.tensor([new_index], dtype=torch.int32, device=self.parent.device))
                    )
                    '''
                    
                    # Increment the new index
                    new_index += 1
                    
        # Reset the Gamma values for all clusters
        self.parent.Gamma = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=True)
                
        # Update the total count of clusters in the federated model
        self.parent.c = new_index  # Update to the new index, which reflects the actual number of clusters after merging

    
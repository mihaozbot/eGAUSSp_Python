import torch

# Attempt to load the line_profiler extension
try:
    from line_profiler import LineProfiler
    profile = LineProfiler()  # If line_profiler is available, use it
except ImportError:
    # If line_profiler is not available, define a dummy profile decorator
    def profile(func): 
        return func
    
class FederalOps:
    def __init__(self, parent):
        ''' Initialize the FederalOps with a reference to the parent class. '''
        self.parent = parent

    def federated_merging(self, max_iterations=100):
        ''' Perform federated merging of clusters based on labels within a specified number of iterations. '''
        
        # Iterate over unique labels in the cluster labels
        for label in torch.unique(self.parent.cluster_labels[:self.parent.c]):
                
            iteration = 0 # Counter to track the number of iterations
            merge = True  # Flag to control the merging process

            # Continue merging while the flag is True and iterations are below the maximum
            while merge and iteration < max_iterations:
                
                # Identify clusters with the current label
                self.parent.matching_clusters = torch.where(self.parent.cluster_labels[:self.parent.c] == label)[0]
                self.parent.merging_mech.valid_clusters = self.parent.matching_clusters
                
                # Check if there are enough clusters to merge
                if len(self.parent.merging_mech.valid_clusters) < 2:
                    merge = False  # Merge cannot happen with less than two clusters
                
                with torch.no_grad():
                    # Check merging conditions and merge clusters if applicable
                    merge = self.parent.merging_mech.compute_cluster_parameters()
                
                iteration += 1  # Increment the iteration counter

    def merge_model_statistics(self, model):
        ''' Merge the global statistical parameters of another model into the current federated model. '''

        # Merge global parameters like covariance matrix, mean, count, and standard deviation
        self.parent.S_glo = (self.parent.S_glo + model.S_glo) + (self.parent.n_glo * model.n_glo / (self.parent.n_glo + model.n_glo)) * (self.parent.mu_glo - model.mu_glo)
        self.parent.mu_glo = (self.parent.n_glo * self.parent.mu_glo + model.n_glo * model.mu_glo) / (self.parent.n_glo + model.n_glo)
        self.parent.n_glo += model.n_glo
        self.parent.s_glo = torch.sqrt(self.parent.S_glo / self.parent.n_glo)

        # Update the minimum cluster size
        self.parent.clustering.update_S_0()
        
    def merge_model(self, model):
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

                # Update or create the label_to_clusters entry for the new label
                if cluster_label not in self.parent.label_to_clusters:
                    self.parent.label_to_clusters[cluster_label] = torch.empty(0, dtype=torch.int32, device=self.parent.device)

                self.parent.label_to_clusters[cluster_label] = torch.cat(
                    (self.parent.label_to_clusters[cluster_label], torch.tensor([before_size + i], dtype=torch.int32, device=self.parent.device))
                )
                
            # Reset the Gamma values for all clusters
            self.parent.Gamma = torch.zeros(after_size, dtype=torch.float32, device=self.parent.device, requires_grad=True)
                
        # Update the total count of clusters in the federated model
        self.parent.c = after_size

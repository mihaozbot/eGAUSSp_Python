import torch
import torch.nn as nn
import math
import numpy as np

from model.clustering_operations import ClusteringOps
from model.removal_mechanism import RemovalMechanism 
from model.merging_mechanism import MergingMechanism
from model.math_operations import MathOps
from model.consequence_operations import ConsequenceOps
from model.model_operations import ModelOps
from model.federated_operations import FederalOps

from utils.utils_train import test_model_in_batches
from collections import defaultdict

# Attempt to load the line_profiler extension

class eGAUSSp(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, kappa_n, num_sigma, kappa_join, S_0, N_r, c_max, device, num_samples = 100000):
        super(eGAUSSp, self).__init__()
        self.device = device
        self.feature_dim = feature_dim #Dimensionality of the features
        self.kappa_n = kappa_n #Minimal number of samples
        self.num_sigma = num_sigma/np.sqrt(self.feature_dim) #Activation distancethreshold*self.feature_dim**(1/np.sqrt(2)) 
        self.kappa_join = kappa_join #Merging threshold
        self.S_0 = S_0 * torch.eye(self.feature_dim, device=self.device) #Initialization covariance matrix
        self.S_0_initial = self.S_0.clone() #Initial covariance matrix
        self.N_r = N_r #Quantization number
        self.num_classes = num_classes #Max number of samples in clusters
        self.c_max = c_max #Max number of clusters

        #Forgetting factor for the number of samples of the clusters
        self.forgeting_factor = 1 - 1/num_samples

        # Dynamic properties initialized with tensors
        self.c = 0 # Number of active clusters
        self.Gamma = torch.empty(0, dtype=torch.float32, device=device,requires_grad=False)
        self.current_capacity = c_max #Initialize current capacity, which will be expanded as needed during training 
        self.cluster_labels = torch.empty((self.current_capacity, num_classes), dtype=torch.int32, device=device) #Initialize cluster labels
        #self.label_to_clusters = {} #Initialize dictionary to map labels to clusters
        
        self.score = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster labels
        self.num_pred = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize number of predictions

        self.one_hot_labels = torch.eye(num_classes, dtype=torch.int32) #One hot labels 
        
        # Trainable parameters
        self.n = nn.Parameter(torch.zeros(self.current_capacity, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster sizes
        self.mu = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster means
        self.S = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        self.S_inv = torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device)  # Initialize covariance matrices

        # Global statistics
        self.n_glo = torch.zeros((num_classes), dtype=torch.float32, device=device)  # Global number of sampels per class
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Global mean
        self.S_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Sum of squares for global variance

        # Initialize subclasses
        self.overseer = ModelOps(self)
        self.mathematician = MathOps(self)
        self.clusterer = ClusteringOps(self)
        self.merging_mech = MergingMechanism(self)
        self.removal_mech = RemovalMechanism(self)
        self.consequence = ConsequenceOps(self)
        self.federal_agent = FederalOps(self)
          
    def toggle_evolving(self, enable=None):
        ''' Function to toggle the evolving state of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_evolving(enable)

    def toggle_adding(self, enable=None):
        ''' Function to toggle the adding mechanism of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable.'''
        self.overseer.toggle_adding(enable)

    def toggle_merging(self, enable=None):
        ''' Function to toggle the merging mechanism of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_merging(enable)

    def toggle_debugging(self, enable=None):
        ''' Function to toggle the debugging state of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_debugging(enable)

    #def merge_model(self, client_model):  
    #    ''' Merges a client model into the main model. '''
    #    self.federal_agent.merge_model(client_model)
    
    def federated_merging(self):
        ''' Executes the merging mechanism for all rules. Conversely, the normal merging mechanism works on a subset of rules based on some conditions. '''
        
        self.federal_agent.federated_merging()

    def clustering(self, data, labels):
        sample_count = 0  # Initialize a counter for samples processed

        for (z, label) in zip(data, labels):
            sample_count += 1  # Increment the sample counter

            # Progress update every 10,000 samples
            if sample_count % 1000 == 0:
                print(f"Processed {sample_count} samples...")

            # Check if the model is in evaluation mode
            # In evaluation mode, match all clusters
                # Update global statistics
            self.clusterer.update_global_statistics(z, label)
            
            # In training mode, match clusters based on the label
            self.matching_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device)
            
            # Compute activation
            self.Gamma = self.mathematician.compute_activation(z)
            #self.Gamma *= self.score[:self.c]

            self.matching_clusters = torch.where(self.cluster_labels[:self.c][:, label] == 1)[0]
            
            # Evolving mechanisms
            if self.evolving:
                with torch.no_grad():
  
                    if self.c>0:
                        self.removal_mech.update_score(label)


                    #Incremental clustering and cluster addition
                    self.clusterer.increment_or_add_cluster(z, label)

                    # S_inv_ = torch.linalg.inv((self.S[:self.c]/
                    #             self.n[:self.c].view(-1, 1, 1))*
                    #             self.feature_dim)
                    # S_inv = self.S_inv[:self.c]
                    # if any(torch.sum(torch.sum(S_inv_-S_inv,dim=2), dim =1)>1e-4):
                    #     print("clustering?")
       
                    #Cluster merging
                    self.merging_mech.merging_mechanism()
    
                    # S_inv_ = torch.linalg.inv((self.S[:self.c]/
                    #             self.n[:self.c].view(-1, 1, 1))*
                    #             self.feature_dim)
                    # S_inv = self.S_inv[:self.c]
                    # if any(torch.sum(torch.sum(S_inv_-S_inv,dim=2), dim =1)>1e-4):
                    #     print("removal?")
                        #Removal mechanism
                    
                    if len(self.matching_clusters) > self.c_max:

                        self.matching_clusters = torch.where((self.cluster_labels[:self.c][:, label] == 1))[0] #*(self.num_pred[:self.c] > self.kappa_n)
                        self.merging_mech.valid_clusters = self.matching_clusters
                        self.removal_mech.remove_overlapping()

                        self.matching_clusters = torch.where((self.cluster_labels[:self.c][:, label] == 1))[0] #*(self.num_pred[:self.c] > self.kappa_n)
                        self.merging_mech.valid_clusters = self.matching_clusters
                        self.removal_mech.removal_mechanism()
                    
                    # S_inv_ = torch.linalg.inv((self.S[:self.c]/
                    #             self.n[:self.c].view(-1, 1, 1))*
                    #             self.feature_dim)
                    # S_inv = self.S_inv[:self.c]
                    # if any(torch.sum(torch.sum(S_inv_-S_inv,dim=2), dim =1)>1e-4):
                    #     print("removal?")
        '''
        scores, preds, clusters = test_model_in_batches(self, (data, labels))

        # Generate a range tensor of valid cluster indices
        valid_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device)

        # Check which clusters in the 'clusters' tensor are not in the 'valid_clusters' tensor
        clusters_to_remove_mask = ~clusters.to(self.device).unsqueeze(1).eq(valid_clusters).any(dim=1)

        # Get the indices of clusters to remove
        clusters_to_remove = torch.where(clusters_to_remove_mask)[0]

        with torch.no_grad():
            # Remove clusters that don't contribute significantly
            for cluster_index in clusters_to_remove:
                self.removal_mech.remove_cluster(cluster_index)
        '''
        
        '''
        preds_list = preds.tolist()
        true_labels_list = labels.tolist()
        clusters_list = clusters.tolist()

        # Initialize a dictionary to track correct and total predictions for each cluster
        cluster_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        # Iterate over each prediction, true label, and cluster
        for pred, true_label, cluster in zip(preds_list, true_labels_list, clusters_list):
            cluster_accuracy[cluster]['total'] += 1
            if pred == true_label:
                cluster_accuracy[cluster]['correct'] += 1

        # Calculate accuracy for each cluster
        for cluster in cluster_accuracy:
            total = cluster_accuracy[cluster]['total']
            correct = cluster_accuracy[cluster]['correct']
            cluster_accuracy[cluster]['accuracy'] = correct / total if total > 0 else 0

        # Sort clusters by accuracy
        sorted_clusters = sorted(cluster_accuracy, key=lambda x: cluster_accuracy[x]['accuracy'], reverse=True)

        # Print the sorted clusters and their accuracies
        for cluster in sorted_clusters:
            accuracy = cluster_accuracy[cluster]['accuracy']
            print(f"Cluster {cluster}: Accuracy = {accuracy:.2f}")
        '''


    def forward(self, data):
        
        # Assuming compute_activation can handle batch data
        self.matching_clusters = torch.arange(self.c).repeat(data.shape[0], 1)
        self.Gamma = self.mathematician.compute_batched_activation(data)
    
        #self.Gamma *= self.score[:self.c].unsqueeze(0)
        #self.matching_clusters = self.matching_clusters[self.n[:self.c]>=self.kappa_n]

        # Evolving mechanisms can be handled here if they can be batch processed

        # Defuzzify label scores for the entire batch
        label_scores, preds_max = self.consequence.defuzzify_batch()  # Adapt this method for batch processing

        # Assuming defuzzify returns batched scores and predictions
        scores = label_scores.clone().detach().requires_grad_(False)
        preds = preds_max
        clusters = self.Gamma.argmax(dim=1)  # Get the cluster indices for the entire batch

        return scores, preds, clusters

    '''
    def forward(self, data, labels):

        scores = []  # List to store scores of the positive class
        pred = []    # List to store predicted class labels
        clusters = []    # List to store predicted class labels
        for (z, _ ) in zip(data, labels):

            # Check if the model is in evaluation mode
            #if not self.training: #In evaluation mode
                
            # In evaluation mode, match all clusters
            self.matching_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device)
            #self.matching_clusters = self.matching_clusters[self.n[:self.c]>=self.kappa_n]
                
            #else: #In training mode
                
                # Update global statistics
                #self.clusterer.update_global_statistics(z)
                
                # In training mode, match clusters based on the label
            #    self.matching_clusters = torch.where(self.cluster_labels[:self.c][:, label] == 1)[0]

            # Compute activation
            self.Gamma = self.mathematician.compute_activation(z)

            # Defuzzify label scores
            # Normalize Gamma by dividing each element by the sum of all elements)
            label_scores, pred_max = self.consequence.defuzzify()

            scores.append(label_scores)  # Extract and store scores for the positive class
            pred.append(pred_max)  # Extract and store class predictions
            clusters.append(self.Gamma.argmax())  # Extract and store class predictions

        scores = torch.vstack(scores).clone().detach().requires_grad_(True)
        pred = torch.tensor(pred)
        clusters = torch.tensor(clusters)

        return scores, pred, clusters
    '''
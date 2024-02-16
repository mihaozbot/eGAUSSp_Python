from tkinter import Label
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
        #self.omega = 0.1
        
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
        self.age = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster age

        self.one_hot_labels = torch.eye(num_classes, dtype=torch.int32, device=device) #One hot labels 
        
        # Trainable parameters
        #Antecedent clusters
        self.n = nn.Parameter(torch.zeros(self.current_capacity, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster sizes
        self.mu = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster means
        self.S = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        self.S_inv = torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device)  # Initialize covariance matrices
        
        #Consequence ARX local linear models 
        self.P0 = (1e1)*torch.eye(feature_dim+1, dtype=torch.float32, device=device, requires_grad=False)
        self.P = nn.Parameter(torch.zeros(self.current_capacity, feature_dim+1, feature_dim+1, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        self.theta =  nn.Parameter(torch.zeros(self.current_capacity, feature_dim+1, self.num_classes, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        
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

        for (x, label) in zip(data, labels):
            sample_count += 1  # Increment the sample counter

            # Progress update every 10,000 samples
            if sample_count % 1000 == 0:
                print(f"Processed {sample_count} samples...")

            # Check if the model is in evaluation mode
            # In evaluation mode, match all clusters
                # Update global statistics
            self.clusterer.update_global_statistics(x, label)
            
            # In training mode, match clusters based on the label
            self.matching_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device)
            
            # Compute activation
            self.Gamma = self.mathematician.compute_activation(x)
            #self.Gamma = self.Gamma**self.omega

            self.matching_clusters = torch.where(self.cluster_labels[:self.c][:, label] == 1)[0]
            
            # Evolving mechanisms
            if self.evolving:
                with torch.no_grad():
  
                    if self.c>0:
                        self.removal_mech.update_score(label)


                    #Incremental clustering and cluster addition
                    self.clusterer.increment_or_add_cluster(x, label)

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
                    
                    if self.c > self.c_max:

                        #threshold = np.exp(-(self.num_sigma) ** 2)
                        #self.valid_clusters = self.matching_clusters[(self.Gamma[self.matching_clusters] > threshold)*
                        #                                                    (self.n[self.matching_clusters] >= self.kappa_n)] #np.sqrt(
                        #self.federal_agent.federated_merging()

                        self.matching_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device)
                        self.removal_mech.removal_mechanism(self.c_max)
                        #self.matching_clusters = torch.where((self.cluster_labels[:self.c][:, label] == 1))[0] #*(self.num_pred[:self.c] > self.kappa_n)
                        
                
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

    def forward(self, x):
        
        # Assuming compute_activation can handle batch data
        self.matching_clusters = torch.arange(self.c).repeat(x.shape[0], 1)
        self.Gamma = self.mathematician.compute_batched_activation(x)
        #self.Gamma = self.Gamma**self.omega
        
        #self.Gamma *= self.score[:self.c].unsqueeze(0)
        #self.matching_clusters = self.matching_clusters[self.n[:self.c]>=self.kappa_n]

        #Evolving mechanisms can be handled here if they can be batch processed

        # Defuzzify label scores for the entire batch
        label_scores, preds_max = self.consequence.defuzzify_batch(x)  # Adapt this method for batch processing

        # Assuming defuzzify returns batched scores and predictions
        scores = label_scores.clone().detach().requires_grad_(False)

        clusters = self.Gamma.argmax(dim=1)  # Get the cluster indices for the entire batch

        return scores, preds_max, clusters
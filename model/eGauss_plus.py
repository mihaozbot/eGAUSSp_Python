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
        self.Gamma = torch.empty(0, dtype=torch.float64, device=device,requires_grad=True)
        self.current_capacity = math.ceil(c_max/2) #Initialize current capacity, which will be expanded as needed during training 
        self.cluster_labels = torch.empty((self.current_capacity, num_classes), dtype=torch.int32, device=device) #Initialize cluster labels
        #self.label_to_clusters = {} #Initialize dictionary to map labels to clusters
        
        self.score = torch.empty((self.current_capacity,), dtype=torch.float64, device=device) #Initialize cluster labels
        
        self.one_hot_labels = torch.eye(num_classes, dtype=torch.int32) #One hot labels 
        
        # Trainable parameters
        self.n = nn.Parameter(torch.zeros(self.current_capacity, dtype=torch.float64, device=device, requires_grad=True))  # Initialize cluster sizes
        self.mu = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, dtype=torch.float64, device=device, requires_grad=True))  # Initialize cluster means
        self.S = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float64, device=device, requires_grad=True))  # Initialize covariance matrices

        # Global statistics
        self.n_glo = torch.zeros((num_classes), dtype=torch.float64, device=device)  # Global number of sampels per class
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float64, device=device)  # Global mean
        self.S_glo = torch.zeros((feature_dim), dtype=torch.float64, device=device)  # Sum of squares for global variance

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
    
    def federated_mergingg(self):
        ''' Executes the merging mechanism for all rules. Conversely, the normal merging mechanism works on a subset of rules based on some conditions. '''
        
        self.federal_agent.federated_merging()

    def clustering(self, data, labels):

        for (z, label) in zip(data, labels):

            # Check if the model is in evaluation mode
            # In evaluation mode, match all clusters
                # Update global statistics
            self.clusterer.update_global_statistics(z, label)
            
            self.matching_clusters = torch.arange(self.c, dtype=torch.int64, device=self.device)
            
            # Compute activation
            self.Gamma = self.mathematician.compute_activation(z)

            # In training mode, match clusters based on the label
            self.matching_clusters = torch.where(self.cluster_labels[:self.c][:, label] == 1)[0]
            
            # Evolving mechanisms
            if self.evolving:
                with torch.no_grad():
                                  
                    self.removal_mech.update_score(label)

                    #Incremental clustering and cluster addition
                    self.clusterer.increment_or_add_cluster(z, label)
            
                    #Cluster merging
                    self.merging_mech.merging_mechanism()
    
                    #Removal mechanism
                    #self.removal_mech.removal_mechanism()

    def forward(self, data):
        
        # Assuming compute_activation can handle batch data
        self.matching_clusters = torch.arange(self.c).repeat(data.shape[0], 1)
        self.Gamma = self.mathematician.compute_batched_activation(data)
        #self.matching_clusters = self.matching_clusters[self.n[:self.c]>=self.kappa_n]

        # Evolving mechanisms can be handled here if they can be batch processed

        # Defuzzify label scores for the entire batch
        label_scores, preds_max = self.consequence.defuzzify_batch()  # Adapt this method for batch processing

        # Assuming defuzzify returns batched scores and predictions
        scores = label_scores.clone().detach().requires_grad_(True)
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
            self.matching_clusters = torch.arange(self.c, dtype=torch.int64, device=self.device)
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
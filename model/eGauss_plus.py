import torch
import torch.nn as nn

# Attempt to load the line_profiler extension
try:
    from line_profiler import LineProfiler
    profile = LineProfiler()  # If line_profiler is available, use it
except ImportError:
    # If line_profiler is not available, define a dummy profile decorator
    def profile(func): 
        return func

from model.clustering_operations import ClusteringOps
from model.removal_mechanism import RemovalMechanism 
from model.merging_mechanism import MergingMechanism
from model.math_operations import MathOps
from model.consequence_operations import ConsequenceOps
from model.model_operations import ModelOps
from model.federated_operations import FederalOps

class eGAUSSp(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, N_max, num_sigma, kappa_join, S_0, c_max, device):
        super(eGAUSSp, self).__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.N_max = N_max
        self.num_sigma = num_sigma
        self.kappa_join = kappa_join
        self.S_0 = S_0 * torch.eye(self.feature_dim, device=self.device)
        self.S_0_initial = self.S_0.clone() 
        self.c_max = c_max
        self.num_classes = num_classes
        
        # Dynamic properties initialized with tensors
        self.c = 0 # Number of active clusters
        self.Gamma = torch.empty(0, dtype=torch.float32, device=device,requires_grad=True)
        self.current_capacity = 2*c_max #Initialize current capacity, which will be expanded as needed during training 
        self.cluster_labels = torch.empty((self.current_capacity), dtype=torch.int32, device=device) #Initialize cluster labels
        self.label_to_clusters = {} #Initialize dictionary to map labels to clusters
        
        # Trainable parameters
        self.n = nn.Parameter(torch.zeros((self.current_capacity), requires_grad=True, device = device)) #Initialize cluster sizes 
        self.mu = nn.Parameter(torch.zeros((self.current_capacity, feature_dim), requires_grad=True, device = device)) #Initialize cluster means
        self.S = nn.Parameter(torch.zeros((self.current_capacity, feature_dim, feature_dim), requires_grad=True, device = device)) #Initialize covariance matrices

        # Global statistics
        self.n_glo = 0  # Total number of samples processed globally
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Global mean
        self.S_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Sum of squares for global variance

        # Initialize subclasses
        self.overseer = ModelOps(self)
        self.mathematician = MathOps(self)
        self.clustering = ClusteringOps(self)
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

    def merge_model(self, client_model):  
        ''' Merges a client model into the main model. '''
        self.federal_agent.merge_model(client_model)
    
    def federated_mergingg(self):
        ''' Executes the merging mechanism for all rules. Conversely, the normal merging mechanism works on a subset of rules based on some conditions. '''
        
        self.federal_agent.federated_merging()

    @profile
    def forward(self, z, label):

        # Check if the model is in evaluation mode
        if not self.training: #In evaluation mode
            
            # In evaluation mode, match all clusters
            self.matching_clusters = torch.arange(self.c, dtype=torch.int64, device=self.device)
            
        else: #In training mode
            
            # Update global statistics
            self.clustering.update_global_statistics(z)
            
            # In training mode, match clusters based on the label
            self.matching_clusters = torch.where(self.cluster_labels[:self.c] == label)[0]

        # Compute activation
        self.Gamma = self.mathematician.compute_activation(z)

        # Evolving mechanisms
        with torch.no_grad():
            
            #Incremental clustering and cluster addition
            if self.enable_adding:
                self.clustering.increment_or_add_cluster(z, label)

            #Cluster merging
            if self.enable_merging:
                self.merging_mech.merging_mechanism()

        # Defuzzify label scores
        label_scores = self.consequence.defuzzify(z)

        return label_scores

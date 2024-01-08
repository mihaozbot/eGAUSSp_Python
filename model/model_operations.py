import logging
import torch
import torch.nn as nn
from torch.nn import Parameter
import math

class ModelOps:
    def __init__(self, parent):
        self.parent = parent
        
        # Flags
        self.parent.enable_debugging = False
        self.parent.enable_adding = True
        self.parent.enable_merging = True

        # Initialize the logging system
        #self.initialize_logging()
    
    def ensure_capacity(self, new_c):
        """
        Adjust the capacity of the model to accommodate a new number of active clusters.
        """
        with torch.no_grad():  # Disable gradient tracking for tensor resizing
            
            # Determine whether to expand or contract the model capacity
            should_expand = new_c >= self.parent.current_capacity
            should_contract = (new_c > 2 * self.parent.c_max) and (self.parent.current_capacity > 2 * self.parent.N_r) and (new_c < (self.parent.current_capacity / 2 - 1))

            if should_expand or should_contract:
                self.modify_capacity(new_c)
                #Note that, 2*self.parent.c_max is the minimal capacity, which also handles c = 1
            
    def modify_capacity(self, new_c):
        
        new_capacity = 2 ** math.ceil(math.log2(new_c))  # Find the next power of two greater than the given number
        
        self.parent.mu = nn.Parameter(self._resize_tensor(self.parent.mu, (new_capacity, self.parent.feature_dim)), requires_grad=False)
        self.parent.S = nn.Parameter(self._resize_tensor(self.parent.S, (new_capacity, self.parent.feature_dim, self.parent.feature_dim)), requires_grad=False)
        self.parent.n = nn.Parameter(self._resize_tensor(self.parent.n, (new_capacity,)), requires_grad=False)
        
        self.parent.S_inv = self._resize_tensor(self.parent.S_inv, (new_capacity, self.parent.feature_dim, self.parent.feature_dim))
        self.parent.cluster_labels = self._resize_tensor(self.parent.cluster_labels, (new_capacity,self.parent.num_classes))
        self.parent.score = self._resize_tensor(self.parent.score, (new_capacity,))
        self.parent.num_pred = self._resize_tensor(self.parent.num_pred, (new_capacity,))
        self.parent.age = self._resize_tensor(self.parent.age, (new_capacity,))

        self.parent.current_capacity = new_capacity

    def _resize_tensor(self, old_tensor, new_size):
        new_tensor = torch.empty(new_size, dtype=old_tensor.dtype, device=old_tensor.device) #Create new tensor
        new_tensor[:self.parent.c] = old_tensor[:self.parent.c] #Copy old tensor into the new tensor
        return new_tensor
    
    def toggle_adding(self, enable=None):
        if enable is None:
            self.parent.enable_adding = not self.parent.enable_adding
            state = "enabled" if self.parent.enable_adding else "disabled"
        else:
            self.parent.enable_adding = enable
            state = "enabled" if enable else "disabled"
        print(f"Cluster adding has been {state}.")

    def toggle_merging(self, enable=None):
        if enable is None:
            self.parent.enable_merging = not self.parent.enable_merging
            state = "enabled" if self.parent.enable_merging else "disabled"
        else:
            self.parent.enable_merging = enable
            state = "enabled" if enable else "disabled"
        print(f"Cluster merging has been {state}.")

    def toggle_debugging(self, enable=None):
        if enable is None:
            self.parent.enable_debugging = not self.parent.enable_debugging
            state = "enabled" if self.parent.enable_debugging else "disabled"
        else:
            self.parent.enable_debugging = enable
            state = "enabled" if enable else "disabled"
        print(f"Debugging has been {state}.")
        
    def toggle_evolving(self, enable=None):
        
        if enable is None:
            self.parent.evolving = not self.parent.evolving
        else:
            self.parent.evolving = enable
        
        # Ensure that adding and merging are aligned with the evolving state
        self.parent.enable_adding = self.parent.evolving
        self.parent.enable_merging = self.parent.evolving

        # Print the new state
        state = "enabled" if self.parent.evolving else "disabled"
        print(f"Evolving has been {state}.")

        
    def initialize_logging(self):
        """Initialize logging and record initial parameters."""
        
        # Set up logging
        logging.basicConfig(filename='eGAUSSp_super.log', level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.critical(f"Critical. ***************** New model created! *******************")
        
        # Log initial parameters
        logging.info(f"Feature Dimension: {self.parent.feature_dim}")
        logging.info(f"Number of Classes: {self.parent.c}")  # Assuming you want to log this as well
        logging.info(f"Number of samples: {self.parent.kappa_n}")
        logging.info(f"Number of Sigmas: {self.parent.num_sigma}")
        logging.info(f"Kappa Join: {self.parent.kappa_join}")
        logging.info(f"S_0: {self.parent.S_0}")
        logging.info(f"Device: {self.parent.device}")


import torch
import torch.nn as nn

class ConsequenceOps():
    def __init__(self, parent):
        self.parent = parent


    def defuzzify(self, z):
        
        # Filter out unlabeled cluster labels (assuming -1 indicates unlabeled)
        #labeled_indices = self.parent.cluster_labels[0:self.parent.c] != -1
        normalized_gamma = self.compute_normalized_gamma()
        
        # Find the index of the maximum value in Gamma
        max_index = torch.argmax(normalized_gamma)

        if 1: #class0:
            # Select only labeled data for Gamma and cluster labels
            label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * self.parent.cluster_labels[:self.parent.c], dim=0)

            # Retrieve the corresponding label
            max_label = torch.argmax(self.parent.cluster_labels[max_index])
            
            #max_label = torch.argmax(label_scores)
            
        else: #class1 
            phi = torch.cat((z, torch.tensor([1],device=self.parent.device))).unsqueeze(1)  # Concatenating the two tensors
            label_scores = torch.softmax(torch.mm(phi.T, self.parent.theta[max_index]))
            max_label = label_scores
        
        return label_scores, max_label

    '''
    def defuzzify_batch(self):

        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.compute_batched_normalized_gamma()
        
        # Select only labeled data for Gamma and cluster labels
        expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)

        # Compute label scores
        label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels, dim=1)

        # Find the indices of the maximum values in label_scores along the label dimension
        max_labels = torch.argmax(label_scores, dim=1)

        return label_scores, max_labels
    '''
    
    def defuzzify_batch(self, Z):
        # Normalize Gamma along the cluster dimension
        normalized_gamma = self.compute_batched_normalized_gamma()

        # Find the indices of the maximum values in normalized_gamma along the cluster dimension
        max_indices = torch.argmax(normalized_gamma, dim=1)

        if 0: #Class0
            # Compute label scores
            expanded_cluster_labels = self.parent.cluster_labels[:self.parent.c].unsqueeze(0).expand(normalized_gamma.shape[0], -1, -1)
            
            # Compute label scores
            label_scores = torch.sum(normalized_gamma.unsqueeze(-1) * expanded_cluster_labels, dim=1)

            # Use max_indices to select the corresponding one-hot encoded class labels
            one_hot_max_labels = expanded_cluster_labels[torch.arange(normalized_gamma.shape[0]), max_indices]

            # Convert one-hot encoding to class indices
            max_labels = torch.argmax(one_hot_max_labels, dim=1)
            
        else: #Class1
            
            # Add bias term to input and prepare for batch operation
            phi = torch.cat((Z, torch.ones(Z.shape[0], 1, device=self.parent.device)), dim=1)  # [batch_size, num_features+1]
            
            # Compute scores for all clusters
            all_scores = torch.einsum('bf, cfo -> boc', phi, self.parent.theta[:self.parent.c])  # [batch_size, output_dim, num_clusters]
            
            # Weight scores by normalized_gamma and sum across clusters
            weighted_scores =  torch.sum(all_scores.transpose(1, 2) *normalized_gamma.unsqueeze(-1), dim=1) 
            
            # Compute softmax across weighted scores for class probabilities
            label_scores = torch.softmax(weighted_scores, dim=1)  # [batch_size, num_classes]
            
            # Determine the predicted class labels
            max_labels = torch.argmax(label_scores, dim=1)         
            #max_labels = torch.argmax(torch.einsum('bf, bfo -> bo', phi, self.parent.theta[max_indices]), dim=1)  
            
        return label_scores, max_labels


    def compute_normalized_gamma(self):
        
        # Compute normalized gamma
        gamma = self.parent.Gamma[0:self.parent.c]
        gamma_sum = gamma.sum()

        # Avoid division by zero or NaN; if gamma_sum is not valid, set normalized_gamma to zero
        if gamma_sum == 0 or torch.isnan(gamma_sum):
            normalized_gamma = torch.zeros_like(gamma)
        else:
            normalized_gamma = gamma / gamma_sum
            # Replace NaN values in normalized_gamma with zeros
            normalized_gamma = torch.nan_to_num(normalized_gamma)

        return normalized_gamma

    def compute_batched_normalized_gamma(self):
        # Compute normalized gamma
        gamma = self.parent.Gamma[:,:self.parent.c]
        gamma_sum = gamma.sum(dim=1, keepdim=True)

        # Replace NaN values in gamma_sum with zeros
        gamma_sum = torch.nan_to_num(gamma_sum, nan= 0.0)

        # Avoid division by zero; if gamma_sum is zero, set normalized_gamma to zero
        normalized_gamma = torch.where(gamma_sum > 0, gamma / gamma_sum, torch.zeros_like(gamma))

        return normalized_gamma

    def recursive_least_squares(self, z, y, j):

        normalized_gamma = self.compute_normalized_gamma()
        phi = torch.cat((z, torch.tensor([1], device=self.parent.device))).unsqueeze(1)  # Concatenating the two tensors
        
        forgetting_factor = 0.95
        
        if 1:
            
            gain = torch.mm(self.parent.P[j], phi) / (torch.mm(torch.mm(phi.T, self.parent.P[j]), phi) + 1)
            self.parent.P[j] = (torch.eye(self.parent.feature_dim+1 , device=self.parent.device) - torch.mm(gain, phi.T)) * self.parent.P[j]
            e_RLS = (y - torch.mm(phi.T, self.parent.theta[j]))#*self.parent.cluster_labels[j]
            self.parent.theta[j] = self.parent.theta[j] + gain * e_RLS
            
        else:

            # Number of clusters
            c = self.parent.c
            
            '''
            # Update the gain for all clusters simultaneously
            gain_den = torch.matmul(torch.matmul(phi.transpose(-2, -1), self.parent.P[:c]), phi) + forgetting_factor/normalized_gamma.unsqueeze(-1).unsqueeze(-1) 
            gain = torch.matmul(self.parent.P[:c], phi) / gain_den

            # Update P for all clusters
            identity = torch.eye(self.parent.P[:c].shape[-1], device=self.parent.device).unsqueeze(0).repeat(c, 1, 1)
            self.parent.P[:c] = (identity - torch.matmul(gain, phi.transpose(-2, -1))) * self.parent.P[:c] / forgetting_factor

            # Compute e_RLS and update theta for all clusters
            y_hat = torch.matmul(self.parent.theta[:c].transpose(-2, -1), phi).squeeze(-1)
            e_RLS = (y.unsqueeze(0) - y_hat)#*self.parent.cluster_labels[:self.parent.c]
            self.parent.theta[:c] = self.parent.theta[:c] + torch.matmul(gain, e_RLS.unsqueeze(1))
            
            #print(y[torch.argmax(torch.sum(torch.matmul(self.parent.theta[:c].transpose(-2, -1), phi).squeeze(-1)* normalized_gamma.unsqueeze(-1),dim=0))])
            
            '''
            for cluster_idx in range(c):
                # Update the gain for the current cluster
                gain_den = torch.matmul(torch.matmul(phi.transpose(-2, -1), self.parent.P[cluster_idx]), phi) * normalized_gamma[cluster_idx] + 1
                gain = torch.matmul(self.parent.P[cluster_idx], phi) * normalized_gamma[cluster_idx] / gain_den
                
                # Update P for the current cluster
                identity = torch.eye(self.parent.P[cluster_idx].shape[-1], device=self.parent.device)
                self.parent.P[cluster_idx] = (identity - torch.matmul(gain, phi.transpose(-2, -1))) * self.parent.P[cluster_idx] / forgetting_factor
                
                # Compute e_RLS and update theta for the current cluster
                theta_phi = torch.matmul(self.parent.theta[cluster_idx].transpose(-2, -1), phi).squeeze(-1)
                e_RLS = (y.unsqueeze(0) - theta_phi)*self.parent.cluster_labels[cluster_idx]
                self.parent.theta[cluster_idx] = self.parent.theta[cluster_idx] + torch.matmul(gain, e_RLS.unsqueeze(1))
                

            
            '''
            #Update the gain
            gain_den = torch.matmul( torch.matmul(phi.T , self.parent.P[:self.parent.c]), phi)*normalized_gamma.unsqueeze(-1).unsqueeze(-1) + forgetting_factor
            gain = torch.matmul(self.parent.P[:self.parent.c], phi)*normalized_gamma.unsqueeze(-1).unsqueeze(-1) / gain_den

            # Update P
            identity = torch.eye(self.parent.P[:self.parent.c].shape[1], device=self.parent.device).unsqueeze(0)
            self.parent.P[:self.parent.c] = (identity - torch.matmul(gain, phi.T )) * self.parent.P[:self.parent.c]/forgetting_factor

            # Compute e_RLS and update theta
            theta_phi = torch.matmul(self.parent.theta[:self.parent.c].transpose(1, 2), phi).squeeze(-1)
            
            e_RLS = (y.unsqueeze(0) - theta_phi)*self.parent.cluster_labels[:self.parent.c]
            self.parent.theta[:self.parent.c] = self.parent.theta[:self.parent.c] + torch.matmul(gain, e_RLS.unsqueeze(1))
    '''
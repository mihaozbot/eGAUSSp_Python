import torch
import numpy as np
import torch.nn as nn

class eGAUSSpPyTorchV3_fixed(nn.Module):
    def __init__(self, feature_dim, N_max, Gamma_max, kappa_join, S_0):
        super(eGAUSSpPyTorchV3_fixed, self).__init__()

        self.feature_dim = feature_dim
        self.N_max = N_max
        self.Gamma_max = Gamma_max
        self.kappa_join = kappa_join
        self.S_0 = S_0

        self.c = 1
        self.n = [1]
        self.mu = [nn.Parameter(torch.zeros(feature_dim), requires_grad=True)]
        self.S = [nn.Parameter(torch.eye(feature_dim), requires_grad=True)]

        self.V_factor = (2 * np.pi ** (feature_dim / 2) / (feature_dim * torch.exp(torch.lgamma(torch.tensor(float(feature_dim) / 2)))))
                    
    def compute_distance_and_activation(self, z):
        Gamma = torch.zeros(self.c)
        d2 = torch.zeros(self.c)
        for i in range(self.c):
            if self.n[i] < self.N_max:
                d2[i] = torch.norm(z - self.mu[i]) ** 2
            else:
                d2[i] = (z - self.mu[i]).T @ torch.pinverse(self.S[i] / self.n[i]) @ (z - self.mu[i])
            Gamma[i] = torch.exp(-d2[i])
        return Gamma, d2

    def increment_or_add_cluster(self, z, Gamma):
        _, j = torch.max(Gamma, 0)
        if Gamma[j] > self.Gamma_max:
            e = z - self.mu[j]
            updated_mu = self.mu[j] + 1 / (1 + self.n[j]) * e
            updated_S = self.S[j] + e.view(-1, 1) @ (z - self.mu[j]).view(1, -1)
            updated_n = self.n[j] + 1
            
            # Assign the new values
            self.mu[j] = updated_mu
            self.S[j] = updated_S
            self.n[j] = updated_n
        else:
            self.c += 1
            self.n.append(1)
            self.mu.append(nn.Parameter(z.clone(), requires_grad=True))
            self.S.append(nn.Parameter(self.S_0 * torch.eye(self.feature_dim), requires_grad=True))
            Gamma = torch.cat((Gamma, torch.tensor([1.0])))
        return Gamma

    def merge_clusters(self, Gamma):
        merge = True
        max_iterations = 100
        iteration = 0
        
        while merge and iteration < max_iterations:
            V = torch.full((self.c, self.c), float('nan'))
            Sigma_ij = torch.zeros(self.feature_dim, self.feature_dim, self.c, self.c)
            mu_ij = torch.zeros(self.feature_dim, self.c, self.c)
            n_ij = torch.zeros(self.c, self.c)
            for i in range(self.c):
                if Gamma[i] > self.Gamma_max / 4:
                    # Using torch.linalg.eig
                    V[i, i] = self.V_factor * torch.prod(torch.linalg.eig(self.S[i] / self.n[i]).eigenvalues.real)
                    for j in range(i + 1, self.c):
                        if Gamma[j] > self.Gamma_max / 4:
                            n_ij[i, j] = self.n[i] + self.n[j]
                            mu_ij[:, i, j] = (self.n[i] * self.mu[i] + self.n[j] * self.mu[j]) / n_ij[i, j]
                            ZiTZi = (self.n[i] - 1) * (1 / self.n[i]) * self.S[i] + torch.diag(self.mu[i]) @ torch.ones(self.feature_dim, self.feature_dim) @ torch.diag(self.mu[i])
                            ZjTZj = (self.n[j] - 1) * (1 / self.n[j]) * self.S[j] + torch.diag(self.mu[j]) @ torch.ones(self.feature_dim, self.feature_dim) @ torch.diag(self.mu[j])
                            Sigma_ij[:, :, i, j] = (1 / (n_ij[i, j] - 1)) * (ZiTZi + ZjTZj - torch.diag(mu_ij[:, i, j]) @ torch.ones(self.feature_dim, self.feature_dim) @ torch.diag(mu_ij[:, i, j]))
                            # Using torch.linalg.eig
                            V[i, j] = self.V_factor * torch.prod(torch.linalg.eig(Sigma_ij[:, :, i, j]).eigenvalues.real)
                            if V[i, j] < 0:
                                V[i, j] = float('nan')

            kappa = torch.full((self.c, self.c), float('inf'))
            for i in range(self.c):
                for j in range(i + 1, self.c):
                    kappa[i, j] = V[i, j] / (V[i, i] + V[j, j])

            kappa_min = torch.min(kappa[kappa == kappa])
            i, j = (kappa == kappa_min).nonzero(as_tuple=True)

            if kappa_min < self.kappa_join:
                self.mu[i] = nn.Parameter((self.n[i] * self.mu[i] + self.n[j] * self.mu[j]) / (self.n[i] + self.n[j]), requires_grad=True)
                self.S[i] = nn.Parameter(self.S[i] + self.S[j], requires_grad=True)
                self.n[i] += self.n[j]

                del self.n[j]
                del self.mu[j]
                del self.S[j]

                self.c -= 1
                merge = True
            else:
                merge = False
            
            iteration += 1

    def forward(self, z):
        Gamma, d2 = self.compute_distance_and_activation(z)
        Gamma = self.increment_or_add_cluster(z, Gamma)
        self.merge_clusters(Gamma)
        return torch.stack(self.mu)

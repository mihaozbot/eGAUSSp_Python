from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

def train_supervised(model, client_data):
        
    #Toggle training mode
    model.toggle_evolving(True)
    model.train()
    
    data, labels = client_data
    data, labels = data.to(model.device), labels.to(model.device)

    # Training loop
    model.forward(data, labels)  # Train the model

def train_models_in_threads(models, datasets, debugging = False):
    threads = []

    for model, dataset in zip(models, datasets):
        model.toggle_debugging(debugging)
        thread = threading.Thread(target=train_supervised, args=(model, dataset))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def train_unsupervised(model, client_data):

    #Toggle training mode
    model.toggle_evolving(True)

    model.train()

    data, _ = client_data
    data = data.to(model.device)

    dummy_labels = torch.full((len(data),), 0, dtype=torch.int32)

    # Training loop
    model.forward(data, dummy_labels)  # Train the model


def test_model(model, dataset):
    data, _ = dataset
    data = data.to(model.device)
    
    #Turn off training
    model.toggle_evolving(False)
    model.eval()

    dummy_labels = torch.full((len(data),), -1, dtype=torch.int32, device = model.device)
    all_scores, pred_max, clusters = model(data, dummy_labels)  # Forward pass

    return all_scores, pred_max, clusters

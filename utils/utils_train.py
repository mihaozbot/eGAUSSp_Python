import threading
import torch
from torch.utils.data import TensorDataset, DataLoader


def train_supervised(model, client_data):

    #Toggle training mode
    model.toggle_evolving(True)
    model.train()
    
    data, labels = client_data
    data, labels = data.to(model.device), labels.to(model.device)

    # Training loop
    model.clustering(data, labels)  # Train the model

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
    model.clustering(data, dummy_labels)  # Train the model

def test_model(model, dataset):
    data, _ = dataset
    data = data.to(model.device)
    
    #Turn off training, although it is not needed
    model.toggle_evolving(False)
    model.eval()

    #dummy_labels = torch.full((len(data),), -1, dtype=torch.int32, device = model.device)
    all_scores, pred_max, clusters = model.forward(data)  # Forward pass

    return all_scores, pred_max, clusters


def test_model_in_batches(model, dataset, batch_size=500):
    model.eval()
    model.toggle_evolving(False)

    # Create a DataLoader
    data, labels = dataset
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_scores_list = []
    pred_max_list = []
    clusters_list = []

    # Process the dataset in batches
    for data, _ in data_loader:
        data = data.to(model.device)

        # Forward pass for the current batch
        all_scores, pred_max, clusters = model.forward(data)

        # Store the results
        all_scores_list.append(all_scores.cpu())
        pred_max_list.append(pred_max.cpu())
        clusters_list.append(clusters.cpu())

    # Concatenate all results
    all_scores_concat = torch.cat(all_scores_list, dim=0)
    pred_max_concat = torch.cat(pred_max_list, dim=0)
    clusters_concat = torch.cat(clusters_list, dim=0)

    return all_scores_concat, pred_max_concat, clusters_concat
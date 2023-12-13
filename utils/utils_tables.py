from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import numpy as np
import torch

def train_supervised(model, client_data):
        
    #Toggle training mode
    model.toggle_evolving(True)
    model.train()
    
    data, labels = client_data
    
    # Training loop
    for idx, (z, label) in enumerate(zip(data, labels)):
        model.forward(z, label)  # Train the model

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(data):
            print(f"Processed {idx + 1} points.Number of clusters: {model.c}")

def train_models_in_threads(models, datasets):
    threads = []

    for model, dataset in zip(models, datasets):
        thread = threading.Thread(target=train_supervised, args=(model, dataset))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def train_unsupervised(model, client_data):
        
    data, labels = client_data
    
    #Toggle training mode
    model.toggle_evolving(True)
    model.train()

    # Training loop
    for z in data:
        model.forward(z, torch.tensor(-1, dtype=torch.int64))  # Train the model


def test_model(model, test_dataset):
    
    # Extract features and labels from the dataset
    test_data, test_labels = test_dataset
    
    #y_test = y_test.numpy()  # Convert y_test to numpy array for evaluation metrics

    #Toggle evaluation mode
    model.toggle_evolving(False)
    model.eval()

    pred_max = []
    for z in test_data:
        output = model.forward(z, -1)  # Forward pass
        pred_max.append(output.argmax())  # Assuming pred is a tensor of class scores

    pred_max = torch.tensor(pred_max)

    # Evaluation metrics with zero_division parameter
    accuracy = accuracy_score(test_labels, pred_max)
    precision = precision_score(test_labels, pred_max, average='weighted', zero_division=1)
    recall = recall_score(test_labels, pred_max, average='weighted', zero_division=1)
    f1 = f1_score(test_labels, pred_max, average='weighted', zero_division=1)

    # Create a dictionary to return the metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return metrics

def calculate_metrics_statistics(metrics_list):
    """Calculate the average and standard deviation of given metrics."""
    mean_metrics = {key: np.mean([metrics[key] for metrics in metrics_list]) for key in metrics_list[0]}
    std_metrics = {key: np.std([metrics[key] for metrics in metrics_list], ddof=1) for key in metrics_list[0]}
    return format_metrics(mean_metrics, std_metrics)

def format_metrics(mean_metrics, std_metrics):
    """Format the metrics as 'mean ± standard deviation'."""
    formatted = {key: f"{mean_metrics[key]:.2f} ± {std_metrics[key]:.2f}" for key in mean_metrics}
    return formatted

def calculate_cluster_stats(cluster_counts):
    """Calculate the average and standard deviation of cluster counts."""
    avg_clusters = np.mean(cluster_counts)
    std_clusters = np.std(cluster_counts, ddof=1)
    return avg_clusters, std_clusters

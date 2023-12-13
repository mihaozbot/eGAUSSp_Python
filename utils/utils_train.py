from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

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
    test_data, _ = test_dataset
    
    #Turn off training
    model.toggle_evolving(False)
    model.eval()

    all_scores = []  # List to store scores of the positive class
    pred_max = []    # List to store predicted class labels
    for z in test_data:
        output = model(z, -1)  # Forward pass
        all_scores.append(output.detach())  # Extract and store scores for the positive class
        pred_max.append(output.argmax().detach())  # Extract and store class predictions

    all_scores = torch.tensor(np.vstack(all_scores))  # Convert list to numpy array
    pred_max = torch.tensor(pred_max)      # Convert list to numpy array

    return all_scores, pred_max


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(pred_max, test_dataset):
    _, test_labels = test_dataset
        
    accuracy = accuracy_score(test_labels, pred_max)
    precision = precision_score(test_labels, pred_max, average='weighted', zero_division='warn')
    recall = recall_score(test_labels, pred_max, average='weighted', zero_division='warn')
    f1 = f1_score(test_labels, pred_max, average='weighted', zero_division='warn')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def calculate_roc_auc(outputs, test_dataset):
    
    _, test_labels = test_dataset
        
    positive_class_scores = outputs[:, 1]  # Assuming index 1 is the positive class
    roc_auc = roc_auc_score(test_labels, positive_class_scores)
    
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, positive_class_scores)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

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

def plot_confusion_matrix(pred_max, test_dataset):
    """
    Plots the confusion matrix.
    
    Args:
    pred_max: Predicted labels.
    test_labels: True labels.
    class_names: List of class names for the labels.
    """
        
    _, test_labels = test_dataset
    
    # Determine the number of unique classes
    num_classes = len(np.unique(test_labels))
    class_names = [str(i) for i in range(num_classes)]

    # Compute the confusion matrix
    cm = confusion_matrix(test_labels, pred_max)

    # Define the figure size
    fig_width = 4
    fig_height = 3
    plt.figure(figsize=(fig_width, fig_height))
    
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
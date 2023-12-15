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
    
    # Training loop
    model.forward(data, labels)  # Train the model

def train_models_in_threads(models, datasets):
    threads = []

    for model, dataset in zip(models, datasets):
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
    dummy_labels = torch.full((len(data),), 0, dtype=torch.int32)

    # Training loop
    model.forward(data, dummy_labels)  # Train the model


def test_model(model, test_dataset):
    test_data, _ = test_dataset
    
    #Turn off training
    model.toggle_evolving(False)
    model.eval()

    dummy_labels = torch.full((len(test_data),), -1, dtype=torch.int32)
    all_scores, pred_max = model(test_data, dummy_labels)  # Forward pass

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
        
    positive_class_scores = outputs[:, 1].detach().numpy() # Assuming index 1 is the positive class
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


def calculate_unsupervised_metrics(X, labels):
    """
    Calculate unsupervised clustering metrics for the given data and labels.

    :param X: Feature set.
    :param labels: Predicted labels for each data point.
    :return: A dictionary containing the computed metrics.
    """

    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Note: The following metrics require true labels to be meaningful.
    # They are included here for completeness, but should be used only if true labels are available.
    # adjusted_rand = adjusted_rand_score(true_labels, labels)
    # normalized_mutual_info = normalized_mutual_info_score(true_labels, labels)
    # homogeneity = homogeneity_score(true_labels, labels)
    # completeness = completeness_score(true_labels, labels)
    # v_measure = v_measure_score(true_labels, labels)

    return {
        "silhouette_score": silhouette,
        "davies_bouldin_score": davies_bouldin,
        "calinski_harabasz_score": calinski_harabasz,
        # "adjusted_rand_score": adjusted_rand,
        # "normalized_mutual_info_score": normalized_mutual_info,
        # "homogeneity_score": homogeneity,
        # "completeness_score": completeness,
        # "v_measure_score": v_measure
    }

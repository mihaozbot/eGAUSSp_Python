
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(pred_max, test_dataset, weight):
    _, test_labels = test_dataset

    accuracy = accuracy_score(test_labels, pred_max)
    precision = precision_score(test_labels, pred_max, average=weight, zero_division='warn')
    recall = recall_score(test_labels, pred_max, average=weight, zero_division='warn')
    f1 = f1_score(test_labels, pred_max, average=weight, zero_division='warn')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def calculate_roc_auc(outputs, test_dataset):
    
    _, test_labels = test_dataset
        
    positive_class_scores = outputs[:, 1].detach().cpu().numpy() # Assuming index 1 is the positive class
    roc_auc = roc_auc_score(test_labels, positive_class_scores)
    
    # Calculate ROC Curve
    fpr, tpr, _ = roc_curve(test_labels, positive_class_scores)

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
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


def calculate_unsupervised_metrics(assignments, dataset):
    """
    Calculate unsupervised clustering metrics for the given data and labels.

    :param assignments: Cluster with highest membership for each data point.
    :param dataset: Feature set.
    :return: A dictionary containing the computed metrics.
    """    
    data, labels = dataset

    silhouette = silhouette_score(data, assignments)
    davies_bouldin = davies_bouldin_score(data, assignments)
    calinski_harabasz = calinski_harabasz_score(data, assignments)

    # Note: The following metrics require true labels to be meaningful.
    # They are included here for completeness, but should be used only if true labels are available.
    adjusted_rand = adjusted_rand_score(labels, assignments)
    normalized_mutual_info = normalized_mutual_info_score(labels, assignments)
    homogeneity = homogeneity_score(labels, assignments)
    completeness = completeness_score(labels, assignments)
    v_measure = v_measure_score(labels, assignments)

    return {
        "silhouette_score": silhouette,
        #"davies_bouldin_score": davies_bouldin,
        #"calinski_harabasz_score": calinski_harabasz,
        #"adjusted_rand_score": adjusted_rand,
        "normalized_mutual_info_score": normalized_mutual_info,
        #"homogeneity_score": homogeneity,
        #"completeness_score": completeness,
        "v_measure_score": v_measure
    }

def compute_bic(memberships, num_params):
    """
    Compute the BIC, given the membership probabilities and the number of parameters.

    :param assignments: An array where each row corresponds to a data point and each column to a cluster.
    :param n_params: The number of parameters in the model.
    :return: The computed BIC.
    """
    # Number of data points
    n = memberships.shape[0]

    one_hot_encodings = np.eye(np.unique(memberships))[memberships]

    # Calculate the log-likelihood
    log_likelihood = np.sum(np.log(one_hot_encodings + 1e-15)) # Avoid log(0)

    # Calculate BIC
    bic = -2 * log_likelihood + num_params * np.log(n)
    return bic

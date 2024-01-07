import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
#from IPython.display import clear_output
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def calculate_mean_std(metrics, key):
    # Calculate mean and standard deviation for the given key in client metrics
    means = []
    stds = []

    num_rounds = len(metrics)
    for round_idx in range(num_rounds):
        round_values = [client_metric['binary'][key]  for client_metric in metrics[round_idx]['client_metrics']]
        means.append(np.mean(round_values))
        stds.append(np.std(round_values))

    return means, stds

def plot_with_intervals(rounds, means, stds, metric_name):
    plt.figure(figsize=(12, 6))
    plt.errorbar(rounds, means, yerr=stds, fmt='o-', ecolor='lightgray', elinewidth=3, capsize=0, label=metric_name)
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(False)

def plot_combined_clusters(rounds, aggregated_clusters, federated_clusters, client_clusters_mean, client_clusters_std):
    plt.figure(figsize=(12, 6))
    plt.plot(rounds, aggregated_clusters, marker='o', color='red', label='Aggregated Model Clusters')
    plt.plot(rounds, federated_clusters, marker='x', color='green', label='Federated Model Clusters')
    plt.plot(rounds, client_clusters_mean, marker='o', linestyle=':', color='blue', label='Average Client Clusters')

    # Adding a shaded area to represent the standard deviation interval
    lower_bound = np.array(client_clusters_mean) - np.array(client_clusters_std)
    upper_bound = np.array(client_clusters_mean) + np.array(client_clusters_std)
    plt.fill_between(rounds, lower_bound, upper_bound, color='gray', alpha=0.3, label='Std Dev Interval')

    plt.title('Cluster Evolution over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Number of Clusters')
    plt.legend()
    plt.grid(False)
    plt.show()
    
def plot_metric(rounds, metric_values, metric_name, title, color='blue', marker='o'):
    plt.figure(figsize=(10, 4))
    plt.plot(rounds, metric_values, marker=marker, color=color, label=metric_name)
    plt.title(title)
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(False)
    plt.show()

def plot_client_metrics(rounds, client_metrics, metric_name):
    plt.figure(figsize=(10, 4))
    for client_idx, metrics in enumerate(client_metrics):
        plt.plot(rounds, metrics, marker='o', label=f'Client {client_idx+1} {metric_name}')
    plt.title(f'{metric_name} per Client over Rounds')
    plt.xlabel('Round')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(False)
    plt.show()
    
def extract_metrics(metrics, key):
    # This function will extract the metric for each client across all rounds
    # Assuming 'metrics' is a list of dictionaries, each representing a round
    # And each round's dictionary contains 'client_metrics' which is a list of dictionaries for each client

    # Initialize a list to hold the metric for each client across all rounds
    client_metrics_all_rounds = []

    # Number of clients - assuming the number of clients is consistent across all rounds
    num_clients = len(metrics[0]['client_metrics'])

    for client_idx in range(num_clients):
        # Extract the metric for this client across all rounds
        client_metrics = [round_metric['client_metrics'][client_idx].get(key, None) for round_metric in metrics]
        client_metrics_all_rounds.append(client_metrics)

    return client_metrics_all_rounds

def plot_aggregated_client_metrics(metrics):
    rounds = [metric['round'] for metric in metrics]

    for key in ['f1_score']: #['accuracy', 'precision', 'recall', 'f1_score']
        # Calculate mean and std for each metric
        means, stds = calculate_mean_std(metrics, key)
        ax = plot_with_intervals(rounds, means, stds, 'Clients F1 score')

def plot_metrics(experiments, client_counts, data_config_indices):
    for client_count in client_counts:
        
        for exp_num, metrics in enumerate(experiments):
                rounds = [metric['round'] for metric in metrics]

                # Plot aggregated client metrics (mean ± std dev) for each key
                plot_aggregated_client_metrics(metrics)

                # Retrieve federated model metrics for accuracy, precision, recall, and F1 score
                fed_binary_metrics = {key: [metric.get('federated_model', {}).get('binary', {}).get(key, None) for metric in metrics] for key in ['accuracy', 'precision', 'recall', 'f1_score']}

                # Plotting F1 score, recall, and precision for the federated model on the same plot
                plt.plot(rounds, fed_binary_metrics['f1_score'], marker='o', color='blue', label='F1 Score')
                plt.plot(rounds, fed_binary_metrics['recall'], marker='x', color='green', label='Recall')
                plt.plot(rounds, fed_binary_metrics['precision'], marker='^', color='red', label='Precision')
                plt.title('Federated Model Performance Metrics over Rounds')
                plt.xlabel('Round')
                plt.ylabel('Metrics')
                plt.legend()
                plt.grid(False)
                plt.show()
                
                # Plot ROC AUC for federated model
                fed_roc_aucs = [metric.get('federated_model', {}).get('roc_auc', None) for metric in metrics]
                plot_metric(rounds, fed_roc_aucs, 'ROC AUC', 'Federated Model ROC AUC over Rounds')

                # Plot number of clusters for aggregated and federated models
                aggregated_clusters = [metric['aggregated_model']['clusters'].cpu() for metric in metrics]
                federated_clusters = [metric['federated_model']['clusters'].cpu() for metric in metrics]
                client_clusters = []
                for client_index in range(client_count):
                    client_clusters.append([client_metrics[client_index]['clusters'].cpu() for client_metrics in [metric['client_metrics'] for metric in metrics]])
                client_clusters_mean = np.mean(client_clusters,axis=0)
                client_clusters_std = np.std(client_clusters,axis=0)

                plot_combined_clusters(rounds, aggregated_clusters, federated_clusters, client_clusters_mean, client_clusters_std)


def plot_all_features_upper_triangle(data, labels, model, N_max, num_sigma, colormap='tab10'):
    """
    Function to plot the upper triangle combinations of features against each other, including model's cluster information.

    :param data: Data points (numpy array or torch tensor).
    :param labels: True labels of data points.
    :param model: Model containing cluster information.
    :param N_max: Threshold for the number of data points in a cluster to be visualized.
    :param num_sigma: Number of standard deviations to plot the ellipses.
    :param colormap: Colormap for plotting.
    """
    # Convert data and labels to numpy if they are torch tensors
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    n_features = data.shape[1]

    # Unique color for each label
    unique_labels = np.unique(labels)
    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, len(unique_labels)))
    label_color_dict = dict(zip(unique_labels, label_colors))

    # Create a grid of subplots for the upper triangle
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(i + 1, n_features):
            ax = axes[i, j]

            # Scatter plot of feature i vs feature j
            data_colors = [label_color_dict[label.item()] for label in labels]
            ax.scatter(data[:, j], data[:, i], c=data_colors, alpha=0.5)

            # Plotting ellipses for clusters
            for cluster_idx in range(model.c):
                if model.n[cluster_idx] > N_max:
                    mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                    S = model.S[cluster_idx].cpu().detach().numpy()
                    cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                    cov_submatrix = cov_matrix[[j, i]][:, [j, i]]
                    mu_subvector = mu_val[[j, i]]

                    vals, vecs = np.linalg.eigh(cov_submatrix)
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    factor = num_sigma
                    width, height = factor * np.sqrt(vals)
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none')
                    ax.add_patch(ell)

            ax.set_xlabel(f'Feature {j}')
            ax.set_ylabel(f'Feature {i}')
            ax.grid(True)

            # Hide plots for the lower triangle and diagonal
            axes[j, i].axis('off')
            if i == j:
                ax.axis('off')

    plt.tight_layout()
    plt.show()
    
def plot_all_features_combinations(data, labels, model, N_max, num_sigma, colormap='tab10'):
    """
    Function to plot all combinations of features against each other, including model's cluster information.

    :param data: Data points (numpy array or torch tensor).
    :param labels: True labels of data points.
    :param model: Model containing cluster information.
    :param N_max: Threshold for the number of data points in a cluster to be visualized.
    :param num_sigma: Number of standard deviations to plot the ellipses.
    :param colormap: Colormap for plotting.
    """
    # Convert data and labels to numpy if they are torch tensors
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    n_features = data.shape[1]

    # Unique color for each label
    unique_labels = np.unique(labels)
    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, len(unique_labels)))
    label_color_dict = dict(zip(unique_labels, label_colors))

    # Create a grid of subplots
    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                # Diagonal: plot distribution of feature i
                ax.hist(data[:, i], bins=30, color='gray', alpha=0.7)
            else:
                # Scatter plot of feature i vs feature j
                data_colors = [label_color_dict[label.item()] for label in labels]
                ax.scatter(data[:, j], data[:, i], c=data_colors, alpha=0.5)

                # Plotting ellipses for clusters
                for cluster_idx in range(model.c):
                    if model.n[cluster_idx] > N_max:
                        mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                        S = model.S[cluster_idx].cpu().detach().numpy()
                        cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                        cov_submatrix = cov_matrix[[j, i]][:, [j, i]]
                        mu_subvector = mu_val[[j, i]]

                        vals, vecs = np.linalg.eigh(cov_submatrix)
                        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                        factor = num_sigma
                        width, height = factor * np.sqrt(vals)
                        ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, angle=angle, edgecolor='black', lw=2, facecolor='none')
                        ax.add_patch(ell)

            ax.set_xlabel(f'Feature {j}')
            ax.set_ylabel(f'Feature {i}')
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    
def plot_first_feature(dataset, model, N_max, num_sigma, colormap='tab10'):
    """Function to color data points based on their true labels against the first feature."""

    # Extract data and labels from the TensorDataset
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels
    
        # Assign a unique color to each label based on its index
        # Convert labels to numpy if it's a torch tensor
    
    if len(data.shape) == 1:
        data = data.unsqueeze(0)

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)
    #clear_output(wait=True)

    n_features = data.shape[1]

    unique_labels = model.num_classes

    # Get a colormap instance
    #cmap = cm.get_cmap(colormap)

    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels))

    # Map data points to the color of their label
    label_color_dict = dict(zip(range(unique_labels), label_colors))

    data_colors = [label_color_dict[label.item()] for label in labels]

    # Plotting logic
    num_plots = n_features - 1

    # Square layout calculation
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = rows if rows * (rows - 1) < num_plots else rows - 1

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Check if axes is an instance of AxesSubplot and wrap it in a list if it is
    #if isinstance(axes, plt.Axes):
    #    axes = [axes]

    # Now use axes.ravel() for iterating
    for idx, ax in enumerate(np.array(axes).ravel()[:num_plots]):
        feature_idx = idx + 1
        ax.scatter(data[:, 0], data[:, feature_idx], c=data_colors, alpha=0.5)

        for cluster_idx in range(model.c):  # loop through all clusters
            ellipse_color = label_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]

            # Darken the ellipse color by reducing the RGB values
            # Convert the color to RGBA if it's not already
            ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color) # type: ignore
            dark_factor = 0.8  # Factor to darken the color, where 1 is the original color and 0 is black
            darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, ellipse_color_rgba[1] * dark_factor, ellipse_color_rgba[2] * dark_factor, 1)

            if model.n[cluster_idx] > N_max:
                mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                S = model.S[cluster_idx].cpu().detach().numpy()
                cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                cov_submatrix = cov_matrix[[0, feature_idx]][:, [0, feature_idx]]
                mu_subvector = mu_val[[0, feature_idx]]
                vals, vecs = np.linalg.eigh(cov_submatrix)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                factor = num_sigma
                width, height = factor * np.sqrt(vals)
                ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')

                #ell = Ellipse(mu_subvector, width, height, angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')
                #ell = Ellipse((mu_subvector[0], mu_subvector[1]), width, height, angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')

                ax.add_patch(ell)
                ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=100, marker='x')

        ax.set_title(f"Feature 1 vs Feature {feature_idx + 1}")
        ax.set_xlabel(f"Feature 1")
        ax.set_ylabel(f"Feature {feature_idx + 1}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def fisher_score(data, labels, feature_idx1, feature_idx2):
    unique_labels = np.unique(labels)

    # Ensure feature indices are within the range of data's columns
    if feature_idx1 >= data.shape[1] or feature_idx2 >= data.shape[1]:
        raise ValueError("Feature indices are out of bounds.")

    means = []
    label_data_list = []  # List to store label data for each label
    for label in unique_labels:
        label_data = data[labels == label]  # Filter rows where label matches
        if label_data.size == 0:
            continue  # Skip if no data for this label
        means.append(np.mean(label_data[:, [feature_idx1, feature_idx2]], axis=0))
        label_data_list.append(label_data)  # Append label data to the list

    overall_mean = np.mean(data[:, [feature_idx1, feature_idx2]], axis=0)

    # Fisher Score calculation - ensure scalar values for s_b and s_w
    s_b = sum(len(ld) * np.sum((m - overall_mean)**2) for ld, m in zip(label_data_list, means))
    s_w = sum(np.sum((ld[:, [feature_idx1, feature_idx2]] - m)**2) for ld, m in zip(label_data_list, means))
    
    return s_b / s_w if s_w > 0 else 0


def select_unique_combinations(feature_combinations, fisher_scores, N_combinations):
    """Select top combinations with priority to unique features."""
    sorted_combinations = sorted(feature_combinations, key=lambda x: fisher_scores[x], reverse=True)
    selected_combinations = []
    used_features = set()

    for comb in sorted_combinations:
        if len(selected_combinations) >= N_combinations:
            break
        if comb[0] not in used_features and comb[1] not in used_features:
            selected_combinations.append(comb)
            used_features.update(comb)

    # If we haven't selected enough combinations, fill in the remaining slots
    for comb in sorted_combinations:
        if len(selected_combinations) >= N_combinations:
            break
        if comb not in selected_combinations:
            selected_combinations.append(comb)

    return selected_combinations

def plot_interesting_features(dataset, model, N_max, num_sigma, N_combinations=5, colormap='tab10'):
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    if len(data.shape) == 1:
        data = data.unsqueeze(0)

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)
    #clear_output(wait=True)

    n_features = data.shape[1]
    unique_labels = model.num_classes

    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels))
    label_color_dict = dict(zip(range(unique_labels), label_colors))
    data_colors = [label_color_dict[label.item()] for label in labels]

    # Standardize features for better comparison
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Calculate Fisher Scores for all combinations
    feature_combinations = list(combinations(range(n_features), 2))
    fisher_scores = {comb: fisher_score(data, labels, *comb) for comb in feature_combinations}

    # Select top N combinations with priority to unique features
    top_combinations = select_unique_combinations(feature_combinations, fisher_scores, N_combinations)

    # Plotting logic
    rows = int(np.ceil(np.sqrt(N_combinations)))
    cols = rows if rows * (rows - 1) < N_combinations else rows - 1

    # Set up the subplot grid – all plots in a single row
    fig, axes = plt.subplots(1, N_combinations, figsize=(3 * N_combinations, 3))  # Adjust figure size as needed

    # Flatten the axes array for easy indexing (if N_combinations is 1, wrap axes in a list)
    if N_combinations == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (feature_idx1, feature_idx2) in enumerate(top_combinations):
        ax = axes[idx]  # Use the flattened axes array
        ax.scatter(data[:, feature_idx1], data[:, feature_idx2], c=data_colors, alpha=0.5)


        for cluster_idx in range(model.c):  # loop through all clusters
            if model.n[cluster_idx] > N_max:
                # Get the mean and covariance of the cluster for the current feature pair
                mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                S = model.S[cluster_idx].cpu().detach().numpy()
                cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                cov_submatrix = cov_matrix[[feature_idx1, feature_idx2]][:, [feature_idx1, feature_idx2]]
                mu_subvector = mu_val[[feature_idx1, feature_idx2]]

                # Eigen decomposition for the ellipse orientation
                vals, vecs = np.linalg.eigh(cov_submatrix)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                # Determine the width and height of the ellipse based on eigenvalues
                factor = num_sigma  # Number of standard deviations to plot
                width, height = factor * np.sqrt(vals)

                # Determine the color for the ellipse
                ellipse_color = label_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]
                ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color)
                dark_factor = 0.8  # Factor to darken the color
                darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, 
                                        ellipse_color_rgba[1] * dark_factor, 
                                        ellipse_color_rgba[2] * dark_factor, 1)

                # Create and add the ellipse patch
                ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), 
                                width=width, height=height, 
                                angle=angle, edgecolor=darker_ellipse_color, 
                                lw=2, facecolor='none')
                ax.add_patch(ell)

                # Mark the cluster center
                ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=100, marker='x')


        ax.set_title(f"Feature {feature_idx1 + 1} vs Feature {feature_idx2 + 1}")
        ax.set_xlabel(f"Feature {feature_idx1 + 1}")
        ax.set_ylabel(f"Feature {feature_idx2 + 1}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_first_feature_horizontal(dataset, model, N_max=0, num_sigma=2, title="", colormap='tab10', legend=False, format = '%.1f', data_name = "Class"):
    """Function to color data points based on their true labels against the first feature."""

    
    # Extract data and labels from the TensorDataset
    data, labels = dataset
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data
    labels = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    if len(data.shape) == 1:
        data = data.unsqueeze(0)

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)

    n_features = data.shape[1]

    unique_labels = model.num_classes

    # Check the number of unique labels and assign color accordingly
    if len(torch.unique(model.cluster_labels[:model.c], dim=0)) == 1:
        # All clusters are red if there's only one unique label
        cluster_colors = np.array([[1.0, 0.0, 0.0, 1.0]] * unique_labels)

    else:
        # Use the given colormap for multiple labels
        cluster_colors = plt.cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels))


    # Use the given colormap for multiple labels
        
    label_colors = plt.cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels))
    # Map data points to the color of their label
    label_color_dict = dict(zip(range(unique_labels), label_colors))

    # Map data points to the color of their label
    cluster_color_dict = dict(zip(range(unique_labels), cluster_colors))

    # Plotting logic
    num_plots = n_features - 1
    fig, axes = plt.subplots(1, num_plots, figsize=(2.5 * num_plots, 2.5))

    # If only one plot, wrap axes in a list
    if num_plots == 1:
        axes = [axes]


    for idx, ax in enumerate(axes):
            
        # Track if a label has already been added
        added_labels = {'scatter': set(), 'ellipse': set()}
        
        feature_idx = idx + 1
        # Scatter plot with labels
        for label in range(unique_labels):
            class_data = data[labels == label]
            class_feature_data = class_data[:, [0, feature_idx]]
            if label not in added_labels['scatter']:
                ax.scatter(class_feature_data[:, 0], class_feature_data[:, 1], c=[label_color_dict[label]], alpha=0.5, label= data_name + f'{label+1}')
                added_labels['scatter'].add(label)
            else:
                ax.scatter(class_feature_data[:, 0], class_feature_data[:, 1], c=[label_color_dict[label]], alpha=0.5)

        for cluster_idx in range(model.c):  # loop through all clusters
            ellipse_color = cluster_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]

            # Darken the ellipse color
            ellipse_color_rgba = plt.cm.colors.to_rgba(ellipse_color)  # type: ignore
            dark_factor = 0.8
            darker_ellipse_color = (ellipse_color_rgba[0] * dark_factor, 
                                    ellipse_color_rgba[1] * dark_factor, ellipse_color_rgba[2] * dark_factor, 1)

            if model.n[cluster_idx] > N_max:

                mu_val = model.mu[cluster_idx].cpu().detach().numpy()
                S = model.S[cluster_idx].cpu().detach().numpy()
                cov_matrix = (S / model.n[cluster_idx].cpu().detach().numpy())
                cov_submatrix = cov_matrix[[0, feature_idx]][:, [0, feature_idx]]
                mu_subvector = mu_val[[0, feature_idx]]
                vals, vecs = np.linalg.eigh(cov_submatrix)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                factor = num_sigma
                width, height = factor * np.sqrt(vals)

                if cluster_idx == 0:
                    ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=10, marker='o', label=f'Centers')
                else:
                    ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=10, marker='o')

                if torch.where(model.cluster_labels[cluster_idx] == 1)[0].item() not in added_labels['ellipse']:
                    if len(torch.unique(model.cluster_labels[:model.c], dim=0)) == 1:
                        cluster_name = "Clusters"
                    else:
                        cluster_name = f'Cluster {torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()+1}'
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, 
                                  angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none', label=cluster_name)

                    added_labels['ellipse'].add(torch.where(model.cluster_labels[cluster_idx] == 1)[0].item())
                else:
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height,
                                  angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')
                   
                                
                ax.add_patch(ell)


        ax.set_xlabel(f"Feature 1", fontsize=8)
        ax.set_ylabel(f"Feature {feature_idx + 1}", fontsize=8)
        ax.grid(False)

    for ax in axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter(format))
        ax.yaxis.set_major_formatter(FormatStrFormatter(format))

    # Adjust the subplots to shift them to the right

    fig.text(-0.02, 0.55, title, va='center', rotation='vertical', fontsize=12)
    
    if legend:
            # Adding legend outside the plot
            # Adding legend to the last axis
        axes[-1].legend(loc='best', fontsize = 6)
        
    plt.tight_layout()
    plt.show()
 
    return fig

def save_figure(figure, filename, format='pdf'):
    """
    Saves the last generated matplotlib figure to the specified file format.

    :param figure: The matplotlib figure object to save.
    :param filename: The name of the file where the figure will be saved.
    :param format: The format of the file ('pdf', 'svg', etc.).
    """

    figure.savefig(filename, format=format, bbox_inches='tight')
    print(f"Figure saved as {filename} in {format} format.")
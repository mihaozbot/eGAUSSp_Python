import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
from IPython.display import clear_output
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
from IPython.display import clear_output
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

   
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
    clear_output(wait=True)

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

def plot_first_feature_horizontal(dataset, model, N_max=0, num_sigma=2, title="", colormap='tab10', legend=False):
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

    # Get a colormap instance
    #cmap = cm.get_cmap(colormap)

    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, unique_labels))

    # Map data points to the color of their label
    label_color_dict = dict(zip(range(unique_labels), label_colors))
    
    # Generate colors for each unique label
    #label_colors = [cmap(i/unique_labels) for i in range(unique_labels)]

    # Map data points to the color of their label
    #label_color_dict = {label: color for label, color in zip(range(unique_labels), label_colors)}

    # Plotting logic
    num_plots = n_features - 1

    # Create a horizontal line of figures
    fig, axes = plt.subplots(1, num_plots, figsize=(2.5* num_plots, 2.5))

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
                ax.scatter(class_feature_data[:, 0], class_feature_data[:, 1], c=[label_color_dict[label]], alpha=0.5, label=f'Class {label+1}')
                added_labels['scatter'].add(label)
            else:
                ax.scatter(class_feature_data[:, 0], class_feature_data[:, 1], c=[label_color_dict[label]], alpha=0.5)

        for cluster_idx in range(model.c):  # loop through all clusters
            ellipse_color = label_color_dict[torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()]

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
                if torch.where(model.cluster_labels[cluster_idx] == 1)[0].item() not in added_labels['ellipse']:
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height, 
                                  angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none', label=f'Cluster {torch.where(model.cluster_labels[cluster_idx] == 1)[0].item()+1}')
                    added_labels['ellipse'].add(torch.where(model.cluster_labels[cluster_idx] == 1)[0].item())
                else:
                    ell = Ellipse(xy=(mu_subvector[0], mu_subvector[1]), width=width, height=height,
                                  angle=angle, edgecolor=darker_ellipse_color, lw=2, facecolor='none')

                                
                ax.add_patch(ell)
                ax.scatter(mu_subvector[0], mu_subvector[1], color='black', s=10, marker='o')

        ax.set_xlabel(f"Feature 1", fontsize=10)
        ax.set_ylabel(f"Feature {feature_idx + 1}", fontsize=10)
        ax.grid(False)

    for ax in axes:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # Adjust the subplots to shift them to the right

    fig.text(-0.02, 0.5, title, va='center', rotation='vertical', fontsize=12)
    
    if legend:
            # Adding legend outside the plot
            # Adding legend to the last axis
        axes[-1].legend(loc='best')
        
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
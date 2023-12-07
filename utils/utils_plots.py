import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.patches import Ellipse
from IPython.display import clear_output

def plot_first_feature(data, labels, model, N_max, num_sigma, colormap='tab10'):
    """Function to color data points based on their true labels against the first feature."""
    
        # Assign a unique color to each label based on its index
        # Convert labels to numpy if it's a torch tensor
    data = data.cpu().detach().numpy() if isinstance(data, torch.Tensor) else data

    if len(data.shape) == 1:
        data = data.unsqueeze(0)

    if len(labels.shape) == 0:
        labels = labels.unsqueeze(0)
    clear_output(wait=True)

    n_features = data.shape[1]

    # Assign a unique color to each label based on its index
        # Convert labels to numpy if it's a torch tensor
    labels_np = labels.cpu().detach().numpy() if isinstance(labels, torch.Tensor) else labels

    # Convert model.cluster_labels to numpy if it's a torch tensor
    model_cluster_labels_np = model.cluster_labels[0:model.c].clone().cpu().numpy() if isinstance(model.cluster_labels[0:model.c], torch.Tensor) else model.cluster_labels[0:model.c]

    # Concatenate both arrays and find unique labels
    combined_labels = np.concatenate((labels_np, model_cluster_labels_np))
    unique_labels = np.unique(combined_labels)

    label_colors = cm.get_cmap(colormap)(np.linspace(0, 0.5, len(unique_labels)))

    # Map data points to the color of their label
    label_color_dict = dict(zip(unique_labels, label_colors))
    data_colors = [label_color_dict[label.item()] for label in labels]

    # Plotting logic
    num_plots = n_features - 1

    # Square layout calculation
    rows = int(np.ceil(np.sqrt(num_plots)))
    cols = rows if rows * (rows - 1) < num_plots else rows - 1

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

    # Check if axes is an instance of AxesSubplot and wrap it in a list if it is
    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Now use axes.ravel() for iterating
    for idx, ax in enumerate(np.array(axes).ravel()[:num_plots]):
        feature_idx = idx + 1
        ax.scatter(data[:, 0], data[:, feature_idx], c=data_colors, alpha=0.5)

        for cluster_idx in range(model.c):  # loop through all clusters
            ellipse_color = label_color_dict[model.cluster_labels[cluster_idx].item()]

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

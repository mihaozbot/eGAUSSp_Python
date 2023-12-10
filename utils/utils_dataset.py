from sklearn.model_selection import train_test_split
import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from sklearn.utils import shuffle

def non_iid_data(X, y, num_clients):
    """
    Redistributes class 0 data from the first client to the second client,
    and class 1 data from the second client to the third client.

    :param X: List of feature arrays split among clients.
    :param y: List of label arrays split among clients.
    :param num_clients: Number of clients.
    """
    # Move class 0 data from first client to second client
    if num_clients > 1:
        class_0_indices = y[0] == 0
        if any(class_0_indices):
            X[1] = np.concatenate((X[1], X[0][class_0_indices]))
            y[1] = np.concatenate((y[1], y[0][class_0_indices]))
            X[0] = X[0][~class_0_indices]
            y[0] = y[0][~class_0_indices]

    # Move class 1 data from second client to third client
    if num_clients > 2:
        class_1_indices = y[1] == 1
        if any(class_1_indices):
            X[2] = np.concatenate((X[2], X[1][class_1_indices]))
            y[2] = np.concatenate((y[2], y[1][class_1_indices]))
            X[1] = X[1][~class_1_indices]
            y[1] = y[1][~class_1_indices]

    return X, y


def prepare_dataset(X, y, num_clients):
    """
    Prepares a dataset for federated learning under a non-IID setting. The dataset is first split 
    into training and testing sets. The training set is then further split among the specified number 
    of clients in a non-IID fashion, where different classes are unevenly distributed among clients.

    :param data: Panda dataframe
    :param num_clients: The number of clients to distribute the data among.
    :return: A tuple containing the training data for each client, the testing data, and the entire dataset.
    """

    # Split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Split the training data among clients in a non-IID fashion
    train_data = []
    for i in range(num_clients):
        indices = np.random.choice(len(X_train), len(X_train) // num_clients, replace=False)
        X_client = X_train[indices]
        y_client = y_train[indices]

        # Remove the selected indices from the training data
        X_train = np.delete(X_train, indices, axis=0)
        y_train = np.delete(y_train, indices, axis=0)

        # Convert to PyTorch tensors
        train_data.append((torch.tensor(X_client, dtype=torch.float32), 
                           torch.tensor(y_client, dtype=torch.int64)))

    # Convert X_test and y_test to tensors and pack together
    test_data = (torch.tensor(X_test, dtype=torch.float32), 
                 torch.tensor(y_test, dtype=torch.int64))

    # Convert the entire dataset to tensors and pack together
    all_data = (torch.tensor(X, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.int64))

    return train_data, test_data, all_data

def prepare_non_iid_dataset(X, y, num_clients):
    
    """
    Prepares a dataset for federated learning under a non-IID setting. The dataset is first split 
    into training and testing sets. The training set is then further split among the specified number 
    of clients in a non-IID fashion, where different classes are unevenly distributed among clients.

    :param data: Panda dataframe
    :param num_clients: The number of clients to distribute the data among.
    :return: A tuple containing the training data for each client, the testing data, and the entire dataset.
    """

    # Split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Split the training data among clients
    X_train_split = np.array_split(X_train, num_clients)
    y_train_split = np.array_split(y_train, num_clients)

    # Move class 0 data from first client to second client
    X_train_split, y_train_split = non_iid_data(X_train_split, y_train_split, num_clients)
    
    # Convert the training data to PyTorch tensors and distribute to clients
    train_data = [(torch.tensor(X_train_split[i], dtype=torch.float32), 
                    torch.tensor(y_train_split[i], dtype=torch.int64)) 
                   for i in range(num_clients)]

    # Convert X_test and y_test to tensors and pack together
    test_data = (torch.tensor(X_test, dtype=torch.float32), 
                 torch.tensor(y_test, dtype=torch.int64))

    # Convert the entire dataset to tensors and pack together
    all_data = (torch.tensor(X, dtype=torch.float32), 
                torch.tensor(y, dtype=torch.int64))

    return train_data, test_data, all_data

# Display the number of samples per class for each client and the test set
def display_dataset_split(client_data, test_dataset):

    # Extract y_test from the test dataset
    _, y_test = test_dataset

    for i, (_, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client.numpy(), return_counts=True)
        print(f"Client {i + 1}: {dict(zip(unique, counts))}")
    
    unique_test, counts_test = np.unique(y_test.numpy(), return_counts=True)
    print(f"Test Set: {dict(zip(unique_test, counts_test))}")

    # Combine all client data and test data
    combined_y = np.concatenate([y_client.numpy() for _, y_client in client_data] + [y_test.numpy()])

    # Count the number of samples per class
    unique, counts = np.unique(combined_y, return_counts=True)
    combined_counts = dict(zip(unique, counts))

    # Display the counts
    print("\nCombined Number of Samples per Class:")
    for class_label, count in combined_counts.items():
        print(f"Class {class_label}: {count} samples")
    
    # Display the total count
    total_samples = sum(combined_counts.values())
    print(f"\nTotal Number of Samples Across All Datasets: {total_samples}")

def plot_dataset_split(client_data, test_dataset):
    
        # Extract y_test from the test dataset
    _, y_test = test_dataset

    num_clients = len(client_data)
    classes, _ = np.unique(y_test.numpy(), return_counts=True)
    num_classes = len(classes)
    
    # Initialize counts
    client_counts = np.zeros((num_clients, num_classes))
    test_counts = np.zeros(num_classes)

    # Count the samples per class for each client
    for i, (_, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client.numpy(), return_counts=True)
        for class_label, count in zip(unique, counts):
            client_counts[i, class_label] = count

    # Count the samples per class for the test set
    unique, counts = np.unique(y_test.numpy(), return_counts=True)
    for class_label, count in zip(unique, counts):
        test_counts[class_label] = count

    # Create an array for the x-axis labels
    labels = [f'Client {i+1}' for i in range(num_clients)] + ['Test Set']

    # Plotting
    bar_width = 0.3
    opacity = 0.8

    # Plot for each class
    for class_label in range(num_classes):
        plt.bar(np.arange(num_clients + 1) + bar_width * class_label, 
                np.append(client_counts[:, class_label], test_counts[class_label]),
                bar_width, alpha=opacity, label=f'Class {class_label}')

    plt.xlabel('Dataset')
    plt.ylabel('Number of Samples')
    #plt.title('Number of Samples per Dataset')
    plt.xticks(np.arange(num_clients + 1) + bar_width / 2, labels)
    plt.legend()

    plt.tight_layout()
    plt.show()

import pandas as pd


def balance_dataset(data, class_column='Class', random_state=None):
    """
    Balances a dataset by undersampling the majority class to match the size of the minority class.

    :param data: pandas DataFrame containing the dataset.
    :param class_column: Name of the column containing class labels.
    :param random_state: Optional random state for reproducibility. If None, each call will be random.
    :return: A balanced pandas DataFrame.
    """
    # Count the number of instances for each class
    total_class_1 = sum(data[class_column] == 1)
    total_class_0 = len(data) - total_class_1

    # Determine the size of the smaller class
    minority_size = min(total_class_1, total_class_0)

    # Separate the dataset into two classes
    class_1_data = data[data[class_column] == 1]
    class_0_data = data[data[class_column] == 0]

    # Randomly sample from the larger class to match the size of the smaller class
    class_1_data_balanced = class_1_data.sample(n=minority_size, random_state=random_state)
    class_0_data_balanced = class_0_data.sample(n=minority_size, random_state=random_state)

    # Combine the balanced datasets
    balanced_data = pd.concat([class_1_data_balanced, class_0_data_balanced])

    # Shuffle the dataset (optional but recommended)
    return balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)

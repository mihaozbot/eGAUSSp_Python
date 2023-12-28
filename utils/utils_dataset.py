from sklearn.model_selection import train_test_split
import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

#from imblearn.over_sampling import SMOTE

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


def prepare_dataset(X, y, num_clients, balance=None):
    """
    Prepares a dataset for federated learning under a non-IID setting, ensuring each client has 
    approximately the same number of samples from each class.

    :param X: Features as a numpy array.
    :param y: Labels as a numpy array.
    :param num_clients: The number of clients to distribute the data among.
    :param balance: Balancing technique, if any.
    :return: Training data for each client, the testing data, and the entire dataset.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # If balancing is required, apply balance_dataset
    if balance:
        X_train, y_train = balance_dataset(X_train, y_train, balance)

    # Split the training data by class
    unique_classes = np.unique(y_train)
    class_splits = {cls: [] for cls in unique_classes}
    for cls in unique_classes:
        class_indices = np.where(y_train == cls)[0]
        class_splits[cls] = np.array_split(class_indices, num_clients)

    # Distribute class-wise splits among clients
    X_train_split = [[] for _ in range(num_clients)]
    y_train_split = [[] for _ in range(num_clients)]
    for cls in unique_classes:
        for i in range(num_clients):
            X_train_split[i].extend(X_train[class_splits[cls][i]])
            y_train_split[i].extend(y_train[class_splits[cls][i]])

    # Shuffle and convert the training data to PyTorch tensors for each client
    train_data_clients = []
    for i in range(num_clients):
        X_split, y_split = shuffle(np.array(X_train_split[i]), np.array(y_train_split[i]))
        train_data_clients.append((torch.tensor(X_split, dtype=torch.float32), torch.tensor(y_split, dtype=torch.int64)))

    # Convert testing data to PyTorch tensors
    test_data = (torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    # Convert the entire dataset to tensors
    all_data = (torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))

    return train_data_clients, test_data, all_data

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
    _, y_test = test_dataset

    num_clients = len(client_data)
    classes, _ = np.unique(y_test.numpy(), return_counts=True)
    num_classes = len(classes)

    client_counts = np.zeros((num_clients, num_classes))
    test_counts = np.zeros(num_classes)

    for i, (_, y_client) in enumerate(client_data):
        unique, counts = np.unique(y_client.numpy(), return_counts=True)
        for class_label, count in zip(unique, counts):
            client_counts[i, class_label] = count

    unique, counts = np.unique(y_test.numpy(), return_counts=True)
    for class_label, count in zip(unique, counts):
        test_counts[class_label] = count

    labels = [f'Client {i+1}' for i in range(num_clients)] + ['Test Set']

    fig_width = 5
    fig_height = 3

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    bar_width = 0.5
    opacity = 0.8

    bottom = np.zeros(num_clients + 1)

    for class_label in range(num_classes):
        bar_values = np.append(client_counts[:, class_label], test_counts[class_label])
        bars = ax.bar(np.arange(num_clients + 1), bar_values, bar_width, alpha=opacity, label=f'Class {class_label+1}', bottom=bottom)

        # Label each bar segment
        for bar_index, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:  # Only label bars with a non-zero height
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, bottom[bar_index] + height / 2),
                            xytext=(0, 0),  # Center the text
                            textcoords="offset points",
                            ha='center', va='center')

        bottom += bar_values

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Number of Samples')
    ax.set_xticks(np.arange(num_clients + 1))
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    return fig


'''
def balance_dataset(data,  proportion, class_column='Class', random_state=None):
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
    class_0_data_balanced = class_0_data.sample(n=proportion*minority_size, random_state=random_state)

    # Combine the balanced datasets
    balanced_data = pd.concat([class_1_data_balanced, class_0_data_balanced])

    # Shuffle the dataset (optional but recommended)
    return balanced_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
'''

def balance_dataset(X, y, technique='random', target_sample_count=3000):
    """
    Balances a dataset by undersampling or oversampling using different techniques.

    :param X: Feature data as a numpy array.
    :param y: Label data as a numpy array.
    :param technique: Technique for balancing ('random', 'tomek', 'centroids', 'nearmiss', 'enn', 'smote', 'smote_random').
    :param target_sample_count: Target number of samples for each class in 'smote_random' technique.
    :return: Balanced feature and label arrays.
    """
    # Select the appropriate sampler based on the technique
    if technique == 'random':
        sampler = RandomUnderSampler(random_state=None)
    elif technique == 'tomek':
        sampler = TomekLinks()
    elif technique == 'centroids':
        sampler = ClusterCentroids(random_state=None)
    elif technique == 'nearmiss':
        sampler = NearMiss(version=1)
    elif technique == 'enn':
        sampler = EditedNearestNeighbours()
    elif technique == 'smote':
        sampler = SMOTE(random_state=None)
    elif technique == 'smote_random':
        unique_classes = np.unique(y)
        class_sample_counts = {label: np.sum(y == label) for label in unique_classes}

        # Oversample the minority classes
        oversample_strategy = {label: target_sample_count for label, count in class_sample_counts.items() if count < target_sample_count}
        if oversample_strategy:
            oversampler = SMOTE(sampling_strategy= oversample_strategy, random_state=None)
            X, y = oversampler.fit_resample(X, y)

        # Undersample the majority classes
        undersample_strategy = {label: target_sample_count for label in unique_classes}
        sampler = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=None)

    else:
        raise ValueError("Unknown technique: choose from 'random', 'tomek', 'centroids', 'nearmiss', 'enn', 'smote', 'smote_random'")

    # Apply the selected sampler to resample the dataset
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    return X_resampled, y_resampled

def prepare_k_fold_non_iid_dataset(X, y, train_index, test_index, num_clients):
    """
    Prepares a dataset for federated learning under a non-IID setting. The dataset is split into training 
    and testing sets based on provided indices. The training set is then further split among the specified 
    number of clients in a non-IID fashion, where different classes are unevenly distributed among clients.

    :param X: Features in the dataset
    :param y: Labels in the dataset
    :param train_index: Indices for the training set
    :param test_index: Indices for the testing set
    :param num_clients: The number of clients to distribute the data among
    :return: A tuple containing the training data for each client, the testing data, and the entire dataset
    """

    # Shuffle the training indices for randomness
    train_index = shuffle(train_index)
    test_index = shuffle(test_index)

    # Create the training and testing sets using provided indices
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]

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


def prepare_k_fold_federated_dataset(X, y, train_index, test_index, num_clients, percentage):
    """
    Prepares a dataset for federated learning by distributing only a specified percentage of the training data 
    among the clients. The dataset is split into training and testing sets based on provided indices. 
    Only a portion of the training set, determined by the percentage, is further split among the specified 
    number of clients.

    :param X: Features in the dataset
    :param y: Labels in the dataset
    :param train_index: Indices for the training set
    :param test_index: Indices for the testing set
    :param num_clients: The number of clients to distribute the data among
    :param percentage: The percentage of the training data to be used
    :return: A tuple containing the training data for each client, the testing data, and the entire dataset
    """

    # Shuffle the training indices for randomness
    train_index = shuffle(train_index)
    test_index = shuffle(test_index)

    # Determine the subset of the training indices to use
    num_train_samples = int(percentage*num_clients*len(train_index))
    train_index_subset = train_index[:num_train_samples]

    # Create the training and testing sets using the subset of indices
    X_train_subset, y_train_subset = X[train_index_subset], y[train_index_subset]
    X_test, y_test = X[test_index], y[test_index]

    # Distribute the training data among clients
    X_train_split = np.array_split(X_train_subset, num_clients)
    y_train_split = np.array_split(y_train_subset, num_clients)

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
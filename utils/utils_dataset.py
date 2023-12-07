from sklearn.model_selection import train_test_split
import torch
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import torch
  
def prepare_non_iid_dataset(X ,y , num_clients):

    # Split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Split the training data among clients
    X_train_split = np.array_split(X_train, num_clients)
    y_train_split = np.array_split(y_train, num_clients)

    # Move class 0 data from first client to second client
    class_0_indices = y_train_split[0] == 0
    if any(class_0_indices) and num_clients > 1:
        X_train_split[1] = np.concatenate((X_train_split[1], X_train_split[0][class_0_indices]))
        y_train_split[1] = np.concatenate((y_train_split[1], y_train_split[0][class_0_indices]))
        X_train_split[0] = X_train_split[0][~class_0_indices]
        y_train_split[0] = y_train_split[0][~class_0_indices]

    # Move class 1 data from second client to third client
    class_1_indices = y_train_split[1] == 1
    if any(class_1_indices) and num_clients > 2:
        X_train_split[2] = np.concatenate((X_train_split[2], X_train_split[1][class_1_indices]))
        y_train_split[2] = np.concatenate((y_train_split[2], y_train_split[1][class_1_indices]))
        X_train_split[1] = X_train_split[1][~class_1_indices]
        y_train_split[1] = y_train_split[1][~class_1_indices]

    # Convert the data to PyTorch tensors and distribute to clients
    client_data = []
    for i in range(num_clients):
        client_X_tensor = torch.tensor(X_train_split[i], dtype=torch.float32)
        client_y_tensor = torch.tensor(y_train_split[i], dtype=torch.int64)
        client_data.append((client_X_tensor, client_y_tensor))

    # Convert X_test and y_test to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    # Convert the entire dataset to tensors
    X_all_tensor = torch.tensor(X, dtype=torch.float32)
    y_all_tensor = torch.tensor(y, dtype=torch.int64)

    return client_data, X_test_tensor, y_test_tensor, X_all_tensor, y_all_tensor

# Display the number of samples per class for each client and the test set
def display_dataset_split(client_data, y_test):
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

def plot_dataset_split(client_data, y_test):
    num_clients = len(client_data)
    num_classes = 3 # Assuming 3 classes in the Iris dataset

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
    fig, ax = plt.subplots()
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


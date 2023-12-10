import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import cycle
import threading
from torch.utils.data import DataLoader


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
    
    print(f"Batch done. Number of clusters = {torch.sum(model.n>model.N_max)}")

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

    print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

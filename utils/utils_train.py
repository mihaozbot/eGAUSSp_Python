import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import cycle


def train_supervised(model, client_data):
    
    #Toggle training mode
    model.toggle_evolving(True)
    model.train()
    
    sample_data, sample_labels = client_data
    
    # Training loop
    for z, label in zip(sample_data, sample_labels):
        model.forward(z, label)  # Train the model
    
def train_unsupervised(model, client_data):
    
    #Toggle training mode
    model.toggle_evolving(True)
    model.train()
    
    data, labels = client_data
    
    # Training loop
    for z in data:
        model.forward(z, torch.tensor(-1, dtype=torch.int64))  # Train the model

        
def test_model(model, X_test, y_test):
    
    #Toggle evaluation mode
    model.toggle_evolving(False)
    model.eval()

    pred_max = []
    for z in X_test:
        output = model.forward(z, -1)  # Forward pass
        pred_max.append(output.argmax())  # Assuming pred is a tensor of class scores

    pred_max = torch.tensor(pred_max)

    # Evaluation metrics with zero_division parameter
    accuracy = accuracy_score(y_test, pred_max)
    precision = precision_score(y_test, pred_max, average='weighted', zero_division=1)
    recall = recall_score(y_test, pred_max, average='weighted', zero_division=1)
    f1 = f1_score(y_test, pred_max, average='weighted', zero_division=1)

    print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

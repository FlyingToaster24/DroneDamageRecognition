import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
import os

def resize(images, size):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (size, size))
        resized_images.append(resized_image)
    return resized_images

def load_images(image_folder):
    image_files = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".tif"):
            image_files.append(os.path.join(image_folder, filename))

    images = []

    for image_file in image_files:
        image = cv2.imread(image_file)
        images.append(image)

    return images


def load_folders(main_folder, sub_folders):
    images = []
    for sub_folder in sub_folders:
        images.extend(load_images(main_folder + sub_folder))
    return images


def augment(images):
    augmented_images = []
    for image in images:
        augmented_images.append(image)
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
        augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))
    return augmented_images

def train_model(X, y, model_name, batch_size=256, num_epochs=50, dropout_ratio=0.2, patience=10, learning_rate=1e-3):
    X = np.array(X, dtype=np.uint8)
    X = X / 255.0  # scale the input data to [0, 1]
    X = np.transpose(X, (0, 3, 1, 2))  # transpose to match PyTorch format

    # define the device to use (either CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert data to PyTorch tensors
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).long().to(device)

    # Split the dataset into training and testing sets
    split = int(0.8 * len(X))
    X_t, y_t = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    X_t, y_t = X_t.to(device), y_t.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Define the CNN model
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(dropout_ratio),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(dropout_ratio),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    model.to(device)
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define the number of epochs and early stopping parameters
    best_val_loss = float('inf')
    best_model_state = None
    num_epochs_since_improvement = 0

    # Create PyTorch data loaders
    train_dataset = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # Train the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            loss = loss_fn(y_pred, batch_y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_X, batch_y in test_loader:
                y_val_pred = model(batch_X)
                val_loss += loss_fn(y_val_pred, batch_y.squeeze()).item()
            val_loss /= len(test_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            num_epochs_since_improvement = 0
        else:
            num_epochs_since_improvement += 1
            if num_epochs_since_improvement >= patience:
                print(f'Early stopping after {epoch + 1} epochs')
                break

        print(f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')


    # Load the best model state and test it on the test set
    model.load_state_dict(best_model_state)
    model.eval()
    model.to('cpu')

    torch.save(model, 'model.pt')

    return model


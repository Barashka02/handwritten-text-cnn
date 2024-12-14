# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.cnn import CNN  
import os

def main():
    # =========================
    # 1. Define Hyperparameters
    # =========================
    n_epochs = 10
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.001
    momentum = 0.9
    log_interval = 100

    # Seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)

    # =========================
    # 2. Prepare Data Loaders
    # =========================
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Ensure the 'data' directory exists
    os.makedirs('data', exist_ok=True)

    # Load the EMNIST Letters dataset
    train_dataset = torchvision.datasets.EMNIST(
        root='data',
        split='letters',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.EMNIST(
        root='data',
        split='letters',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # 3. Initialize Model and Optimizer
    # =========================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Instantiate the CNN
    model = CNN(num_classes=26).to(device)

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # =========================
    # 4. Define Training and Testing Functions
    # =========================
    def train(epoch):
        """
        Trains the model for one epoch.
        """
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), (target - 1).to(device)  # Shift labels from 1-26 to 0-25

            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = nn.NLLLoss()(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test():
        """
        Evaluates the model on the test dataset.
        """
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():  # Disable gradient computation
            for data, target in test_loader:
                data, target = data.to(device), (target - 1).to(device)  # Shift labels from 1-26 to 0-25
                output = model(data)  # Forward pass
                test_loss += nn.NLLLoss(reduction='sum')(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)  # Average loss
        accuracy = 100. * correct / len(test_loader.dataset)  # Accuracy percentage

        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    # =========================
    # 5. Run Training and Testing
    # =========================
    print("Starting training...")
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    # =========================
    # 6. Save the Trained Model
    # =========================
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/emnist_letters_cnn.pth')
    print("Model saved to 'models/emnist_letters_cnn.pth'")

if __name__ == "__main__":
    main()

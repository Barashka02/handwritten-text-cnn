# models/cnn.py

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    A Convolutional Neural Network for EMNIST letter classification.
    Architecture:
    - Conv Layer 1: 1 input channel, 32 output channels, 3x3 kernel
    - Conv Layer 2: 32 input channels, 64 output channels, 3x3 kernel
    - Max Pooling: 2x2
    - Fully Connected Layer 1: 3136 input features, 128 output features
    - Fully Connected Layer 2: 128 input features, 26 output features (A-Z)
    """
    def __init__(self, num_classes=26):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Apply first conv layer, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv1(x)))  # (N, 32, 14, 14)
        # Apply second conv layer, ReLU activation, and pooling
        x = self.pool(F.relu(self.conv2(x)))  # (N, 64, 7, 7)
        # Flatten the tensor
        x = x.view(-1, 64 * 7 * 7)
        # Apply first fully connected layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply second fully connected layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

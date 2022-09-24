import torch
import torch.nn as nn


class CatDogClassifier(nn.Module):
    """Class for the classifier"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # conv2d(in_channels, out_channels, kernel_size, stride=1,padding(?))
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)

        self.pool = nn.MaxPool2d(2, 2) # kernel_size=2, stride=2
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear( 14*14* 128, 256) # 14x14 is the output dimension for each channel after last conv+max_pool layer, 128 is nr of channels
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
      
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
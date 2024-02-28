import torch.nn as nn
import torch.nn.functional as F
import torch

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Height = (Input Height + Padding height (top and bottom) - Kernel Height) / Stride Height + 1
        # Width = (Input Width + Padding width (left and right) - Kernel Width) / Stride Width + 1
        
        # [BatchSize Channels Width Height]
        # Input [64 3 32 32]
        # Output [BatchSize NUM_KERNELS 32+0-3/1+1 32+0-3/1+1] = [64 32 30/1 30/1] = [64 32 30 30]
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=0),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=0),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(1, 3, 32, 32)).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out
    

class OLDCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Height = (Input Height + Padding height (top and bottom) - Kernel Height) / Stride Height + 1
        # Width = (Input Width + Padding width (left and right) - Kernel Width) / Stride Width + 1
        
        # [BatchSize Channels Width Height]
        # Input [64 3 32 32]
        # Output [BatchSize NUM_KERNELS 32+0-3/1+1 32+0-3/1+1] = [64 32 30/1 30/1] = [64 32 30 30]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=0)
        self.fc1 = nn.Linear(256 * 16 * 16, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        #x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout()
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x
    

import torch.nn as nn
import torch.nn.functional as F


class SimplePose(nn.Module):
    
    def __init__(self):
        super(SimplePose, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.fc1 = nn.Linear(in_features=256*16*16, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=28)
    
    def forward(self, x):
        # x -> [-1, 3, 128, 128]
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # x -> [-1, 32, 64, 64]
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # x -> [-1, 32, 32, 32]
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        # x -> [-1, 64, 16, 16]
        # x -> [-1, 64, 16, 16]

        x = x.view(-1, 256*16*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
import torch
import torch.nn as nn

class StaticGestureMLP(nn.Module):
    def __init__(self, input_size=63, hidden1=128, hidden2=64, num_classes=26):
        super(StaticGestureMLP, self). __init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.network(x)

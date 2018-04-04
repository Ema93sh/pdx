import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class QModel(nn.Module):
    """Model for DQN"""
    def __init__(self, input_size, action_size):
        super(QModel, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.fc1 = nn.Linear(input_size, 250)
        self.output = nn.Linear(250, action_size)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.output(x)
        return x

    def clone(self):
        model = QModel(self.input_size, self.action_size)
        model.load_state_dict(self.state_dict())
        return model

    def clone_from(self, model):
        self.load_state_dict(model.state_dict())

class QModelCNN(nn.Module):
    """Model for DQN"""
    def __init__(self, input_size, action_size):
        super(QModelCNN, self).__init__()
        self.input_size = input_size
        self.action_size = action_size

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))


        self.output = nn.Linear(288, action_size)

    def forward(self, input):
        x = self.layer1(input)
        out = self.layer2(x)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out

    def clone(self):
        model = QModelCNN(self.input_size, self.action_size)
        model.load_state_dict(self.state_dict())
        return model

    def clone_from(self, model):
        self.load_state_dict(model.state_dict())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DQNModel(nn.Module):
    """Model for DQN"""
    def __init__(self, input_size, action_size):
        super(DQNModel, self).__init__()
        self.input_size = input_size
        self.action_size = action_size
        self.fc1 = nn.Linear(input_size, 50)
        self.output = nn.Linear(50, action_size)

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.output(x)
        return x

    def clone(self):
        model = DQNModel(self.input_size, self.action_size)
        model.load_state_dict(self.state_dict())
        return model

    def clone_from(self, model):
        self.load_state_dict(model.state_dict())

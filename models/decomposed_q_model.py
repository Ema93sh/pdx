import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DecomposedQModel(nn.Module):
    """ Decompsed Q values """
    def __init__(self, reward_types, input_size, action_size):
        super(DecomposedQModel, self).__init__()
        self.reward_types = reward_types
        self.input_size = input_size
        self.action_size = action_size
        for reward_type in range(self.reward_types):
            network = nn.Sequential(nn.Linear(input_size, 110),
                                  nn.ReLU(),
                                  nn.Linear(110, action_size))
            setattr(self, 'q_{}_network'.format(reward_type), network)

    def forward(self, input):
        q_values = []
        for reward_type in range(self.reward_types):
            q_value = getattr(self, 'q_{}_network'.format(reward_type))(input)
            q_values.append(q_value)

        q_values = torch.stack(q_values)
        combined_q_value = torch.sum(q_values, 0)
        return combined_q_value, q_values

    def clone(self):
        model = DecomposedQModel(self.reward_types, self.input_size, self.action_size)
        model.load_state_dict(self.state_dict())
        return model

    def clone_from(self, model):
        self.load_state_dict(model.state_dict())


class DecomposedModel(object):
    """ Decompsed Q values """
    def __init__(self, reward_types, input_size, action_size):
        super(DecomposedModel, self).__init__()
        self.networks = []
        for reward_type in range(reward_types):
            network = nn.Sequential(nn.Linear(input_size, 250),
                                  nn.ReLU(),
                                  nn.Linear(250, action_size))

            self.networks.append(network)

    def forward(self, input):
        q_values = []
        for network in self.networks:
            q_value = network(input)
            q_values.append(q_value)

        combined_q_value = #TODO
        return combined_q_value, q_values

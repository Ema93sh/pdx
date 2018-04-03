import numpy as np
import torch
import copy
from torch.autograd import Variable

class Explanation(object):
    """Explanation class related to generating explanations"""

    def __init__(self):
        super(Explanation, self).__init__()
        # TODO

    def get_pdx(self, q_values, selected_action, target_actions):
        """predicted delta explanations for the selected action in comparision with the target action.

        :param list q_values: decomposed q-values
        :param int selected_action:
        :param list target_actions:

        >>> Explanation().get_pdx(np.array([[1, 1, 1, 1], [4, 3, 2, 1]]), 0, [1, 2])
        ([[0, 0], [1, 2]], [[0.0, 0.0], [1.0, 1.0]])

        >>> Explanation().get_pdx(np.array([[1, 0, 2, 1], [3, 1, 2, 0]]), 0, [1, 2])
        ([[1, -1], [2, 1]], [[0.33, 0.0], [0.67, 1.0]])

        """
        pdx = [[(q_values[r][selected_action] - q_values[r][target]) for target in target_actions]
               for r in range(len(q_values))]

        _pdx = np.array(pdx.copy())
        _pdx[_pdx < 0] = 0
        _atomic_pdx = np.array(_pdx).sum(axis=0)
        contribution = [[round(_pdx[r][target] / (_atomic_pdx[target] + 0.001), 2)
                         for target in range(len(target_actions))]
                        for r in range(len(q_values))]
        return pdx, contribution

    def get_gtx(self, env, model, state_config, action_space, episodes = 1):
        """Estimate the ground truth for state and actions"""

        expected_q_values = []
        for episode in range(episodes):
            current_config = copy.deepcopy(state_config)
            state = env.reset(**current_config)
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)
            episode_reward = []
        
            for action in range(action_space):
                rewards = []
                state, reward, done, info = env.step(action)
                rewards.append(reward)

                while not done:
                    state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)
                    cominded_q_values, q_values = model(state)
                    q_values = q_values.squeeze(1).data.numpy()
                    action = int(cominded_q_values.data.max(1)[1])

                    state, reward, done, info = env.step(action)
                    rewards.append(reward)

                rewards = np.stack(rewards)
                episode_reward.append(rewards.sum(0))

            expected_q_values.append(episode_reward)
        expected_q_values = np.stack(expected_q_values, 1)

        q_values = expected_q_values.mean(1)

        return q_values


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)

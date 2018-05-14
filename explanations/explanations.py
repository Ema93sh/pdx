import numpy as np
import torch
import time
import copy
from functools import partial
from torch.autograd import Variable
import threading
from multiprocessing import Queue

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


    def run_episode(self, episode, q, model, state_config, action_space, env, gamma, epsilon):
        model.eval()
        current_config = copy.deepcopy(state_config)
        episode_reward = []
        start_time = time.time()
        model_time = 0
        env_time = 0
        state_gen_time = 0
        for action in range(action_space):
            _ = env.reset(**current_config)
            rewards = []
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            ep_step = 0
            while not done:
                should_explore = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
                if should_explore:
                    action = np.random.choice(action_space)
                else:
                    state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)
                    model_start_time = time.time()
                    cominded_q_values = model(state)
                    model_end_time = time.time()
                    model_time += model_end_time - model_start_time
                    if len(cominded_q_values) == 2:  # TODO need a better way
                        cominded_q_values = cominded_q_values[0]
                    action = int(cominded_q_values.data.max(1)[1])

                env_start_time = time.time()
                state, reward, done, info = env.step(action)
                env_end_time = time.time()
                state_gen_time += info["state_gen_time"]
                env_time += env_end_time - env_start_time
                rewards.append(((gamma ** ep_step) * np.array(reward)).tolist())
                ep_step += 1
            rewards = np.stack(rewards)
            episode_reward.append(rewards.sum(0))
        end_time = time.time()
        # print("Done running episode %d with %d step took %.2fs model time %.2fs env time %.2f state gen time %.2f epsilon %.2f " % (episode, ep_step, end_time - start_time, model_time, env_time, state_gen_time, epsilon))
        q.put(episode_reward)
        return episode_reward


    def gt_q_values(self, env, model, state_config, action_space, episodes=1, gamma=0.99, epsilon = 0):
        """Estimate the ground truth Q-Values for a given state and it's action space"""

        expected_q_values = []
        multi_q = Queue()
        T = []
        for episode in range(episodes):
            _env = copy.deepcopy(env)
            _model = copy.deepcopy(model)
            _state_config = copy.deepcopy(state_config)
            t = threading.Thread(target=self.run_episode, args = (episode, multi_q, _model, _state_config, action_space, _env, gamma, epsilon))
            t.daemon = True
            t.start()
            T.append(t)


        for t in T:
            t.join()

        for t in T:
            expected_q_values.append(multi_q.get())

        expected_q_values = np.stack(expected_q_values, 1)

        q_values = expected_q_values.mean(1)

        return q_values.T

    def mse_pdx(self, prediction_x, target_x):
        """Mean Square Error between Predicted and Target explanations

        >>> Explanation().mse_pdx([[0, 0], [1, 2]],[[0, 0], [1, 2]])
        0.0
        >>> Explanation().mse_pdx([[3, -2], [1, 2]],[[0, 0], [1, 2]])
        3.25
        """

        return np.square(np.subtract(prediction_x, target_x)).mean()


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)

import torch
import random
import math
import time
import copy

import numpy as np
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn

from .replay_memory import ReplayMemory, Transition
from .base_task_runner import BaseTaskRunner
from explanations import Explanation

class TaskRunner(BaseTaskRunner):
    """Class that runs the task"""

    def __init__(self, env, model, config, query_states=[], viz = None):
        super(TaskRunner, self).__init__(config)
        self.env = env
        self.model = model
        self.viz = viz
        self.action_space = env.action_space
        self.target_model = model.clone()
        self.optimizer = Adam(self.model.parameters(), lr = self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.query_states = query_states

    def select_action(self, state, restart_epsilon = False):
        sample = random.random()
        self.current_epsilon_step += 1

        if restart_epsilon:
            self.current_epsilon_step = 100

        self.epsilon = np.max([0, self.starting_epsilon * (0.96 ** (self.current_epsilon_step / self.decay_rate))])

        should_explore = np.random.choice([True, False],  p = [self.epsilon, 1 - self.epsilon])

        if not should_explore:
            cominded_q_values = self.model(state)
            return cominded_q_values.data.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(self.action_space)]])

    def train(self, training_episodes = 5000, max_steps = 10000):
        self.model.train()
        restart_epsilon = False
        for episode in range(training_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                self.global_steps += 1

                action = self.select_action(state)
                restart_epsilon = False

                next_state, reward, done, info = self.env.step(int(action))
                total_reward += reward

                reward = torch.FloatTensor([reward])

                next_state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                self.replay_memory.push(state, action, reward, next_state, torch.LongTensor([done]))

                state = next_state

                self.update_model()

                if self.global_steps % self.update_steps == 0:
                    self.target_model.clone_from(self.model)

                if self.current_epsilon_step != 0 and self.restart_epsilon_steps != 0 and self.current_epsilon_step % self.restart_epsilon_steps == 0:
                    restart_epsilon = True

                if self.global_steps % self.save_steps == 0:
                    self.generate_explanation(episode)
                    self.plot_summaries()
                    self.save()

                if done:
                    self.summary_log(episode + 1, "Total Reward", total_reward)
                    self.summary_log(episode + 1, "Epsilon", self.epsilon)
                    self.summary_log(episode + 1, "Total Step", step + 1)
                    if episode % self.log_interval == 0:
                        print("Training Episode %d total reward %d with steps %d" % (episode + 1, total_reward, step + 1))
                    break

        self.plot_summaries()
        self.save()

    def generate_explanation(self, episode):
        # Explanation
        explanation = Explanation()
        pdx_mse = 0
        for state_config in self.query_states:
            current_config = copy.deepcopy(state_config)

            state = self.env.reset(**current_config)
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            _q_values = self.model(state)
            _q_values =_q_values.data
            state_action = int(_q_values.max(1)[1][0])


            gt_q = explanation.gt_q_values(self.env, self.model, current_config, self.env.action_space,
                                           episodes=10)

            _target_actions = [i for i in range(self.env.action_space) if i != state_action]
            predict_x, _ = explanation.get_pdx(_q_values, state_action, _target_actions)
            target_x, _ = explanation.get_pdx([gt_q], state_action, _target_actions)
            pdx_mse += explanation.mse_pdx(predict_x, target_x)
        pdx_mse /= len(self.query_states)
        self.summary_log(episode + 1, "MSE - PDX", pdx_mse)



    def update_model(self):
        if len(self.replay_memory) < self.batch_size:
            return


        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_state_batch = torch.cat(batch.next_state)
        done_batch = Variable(torch.cat(batch.done), volatile = True)

        q_values = self.model(state_batch).gather(1, action_batch)

        q_next = Variable(torch.zeros(self.batch_size).type(torch.FloatTensor))

        non_terminal_batch =  1 - done_batch

        q_next[non_terminal_batch] = self.target_model(next_state_batch[non_terminal_batch]).max(1)[0]

        target_q_values = reward_batch + self.discount_factor * q_next

        target_q_values = Variable(target_q_values.data)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


    def test(self, test_episodes= 100, max_steps = 100, render = False, sleep=1):
        self.model.eval()

        q_box_opts = dict(
            title='Q Values',
            rownames=[action for action in self.env.get_action_meanings]
        )
        q_box = None

        for episode in range(test_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                q_values = self.model(state).data
                action  =  q_values.max(1)[1]

                state, reward, done, info = self.env.step(int(action))
                total_reward += reward

                if render:
                    self.env.render()
                    if q_box is None:
                        q_box = self.viz.bar( X = q_values.numpy()[0], opts=q_box_opts)
                    else:
                        self.viz.bar( X =q_values.numpy()[0], opts=q_box_opts, win=q_box)

                    time.sleep(sleep)

                if done:
                    print("Test Episode %d total reward %d with steps %d" % (episode+1, total_reward, step + 1))
                    break

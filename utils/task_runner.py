import torch
import random
import math

from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from .replay_memory import ReplayMemory, Transition
from .base_task_runner import BaseTaskRunner

class TaskRunner(BaseTaskRunner):
    """Class that runs the task"""

    def __init__(self, env, model, config):
        super(TaskRunner, self).__init__(config)
        self.env = env
        self.model = model
        self.action_space = env.action_space
        self.target_model = model.clone()
        self.optimizer = Adam(self.model.parameters(), lr = self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = 0 + (1.0 - 0.01) * math.exp(-1. * self.global_steps / self.decay_rate)
        if sample > eps_threshold:
            q_values = self.model(state).data
            return q_values.max(1)[1].view(1, 1)
        else:
            return torch.LongTensor([[random.randrange(self.action_space)]])

    def train(self, training_episodes = 5000, max_steps = 10000):
        self.model.train()

        for episode in range(training_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                self.global_steps += 1

                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(int(action))
                total_reward += reward

                reward = torch.FloatTensor([reward])

                next_state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                self.replay_memory.push(state, action, reward, next_state, torch.LongTensor([done]))

                state = next_state

                self.update_model()

                if self.global_steps % self.update_steps == 0:
                    self.target_model.clone_from(self.model)

                #TODO Generate plots!

                if done:
                    self.summary_log(self.global_steps, "Total Reward", total_reward)
                    self.summary_log(self.global_steps, "Total Step", step + 1)

                    if self.global_steps % self.log_interval == 0:
                        print("Training Episode %d total reward %d with steps %d" % (episode+1, total_reward, step + 1))
                    break

        self.save()


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


    def test(self, test_episodes= 100, max_steps = 100, render = False):
        self.model.eval()

        for episode in range(test_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                q_values = self.model(state).data
                action  =  q_values.max(1)[1].view(1, 1)

                next_state, reward, done, info = self.env.step(int(action))
                total_reward += reward
                state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                if done:
                    print("Test Episode %d total reward %d with steps %d" % (episode+1, total_reward, step + 1))
                    break


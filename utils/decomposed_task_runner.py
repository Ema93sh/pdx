import torch
import time
import random
import math
import numpy as np
import copy
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from .replay_memory import ReplayMemory, Transition
from .base_task_runner import BaseTaskRunner
from explanations import Explanation


class DecomposedQTaskRunner(BaseTaskRunner):
    """Training and evaluation for decomposed Q learning"""

    def __init__(self, env, model, config, query_states=[], viz=None):
        super(DecomposedQTaskRunner, self).__init__(config)
        self.env = env
        self.model = model
        self.action_space = env.action_space
        self.target_model = model.clone()
        self.replay_memory = ReplayMemory(self.replay_capacity)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.viz = viz
        self.query_states = query_states


    def select_action(self, state, restart_epsilon = False):
        sample = random.random()
        self.current_epsilon_step += 1

        if restart_epsilon:
            self.current_epsilon_step = 100

        self.epsilon = np.max([0, self.starting_epsilon * (0.96 ** (self.current_epsilon_step / self.decay_rate))])

        should_explore = np.random.choice([True, False],  p = [self.epsilon, 1 - self.epsilon])

        if not should_explore:
            cominded_q_values, q_values = self.model(state)
            return cominded_q_values.data.max(1)[1]
        else:
            return torch.LongTensor([random.randrange(self.action_space)])

    def train(self, training_episodes=5000, max_steps=10000):
        self.model.train()
        restart_epsilon = False
        for episode in range(training_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                self.global_steps += 1

                action = self.select_action(state, restart_epsilon)
                restart_epsilon = False

                next_state, reward, done, info = self.env.step(int(action))
                total_reward += sum(reward)

                reward = torch.FloatTensor([reward])

                next_state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                self.replay_memory.push(state, action, reward, next_state, torch.FloatTensor([done]))

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
                        print(
                            "Training Episode %d total reward %d with steps %d" % (episode + 1, total_reward, step + 1))
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

            cominded_q_values, _q_values = self.target_model(state)
            state_action = int(cominded_q_values.data.max(1)[1][0])
            _q_values = _q_values.data.numpy().squeeze(1)

            gt_q = explanation.gt_q_values(self.env, self.target_model, current_config, self.env.action_space,
                                           episodes=10)

            _target_actions = [i for i in range(self.env.action_space) if i != state_action]
            predict_x = explanation.get_pdx(_q_values, state_action, _target_actions)
            target_x = explanation.get_pdx(gt_q, state_action, _target_actions)
            pdx_mse += explanation.mse_pdx(predict_x, target_x)
        pdx_mse /= len(self.query_states)
        self.summary_log(episode + 1, "MSE - PDX", pdx_mse)


    def update_model(self):
        if len(self.replay_memory) < self.batch_size:
            return 0

        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_state_batch = torch.cat(batch.next_state)
        done_batch = Variable(torch.cat(batch.done), volatile=True)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        _, q_values = self.model(state_batch)

        q_values = q_values[:, batch_index, action_batch]

        _, q_next = self.target_model(next_state_batch)

        q_next = q_next.mean(2)

        q_next = (1 - done_batch) * q_next

        target_q_values = reward_batch.t() + self.discount_factor * q_next

        target_q_values = Variable(target_q_values.data)

        loss = self.loss_fn(q_values, target_q_values)
        update_start_time = time.time()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        update_end_time = time.time()
        return (update_end_time - update_start_time)


    def test(self, test_episodes=100, max_steps=100, render=False, sleep=1):
        self.model.eval()
        if render:
            info_text_box = None
            info_box_opts = dict(title="Info Box")
            q_box = None
            q_box_title = 'Q Values'
            q_box_opts = dict(
                title='Q Values',
                rownames=[action for action in self.env.get_action_meanings]
            )
            decomposed_q_box = None
            decomposed_q_box_opts = dict(
                title='Decomposed Q Values',
                stacked=False,
                legend=['R' + str(i) for i in range(self.env.reward_types)],
                rownames=[action for action in self.env.get_action_meanings]
            )
            pdx_box = None
            pdx_box_opts = dict(
                title='PDX',
                stacked=False,
                legend=['R' + str(i) for i in range(self.env.reward_types)],
            )
            pdx_box_title = 'PDX'
            pdx_contribution_box = None
            pdx_contribution_box_title = 'PDX Contribution(%)'
            cont_pdx_box_opts = dict(
                title='PDX Contribution(%)',
                stacked=False,
                legend=['R' + str(i) for i in range(self.env.reward_types)],
            )
        explaination = Explanation()
        for episode in range(test_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                cominded_q_values, q_values = self.model(state)
                q_values = q_values.squeeze(1).data.numpy()
                action = int(cominded_q_values.data.max(1)[1])

                next_state, reward, done, info = self.env.step(action)
                total_reward += sum(reward)
                state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                if render:
                    self.env.render()
                    q_box_opts['title'] = q_box_title + '- (Selected Action:' + str(self.env.get_action_meanings[action]) + ')'
                    if q_box is None:
                        q_box = self.viz.bar(X=cominded_q_values.data.numpy()[0], opts=q_box_opts)
                    else:
                        self.viz.bar(X=cominded_q_values.data.numpy()[0], opts=q_box_opts, win=q_box)

                    if decomposed_q_box is None:
                        decomposed_q_box = self.viz.bar(X=q_values.T, opts=decomposed_q_box_opts)
                    else:
                        self.viz.bar(X=q_values.T, opts=decomposed_q_box_opts, win=decomposed_q_box)

                    pdx_box_opts['rownames'] = ['(' + self.env.get_action_meanings[action] + ',' + self.env.get_action_meanings[i] + ')'
                                                for i in range(self.env.action_space) if i != action]
                    if len(pdx_box_opts['rownames']) == 1:
                        pdx_box_opts['title'] = pdx_box_title + '    ' + pdx_box_opts['rownames'][0]
                        cont_pdx_box_opts['title'] = pdx_contribution_box_title + '   ' + pdx_box_opts['rownames'][0]
                        pdx_box_opts.pop('rownames')
                    else:
                        cont_pdx_box_opts['rownames'] = pdx_box_opts['rownames']

                    _target_actions = [j for j in range(self.env.action_space) if j != action]
                    pdx, contribute = explaination.get_pdx(q_values, action, _target_actions)

                    if pdx_box is None:
                        pdx_box = self.viz.bar(X=np.array(pdx).T, opts=pdx_box_opts)
                    else:
                        self.viz.bar(X=np.array(pdx).T, opts=pdx_box_opts, win=pdx_box)
                    if pdx_contribution_box is None:
                        pdx_contribution_box = self.viz.bar(X=np.array(contribute).T, opts=cont_pdx_box_opts)
                    else:
                        self.viz.bar(X=np.array(contribute).T, opts=cont_pdx_box_opts, win=pdx_contribution_box)

                    # js_injection = '<javascript>' \
                    #                'document.querySelector("button[data-original-title=Repack]").click()' \
                    #                '</javascript>'
                    # if info_text_box is None:
                    #     info_text_box = self.viz.text(js_injection, opts=info_box_opts)
                    # else:
                    #     self.viz.text(js_injection, win=info_text_box, opts=info_box_opts)

                    time.sleep(sleep)

                    if done:
                        print("Test Episode %d total reward %d with steps %d" % (episode + 1, total_reward, step + 1))
                        break
            pass

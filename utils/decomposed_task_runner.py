import torch
import time
import random
import math
import numpy as np
import copy
import threading
from multiprocessing import Queue

from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from .replay_memory import ReplayMemory, Transition
from .base_task_runner import BaseTaskRunner
from explanations import Explanation

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


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
        self.decomposed_q_box = None
        self.pdx_box = None
        self.pdx_contribution_box = None
        self.q_box = None

    def select_action(self, state, restart_epsilon = False):
        self.current_epsilon_step += 1

        if restart_epsilon:
            self.current_epsilon_step = 100

        self.epsilon = np.max([self.minimum_epsilon, self.starting_epsilon * (0.96 ** (self.current_epsilon_step / self.decay_rate))])

        should_explore = np.random.choice([True, False], p=[self.epsilon, 1 - self.epsilon])

        if not should_explore:
            cominded_q_values, q_values = self.model(state)
            return cominded_q_values.data.max(1)[1]
        else:
            return LongTensor([random.randrange(self.action_space)])

    def explore_policy(self):
        pass

    def train(self, training_episodes=5000, max_steps=10000):
        self.model.train()
        self.best_score = 0
        restart_epsilon = False
        explore = None
        for episode in range(training_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)
            episode_time = time.time()
            total_decision_time = 0
            update_time = 0
            for step in range(max_steps):
                self.global_steps += 1

                decision_time = time.time()
                action = self.select_action(state, restart_epsilon)
                decision_time = time.time() - decision_time

                total_decision_time += decision_time

                restart_epsilon = False

                next_state, reward, done, info = self.env.step(int(action))
                total_reward += sum(reward)

                reward = FloatTensor([reward])

                next_state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)

                self.replay_memory.push(state, action, reward, next_state, FloatTensor([done]))

                state = next_state

                if self.global_steps % self.update_frequency == 0:
                    step_update_time = time.time()
                    self.update_model()
                    update_time += (time.time() - step_update_time)

                if self.global_steps % self.target_update_frequency == 0:
                    self.target_model.clone_from(self.model)

                if self.current_epsilon_step != 0 and self.restart_epsilon_steps != 0 and self.current_epsilon_step % self.restart_epsilon_steps == 0:
                    restart_epsilon = True

                if self.global_steps % self.save_steps == 0:
                    self.save_best_model(episode)
                    self.save_summaries()
                    self.generate_explanation(episode)
                    self.plot_summaries()

                if done:
                    episode_time =  time.time() - episode_time
                    self.summary_log(episode + 1, "Total Reward", total_reward)
                    self.summary_log(episode + 1, "Epsilon", self.epsilon)
                    self.summary_log(episode + 1, "Total Step", step + 1)
                    if episode % self.log_interval == 0:
                        print("Training Episode %d total reward %d with steps %d. " % (episode + 1, total_reward, step + 1))
                        print("Total Time %.2f Decision Time %.2f Update Time %.2f" %(episode_time, total_decision_time, update_time))
                    break

        self.save_best_model(episode)
        self.plot_summaries()

    def save_best_model(self, episode):
        if not self.save_model:
            return

        current_model_score = self.test(test_episodes=5, log_steps=0)
        self.model.train()
        self.summary_log(episode + 1, "Test Score", current_model_score)

        if self.best_score < current_model_score:
            print("Best Score...", current_model_score)
            self.best_score = current_model_score
            self.save()

    def generate_for_state(self, id, env, model, state_config, q, explanation):
        start_time = time.time()
        current_config = copy.deepcopy(state_config)
        state = env.reset(**current_config)
        state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

        cominded_q_values, _q_values = model(state)
        state_action = int(cominded_q_values.data.max(1)[1][0])
        _q_values = _q_values.data.numpy().squeeze(1)

        gt_start_time = time.time()
        epsilon = self.epsilon if self.explore_gt else 0
        gt_q = explanation.gt_q_values(env, model, current_config, env.action_space,
                                       episodes=10, gamma=self.discount_factor, epsilon = epsilon)
        gt_end_time = time.time()
        gt_time = gt_end_time - gt_start_time
        _target_actions = [i for i in range(env.action_space) if i != state_action]
        predict_x, _ = explanation.get_pdx(_q_values, state_action, _target_actions)
        target_x, _ = explanation.get_pdx(gt_q, state_action, _target_actions)
        pdx_mse = explanation.mse_pdx(predict_x, target_x)
        q_values_mse = explanation.mse_pdx(_q_values, gt_q)
        end_time = time.time()
        print("Done running scenario %d, %.2f with total time %.2f with epsilon %.2f" % (id, end_time - start_time, gt_time, epsilon))
        q.put((pdx_mse,  q_values_mse))
        return

    def generate_explanation(self, episode):
        # Explanation
        if len(self.query_states) == 0:
            return

        print("Generating explanation....%d step" % self.global_steps)
        explanation = Explanation()
        pdx_mse = 0
        q_values_mse = 0
        start_time = time.time()

        T = []
        multi_q = Queue()
        for i, state_config in enumerate(self.query_states):
            env = copy.deepcopy(self.env)
            model = copy.deepcopy(self.model)
            t = threading.Thread(target=self.generate_for_state, args = (i, env, model, state_config, multi_q, explanation))
            t.daemon = True
            t.start()
            T.append(t)

        for t in T:
            t.join()

        for t in T:
            _pdx_mse, _q_values_mse = multi_q.get()
            pdx_mse += _pdx_mse
            q_values_mse += _q_values_mse

        end_time = time.time()
        print("Done...Took %.2f seconds" % (end_time - start_time))
        pdx_mse /= len(self.query_states)
        self.summary_log(episode + 1, "MSE - PDX", pdx_mse)
        self.summary_log(episode + 1, "MSE - Q-values", q_values_mse)

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

        self.summary_log(self.global_steps, "Loss", loss.data[0])

        update_start_time = time.time()

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        update_end_time = time.time()
        return (update_end_time - update_start_time)

    def test(self, test_episodes=100, max_steps=1000, render=False, sleep=1, log_steps=1):
        self.model.eval()

        test_score = 0

        for episode in range(test_episodes):
            state = self.env.reset()
            total_reward = 0
            state = Variable(torch.Tensor(state.tolist())).unsqueeze(0)

            for step in range(max_steps):
                cominded_q_values, q_values = self.model(state)
                q_values = q_values.squeeze(1).data.numpy()
                action = int(cominded_q_values.data.max(1)[1])

                if render:
                    self.env.render()
                    self.render_q_values(action, cominded_q_values, q_values)
                    time.sleep(sleep)

                next_state, reward, done, info = self.env.step(action)
                total_reward += sum(reward)
                state = Variable(torch.Tensor(next_state.tolist())).unsqueeze(0)


                if done:
                    test_score += total_reward
                    if log_steps != 0 and step % log_steps == 0:
                        print("Test Episode %d total reward %d with steps %d" % (episode + 1, total_reward, step + 1))
                    break

        return test_score / test_episodes

    def render_q_values(self, action, cominded_q_values, q_values):
        # TODO clean up this mess
        explaination = Explanation()

        info_text_box = None
        info_box_opts = dict(title="Info Box")
        q_box = None
        q_box_title = 'Q Values'
        q_box_opts = dict(
            title='Q Values',
            rownames=[action for action in self.env.get_action_meanings]
        )
        decomposed_q_box_opts = dict(
            title='Decomposed Q Values',
            stacked=False,
            legend=[r_type for r_type in self.env.get_reward_meanings],
            rownames=[action for action in self.env.get_action_meanings]
        )

        pdx_box_opts = dict(
            title='PDX',
            stacked=False,
            legend=[r_type for r_type in self.env.get_reward_meanings],
        )
        pdx_box_title = 'PDX'

        pdx_contribution_box_title = 'PDX Contribution(%)'
        cont_pdx_box_opts = dict(
            title='PDX Contribution(%)',
            stacked=False,
            legend=[r_type for r_type in self.env.get_reward_meanings],
        )

        q_box_opts['title'] = q_box_title + '- (Selected Action:' + str(
            self.env.get_action_meanings[action]) + ')'

        if self.q_box is None:
            self.q_box = self.viz.bar(X=cominded_q_values.data.numpy()[0], opts=q_box_opts)
        else:
            self.viz.bar(X=cominded_q_values.data.numpy()[0], opts=q_box_opts, win=self.q_box)

        if self.decomposed_q_box is None:
            self.decomposed_q_box = self.viz.bar(X=q_values.T, opts=decomposed_q_box_opts)
        else:
            self.viz.bar(X=q_values.T, opts=decomposed_q_box_opts, win=self.decomposed_q_box)

        pdx_box_opts['rownames'] = [
            '(' + self.env.get_action_meanings[action] + ',' + self.env.get_action_meanings[i] + ')'
            for i in range(self.env.action_space) if i != action]

        if len(pdx_box_opts['rownames']) == 1:
            pdx_box_opts['title'] = pdx_box_title + '    ' + pdx_box_opts['rownames'][0]
            cont_pdx_box_opts['title'] = pdx_contribution_box_title + '   ' + pdx_box_opts['rownames'][0]
            pdx_box_opts.pop('rownames')
        else:
            cont_pdx_box_opts['rownames'] = pdx_box_opts['rownames']

        _target_actions = [j for j in range(self.env.action_space) if j != action]
        pdx, contribute = explaination.get_pdx(q_values, action, _target_actions)

        if self.pdx_box is None:
            self.pdx_box = self.viz.bar(X=np.array(pdx).T, opts=pdx_box_opts)
        else:
            self.viz.bar(X=np.array(pdx).T, opts=pdx_box_opts, win=self.pdx_box)

        if self.pdx_contribution_box is None:
            self.pdx_contribution_box = self.viz.bar(X=np.array(contribute).T, opts=cont_pdx_box_opts)
        else:
            self.viz.bar(X=np.array(contribute).T, opts=cont_pdx_box_opts, win=self.pdx_contribution_box)

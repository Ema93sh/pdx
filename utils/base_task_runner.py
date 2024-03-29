import os
import torch
import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from .replay_memory import ReplayMemory, Transition

class BaseTaskRunner(object):

    def __init__(self, config):
        self.learning_rate = config["learning_rate"]
        self.replay_capacity = config["replay_capacity"]
        self.batch_size = config["batch_size"]
        self.discount_factor = config["discount_factor"]
        self.save_model = config["save_model"]
        self.global_steps = 0
        self.decay_rate = config["decay_rate"]
        self.target_update_frequency = config["target_update_frequency"]
        self.update_frequency = config["update_frequency"]
        self.log_interval = config["log_interval"]
        self.replay_memory = ReplayMemory(self.replay_capacity)
        self.file_name = config["file_name"]
        self.summaries = {}
        self.result_path = config["result_path"]
        self.save_steps = config["save_steps"]
        self.current_epsilon_step = self.global_steps
        self.starting_epsilon = config['starting_episilon']
        self.minimum_epsilon = config['minimum_epsilon']
        self.restart_epsilon_steps = config["restart_epsilon_steps"]
        self.explore_gt = config["explore_gt"]


    def save(self):
        if self.save_model and self.file_name:
            cwd = os.getcwd()
            network_path = self.result_path if self.result_path != "" else "results/saved_models"
            network_path = os.path.join(cwd, network_path, self.file_name)
            if not os.path.exists(os.path.dirname(network_path)):
                os.makedirs(os.path.dirname(network_path))
            torch.save(self.model.state_dict(), network_path)

    def summary_log(self, step, tag, value):
        #TODO get the labels for x and y valuess
        if tag in self.summaries:
            self.summaries[tag].append((step, value))
        else:
            self.summaries[tag] = [(step, value)]

    def save_summaries(self):
        cwd = os.getcwd()
        plots_dir_path = os.path.join(cwd, self.result_path)
        if not os.path.exists(plots_dir_path):
            os.makedirs(plots_dir_path)

        with open(os.path.join(plots_dir_path, "summaries.pickle"), "wb") as file:
            pickle.dump(self.summaries, file, protocol=pickle.HIGHEST_PROTOCOL)

    def plot_summaries(self):
        cwd = os.getcwd()
        plots_dir_path = os.path.join(cwd, self.result_path)
        if not os.path.exists(plots_dir_path):
            os.makedirs(plots_dir_path)

        for title, values in self.summaries.items():
            x_values = list(map(lambda x: x[0], values))
            y_values = list(map(lambda x: x[1], values))
            plt.plot(x_values, y_values)
            plt.grid(True)
            plt.title(title)
            plt.savefig(os.path.join(plots_dir_path, title + ".png"))
            plt.clf()

import os
import torch

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
        self.update_steps = config["update_steps"]
        self.log_interval = config["log_interval"]
        self.replay_memory = ReplayMemory(self.replay_capacity)
        self.file_name = config["file_name"]


    def save(self):
        if self.save_model and self.file_name:
            cwd = os.getcwd()
            network_path = os.path.join(cwd, "results/saved_models", self.file_name)
            torch.save(self.model.state_dict(), network_path)

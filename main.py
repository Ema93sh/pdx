import torch
from envs import FruitCollection1D
from models.q_model import QModel
from models.decomposed_q_model import DecomposedQModel
from utils.task_runner import TaskRunner
from utils.decomposed_task_runner import DecomposedQTaskRunner


if __name__ == '__main__':

    #TODO args to setup the experiment

    # env = FruitCollection1D()
    # state = env.reset()
    #
    # model = QModel(len(state), 2)
    #
    # task_runner = TaskRunner(env, model)
    #
    # task_runner.train()
    # task_runner.test()

    env = FruitCollection1D()
    state = env.reset()

    model = DecomposedQModel(2, len(state), 2)

    task_runner = DecomposedQTaskRunner(env, model)

    task_runner.train()
    task_runner.test()

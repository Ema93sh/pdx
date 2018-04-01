from envs import FruitCollection1D
from models.dqn import DQNModel
from task_runner import TaskRunner


if __name__ == '__main__':

    #TODO args to setup the experiment

    env = FruitCollection1D()
    state = env.reset()

    model = DQNModel(len(state), 2)

    task_runner = TaskRunner(env, model)

    task_runner.train()
    task_runner.test()

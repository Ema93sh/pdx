import argparse
import torch
from envs import FruitCollection1D, FruitCollection2D
from models.q_model import QModel
from models.decomposed_q_model import DecomposedQModel
from utils.task_runner import TaskRunner
from utils.decomposed_task_runner import DecomposedQTaskRunner


def get_env(name, decompose):
    env_map = {
        "FruitCollection1D": FruitCollection1D,
        "FruitCollection2D": FruitCollection2D
    }
    env = env_map[name](hybrid = decompose)
    return env

def get_model(env, decompose):
    state = env.reset()
    if decompose:
        model = DecomposedQModel(env.reward_types, len(state), env.action_space)
    else:
        model = QModel(len(state), env.action_space)
    return model

def get_task_runner(env, model, decompose):
    if decompose:
        return DecomposedQTaskRunner(env, model)

    return TaskRunner(env, model)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run PDX!')

    parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 10)')
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch Size(No. of Episodes) for Training')
    parser.add_argument('--log-interval', type=int, default=5,
                        help='interval between training status logs (default: 5)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables Cuda Usage')
    parser.add_argument('--train', action='store_true', default=False, help='Train the network')
    parser.add_argument('--test', action='store_true', default=False, help='Test the network')
    parser.add_argument('--train_episodes', type=int, default=500, help='Episode count for training')
    parser.add_argument('--test_episodes', type=int, default=100, help='Episode count for testing')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate for Training (Adam Optimizer)')
    parser.add_argument('--scratch', action='store_true', default=False,
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    parser.add_argument('--env', default="FruitCollection1D",
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    parser.add_argument('--decompose', action='store_true', help='Decompose Q Values')
    parser.add_argument('--sleep', type=int, help='Sleep time for render', default=1)


    args = parser.parse_args()

    env = get_env(args.env, args.decompose)

    state = env.reset()

    model = get_model(env, args.decompose)

    task_runner = get_task_runner(env, model, args.decompose)

    task_runner.train(training_episodes = args.train_episodes, max_steps = args.max_steps)


    task_runner.test(test_episodes = args.test_episodes, max_steps = args.max_steps, render = args.render)

import argparse
import torch
import os
import visdom
import json
from envs import FruitCollection1D, FruitCollection2D, TreasureHunter

from models.q_model import QModel, QModelCNN
from models.decomposed_q_model import DecomposedQModel, DecomposedCNNQModel
from utils.task_runner import TaskRunner
from utils.decomposed_task_runner import DecomposedQTaskRunner
from explanations import Explanation

from torch.autograd import Variable


def get_env(args, viz):
    env_map = {
        "FruitCollection1D": FruitCollection1D,
        "FruitCollection2D": FruitCollection2D,
        "TreasureHunter": TreasureHunter
    }

    scenarios = []
    if not args.test and args.scenarios_path != "":
        print("Loading scenarios...")
        scenarios = json.load(open(args.scenarios_path))

    state_representation = "grid" if args.cnn else "linear"
    env = env_map[args.env](hybrid=args.decompose, vis=viz, state_representation = state_representation)

    return env, scenarios


def get_model(env, args):
    state = env.reset()
    if args.decompose:
        if args.cnn:
            model = DecomposedCNNQModel(env.reward_types, len(state), env.action_space)
        else:
            model = DecomposedQModel(env.reward_types, len(state), env.action_space)

    else:
        model = QModelCNN(len(state), env.action_space) if args.cnn else QModel(len(state), env.action_space)

    if args.load:
        cwd = os.getcwd()
        file_name = "%s_%s_.torch" % (env.name, "decompose" if args.decompose else "simple")
        network_path = args.result_path if args.result_path != "" else "results/saved_models"
        network_path = os.path.join(cwd, network_path, file_name)
        model.load_state_dict(torch.load(network_path))
    return model


def get_task_runner(env, model, args, query_states, viz=None):
    file_name = "%s_%s_.torch" % (env.name, "decompose" if args.decompose else "simple")
    result_path = "results/%s/%s/%s" % (env.name, "cnn" if args.cnn else "linear", "decompose" if args.decompose else "non_decompose")

    result_path = args.result_path if args.result_path != "" else result_path

    config = {
        "learning_rate": args.lr,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "discount_factor": args.gamma,
        "save_model": args.save,
        "decay_rate": args.decay_rate,
        "update_steps": args.update_steps,
        "log_interval": args.log_interval,
        "file_name": file_name,
        "result_path": result_path,
        "save_steps": args.save_steps,
        "restart_epsilon_steps": args.restart_epsilon_steps
    }

    if args.decompose:
        return DecomposedQTaskRunner(env, model, config, query_states, viz=viz)

    return TaskRunner(env, model, config, query_states, viz=viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PDX!')

    # Evaluation Config
    parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 10)')
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')
    parser.add_argument('--train-episodes', type=int, default=500, help='Episode count for training')
    parser.add_argument('--test-episodes', type=int, default=100, help='Episode count for testing')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--env', default="FruitCollection1D",
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    parser.add_argument('--decompose', action='store_true', help='Decompose Q Values')
    parser.add_argument('--sleep', type=int, help='Sleep time for render', default=1)
    parser.add_argument('--log-interval', type=int, default=5,
                        help='interval between training status logs (default: 5)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables Cuda Usage')
    parser.add_argument('--test', action='store_true', default=False, help='Disables Cuda Usage')
    parser.add_argument('--save-steps', type=int, default=1000, help='Will save after n steps')
    parser.add_argument('--restart-epsilon-steps', type=int, default=0, help='Will restart epsilon after n steps. If 0 no restart')
    parser.add_argument('--result-path', type=str, default="", help='Path to save all the plots and model')

    # Reinforcement Config
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=35, help='Batch Size(No. of Episodes) for Training')
    parser.add_argument('--replay-capacity', type=int, default=5000, help='Size of Experience replay')
    parser.add_argument('--decay-rate', type=int, default=10, help='Size of Experience replay')
    parser.add_argument('--update-steps', type=int, default=50, help='Size of Experience replay')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate for Training (Adam Optimizer)')

    # Network Config
    parser.add_argument('--cnn', action="store_true", default=False)
    parser.add_argument('--load', action='store_true', default=False,
                        help='Train the network from scratch ( or Does not load pre-trained model)')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save the model after training')

    # Explanation Config
    parser.add_argument('--scenarios-path', type=str, default="",
                        help='Path to scenarios. Will run all the scenarios')


    args = parser.parse_args()
    viz = visdom.Visdom() if args.render else None

    env, query_states_config = get_env(args, viz=viz)

    state = env.reset()

    model = get_model(env, args)
    args.cuda = args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        model = model.cuda()

    task_runner = get_task_runner(env, model, args, query_states_config, viz=viz)

    if not args.test:
        task_runner.train(training_episodes=args.train_episodes)

    task_runner.test(test_episodes=args.test_episodes, render=args.render, sleep=args.sleep)

    #
    # explanation = Explanation()
    # state_config = {
    #     "fruits_loc": [1],
    #     "step_count": 0,
    #     "agent_position": [0, 4],
    #     "score": 0,
    #     "step_reward": 0,
    #     "fruit_collected": 0
    # }
    #
    # gtx = explanation.gt_q_values(env, model, state_config, env.action_space, episodes=100)
    # print(gtx)

import argparse
import torch
import os
import visdom
import json
import sys

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
        "TreasureHunter": TreasureHunter,
        "Traveller": None
    }

    scenarios = []
    if not args.test and args.scenarios_path != "":
        print("Loading scenarios...")
        scenarios = json.load(open(args.scenarios_path))


    state_representation = "grid" if args.cnn else "linear"
    env = env_map[args.env](hybrid=args.decompose, vis=viz, state_representation=state_representation,
                            map_name="10x10_easy")

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

    if args.load_path != "":
        print("Loading model from...%s" % args.load_path)
        cwd = os.getcwd()
        network_path = os.path.join(cwd, args.load_path)
        model.load_state_dict(torch.load(network_path))
    return model


def get_task_runner(env, model, args, query_states, viz=None):
    file_name = "%s_%s_.torch" % (env.name, "decompose" if args.decompose else "simple")
    result_path = "results/%s/%s/%s" % (
        env.name, "cnn" if args.cnn else "linear", "decompose" if args.decompose else "non_decompose")

    result_path = args.result_path if args.result_path != "" else result_path

    config = {
        "learning_rate": args.lr,
        "replay_capacity": args.replay_capacity,
        "batch_size": args.batch_size,
        "discount_factor": args.gamma,
        "save_model": args.save,
        "decay_rate": args.decay_rate,
        "update_frequency": args.update_frequency,
        "target_update_frequency": args.target_update_frequency,
        "log_interval": args.log_interval,
        "file_name": file_name,
        "result_path": result_path,
        "save_steps": args.save_steps,
        "restart_epsilon_steps": args.restart_epsilon_steps,
        "starting_episilon": args.starting_episilon,
        "minimum_epsilon": args.minimum_epsilon,
        "prioritized_replay": args.pr,
        "explore_gt": args.explore_gt
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
    parser.add_argument('--test', action='store_true', default=False, help='Run only test')
    parser.add_argument('--train', action='store_true', default=False, help='Run only train')
    parser.add_argument('--save-steps', type=int, default=1000, help='Will save after n steps')
    parser.add_argument('--restart-epsilon-steps', type=int, default=0,
                        help='Will restart epsilon after n steps. If 0 no restart')
    parser.add_argument('--result-path', type=str, default="", help='Path to save all the plots and model')
    parser.add_argument('--starting-episilon', type=float, default=1.0,
                        help='Starting value of epsilon')
    parser.add_argument('--minimum-epsilon', type=float, default=0.1,
                        help='Minimum value of epsilon')

    # Reinforcement Config
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=35, help='Batch Size(No. of Episodes) for Training')
    parser.add_argument('--replay-capacity', type=int, default=5000, help='Size of Experience replay')
    parser.add_argument('--decay-rate', type=int, default=10, help='Decay rate')
    parser.add_argument('--target-update-frequency', type=int, default=50, help='Update frequency for target')
    parser.add_argument('--update-frequency', type=int, default=5, help='model update frequency')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate for Training (Adam Optimizer)')
    parser.add_argument('--pr', action="store_true", default=False, help='Use prioritized_replay')

    # Network Config
    parser.add_argument('--cnn', action="store_true", default=False)
    parser.add_argument('--load-path', default="", help='Load model from this path')
    parser.add_argument('--save', action='store_true', default=False,
                        help='Save the model after training')

    # Explanation Config
    parser.add_argument('--scenarios-path', type=str, default="",
                        help='Path to scenarios. Will run all the scenarios')

    parser.add_argument('--render-scenarios', action='store_true', help='Render all the scenarios')
    parser.add_argument('--explore-gt', action='store_true', default=False, help='Use exploration for ground truth')

    args = parser.parse_args()
    viz = visdom.Visdom() if args.render else None

    env, scenarios = get_env(args, viz=viz)

    if args.render_scenarios:
        if len(scenarios) == 0:
            print("No scenarios loaded!!!!!!!!!")

        for scenario in scenarios:
            env.image_window = None
            print(scenario)
            env.reset(**scenario)
            env.render()
        sys.exit(0)


    state = env.reset()

    model = get_model(env, args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model = model.cuda()

    task_runner = get_task_runner(env, model, args, scenarios, viz=viz)

    if not args.test and not args.render_scenarios:
        task_runner.train(training_episodes=args.train_episodes)

    if not args.train and not args.render_scenarios:
        task_runner.test(test_episodes=args.test_episodes, render=args.render, sleep=args.sleep)

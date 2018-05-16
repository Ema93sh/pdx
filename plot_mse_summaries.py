import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def load_summaries(path):
    path = os.path.join(path, "mse_summaries.pickle")
    with open(path, "rb") as f:
        summary = pickle.load(f)
    return summary

def plot_summaries(summary, path):
    cwd = os.getcwd()
    plots_dir_path = os.path.join(cwd, path)
    y_values = list(map(lambda x: len(summary[x]), sorted(summary.keys())) )
    x_values = sorted(summary.keys())
    width = 1/1.5
    title = "Minimum Sufficient Explanation"
    plt.bar(x_values, y_values, width)
    plt.title(title)
    plt.savefig(os.path.join(plots_dir_path, title + ".png"))
    plt.clf()

def main():
    parser = argparse.ArgumentParser(description='MSE summaries')
    parser.add_argument('--path', type=str, help='Path to summaries folder')
    args = parser.parse_args()

    summary = load_summaries(args.path)
    plot_summaries(summary, args.path)


if __name__ == '__main__':
    main()

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

def load_summaries(summary_paths):
    summaries = []

    for path in summary_paths:
        with open(path, "rb") as f:
         summary = pickle.load(f)
         summaries.append(summary)
    return summaries

def plot_summaries(summaries, tags, result_path):
    cwd = os.getcwd()
    plots_dir_path = os.path.join(cwd, result_path)
    if not os.path.exists(plots_dir_path):
        os.makedirs(plots_dir_path)

    for title in tags:
        min_len = min([len(summary[title]) for summary in summaries])
        x_values = np.array([np.array(list(map(lambda x: x[0], summary[title]))[:min_len]) for summary in summaries])
        x_values = x_values.mean(0)

        y_values = np.array([np.array(list(map(lambda x: x[1], summary[title]))[:min_len]) for summary in summaries])
        y_values = y_values.mean(0)

        plt.plot(x_values, y_values)
        plt.grid(True)
        plt.title(title)
        plt.savefig(os.path.join(plots_dir_path, title + ".png"))
        plt.clf()

def get_summaries_path(path):
    summaries_path = []
    dirs = os.listdir(path)
    for dir in dirs:
        if dir not in [".DS_Store", "merged"]:
            summaries_path.append(os.path.join(path,  dir, "summaries.pickle"))
    return summaries_path


def main():
    path = "./results/TreasureHunter/decompose/average"
    summary_paths = get_summaries_path(path)
    summaries = load_summaries(summary_paths)
    tags = ["Total Reward", "MSE - PDX", "Epsilon", "MSE - Q-values", "Total Step"]
    result_path = os.path.join(path, "merged")
    plot_summaries(summaries, tags, result_path)



if __name__ == '__main__':
    main()

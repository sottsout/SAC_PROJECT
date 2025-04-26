import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def make_logger(log_dir="logs", tb=True):
    outputs = ["stdout", "csv"]
    if tb:
        outputs.append("tensorboard")
    return configure(log_dir, outputs)

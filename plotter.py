from itertools import chain
import matplotlib.pyplot as plt
import numpy as np

def create_fig(num_reward_fns, title):
    plt.ion()
    nrows = 2
    ncols = num_reward_fns
    
    fig, axs = plt.subplots(nrows, ncols, sharex=True)
    fig.suptitle(f'MAML_{title}')

    if num_reward_fns == 1:
        return fig, list(axs)
    else:
        return fig, list(chain.from_iterable(axs))


def set_titles(axs, reward_fns):
    titles = reward_fns * 2
    for i in range(len(reward_fns)):
        titles[i] = f'Train returns ({reward_fns[i]} task)'
        titles[len(reward_fns) + i] = f'Valid returns ({reward_fns[i]} task)'
    
    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
    

def create_lines(axs):
    lines_train = []
    half_axes = int(len(axs)/2)
    
    # first part of axes
    for i, ax in enumerate(axs[:half_axes]):
        line, = ax.plot([])
        lines_train.append(line)
    
    lines_valid = []
    # second part of axes
    for i, ax in enumerate(axs[half_axes:]):
        line, = ax.plot([])
        lines_valid.append(line)

    return lines_train, lines_valid


def set_lines_data(lines, returns, reward_fns):
    for i, line in enumerate(lines):
        entries = returns[reward_fns[i]]
        line.set_data(range(len(entries)), entries)

def render(axs):
    for ax in axs:
        ax.relim()
        ax.autoscale()

    plt.draw()
    plt.pause(0.0000001)



if __name__ == '__main__':
    reward_fns = ['a', 'b', 'c']

    fig, axs = create_fig(len(reward_fns), 'bobbs')
    set_titles(axs, reward_fns)
    plt.show()
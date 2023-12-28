import random
import math
import numpy as np
import scipy.stats as stats
from gym.envs.classic_control import MountainCarEnv as MountainCarEnv_

class MountainCarMetaLearning(MountainCarEnv_):
    def __init__(self, low=0.0, high=0.6):
        super().__init__()

        if high > self.max_position:
            print(f'Warning: clipping value of high="{high}" to "{self.max_position}", since this is the maximum allowed position in the environment')
            high = self.max_position

        self.min_goal_pos = low
        self.max_goal_pos = high
        

    def sample_tasks(self, num_tasks):
        # lower, upper = self.min_goal_pos, self.max_goal_pos
        # mu, sigma = 0.5, 0.1
        # # create a truncated norm so that most of the tasks are still 0.5 (original)
        # X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # positions = [round(x, 2) for x in X.rvs(num_tasks)]
        tasks = [{'position': 0.5} for _ in range(num_tasks)]
        return tasks


    def reset_task(self, task):
        self._task = task
        self.goal_position = task['position']

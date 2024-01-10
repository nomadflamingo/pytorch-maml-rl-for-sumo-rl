import random
import math
import numpy as np
import scipy.stats as stats
from gym.envs.classic_control import MountainCarEnv as MountainCarEnv_

class MountainCarMetaLearning(MountainCarEnv_):

    def register_reward_fns(self, reward_fns):
        for subgoal in reward_fns:
            self.subgoals.append(subgoal)

    def __init__(self, low=0.0, high=0.6):
        super().__init__()

        self.subgoals = []
        

    def sample_tasks(self, num_tasks):
        # lower, upper = self.min_goal_pos, self.max_goal_pos
        # mu, sigma = 0.5, 0.1
        # # create a truncated norm so that most of the tasks are still 0.5 (original)
        # X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

        # positions = [round(x, 2) for x in X.rvs(num_tasks)]
        positions = random.choices(self.subgoals, k=num_tasks)
        return positions


    def reset_task(self, task):
        self._task = task
        self.goal_position = task#task['position']

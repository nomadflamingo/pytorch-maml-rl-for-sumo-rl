from gym.envs.classic_control import CartPoleEnv as CartPoleEnv_

class CartPoleMetaLearning(CartPoleEnv_):
    def sample_tasks(self, num_tasks):
        return [{} for _ in range(num_tasks)]


    def reset_task(self, task):
        pass

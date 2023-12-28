from sumo_rl import SumoEnvironment
from sumo_rl.environment.traffic_signal import TrafficSignal
import random
from typing import Optional

IMPLEMENTED_REWARD_FNS = ['diff-waiting-time', 'average-speed', 'queue', 'pressure']
class SumoEnvironmentMetaLearning(SumoEnvironment):
    def register_reward_fns(self, reward_fns):
        for fn in reward_fns:
            if fn not in IMPLEMENTED_REWARD_FNS:
                raise Exception(f'Reward function {fn} not implemented')
        
        self.reward_fns = reward_fns
    

    def sample_tasks(self, num_tasks):
        tasks = random.choices(self.reward_fns, k=num_tasks)
        return tasks


    def reset_task(self, task):
        self.reward_fn = task

        # update traffic light rewards
        for ts in self.traffic_signals.values():
            ts.reward_fn = task


    def reset(self, seed: Optional[int] = None, return_info=False, **kwargs):
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        self.traffic_signals = {ts: TrafficSignal(self, 
                                                    ts, 
                                                    self.delta_time, 
                                                    self.yellow_time, 
                                                    self.min_green, 
                                                    self.max_green, 
                                                    self.begin_time,
                                                    self.reward_fn,
                                                    self.sumo) for ts in self.ts_ids}
        self.vehicles = dict()

        if self.single_agent:
            if return_info:
                return self._compute_observations()[self.ts_ids[0]], self._compute_info()
            else:
                return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()
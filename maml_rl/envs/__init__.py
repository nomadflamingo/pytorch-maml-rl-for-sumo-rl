from gym.envs.registration import register

# SUMO-RL
# ----------------------------------------
register(
    'SUMO-RL-1.3-v0',
    entry_point='maml_rl.envs.sumo_rl:SumoEnvironmentMetaLearning',
    kwargs={
        'net_file': '/home/beimukvo/Documents/work/github_repos_NO_JOKES/sumo-rl/nets/2way-single-intersection/single-intersection.net.xml',
        'route_file': '/home/beimukvo/Documents/work/github_repos_NO_JOKES/sumo-rl/nets/2way-single-intersection/single-intersection-gen.rou.xml',
        'use_gui': False,
        'single_agent': True,
        'num_seconds': 900,
        'sumo_warnings': False,
    },
)

# Mountain Car
# ----------------------------------------
register(
    'MountainCarMetaLearning-v0',
    entry_point='maml_rl.envs.mountain_car:MountainCarMetaLearning',
    kwargs={'low': 0.0, 'high': 0.6},
    max_episode_steps=200,
    reward_threshold=-110,
    order_enforce=True,
    nondeterministic=False
)

# Cart Pole
# ----------------------------------------
register(
    'CartPoleMetaLearning-v1',
    entry_point='maml_rl.envs.cartpole:CartPoleMetaLearning',
    order_enforce=True,
    reward_threshold=475.0,
    nondeterministic=False,
    max_episode_steps=500,

)
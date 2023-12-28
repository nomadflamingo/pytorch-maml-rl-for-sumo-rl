from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='maml_rl.envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='maml_rl.envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntVelEnv'}
)

register(
    'AntDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntDirEnv'}
)

register(
    'AntPos-v1',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.ant:AntPosEnv'}
)

register(
    'HalfCheetahVel-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
    'HalfCheetahDir-v2',
    entry_point='maml_rl.envs.utils.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'maml_rl.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='maml_rl.envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

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
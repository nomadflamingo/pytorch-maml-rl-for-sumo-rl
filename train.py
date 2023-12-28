import gym
import torch
import json
import os
import yaml
from tqdm import trange

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns
import plotter

import matplotlib.pyplot as plt

from datetime import datetime


def main(args):
    # read config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # make paths
    curr_time = datetime.now().strftime('%d_%H:%M:%S')
    output_folder = f"outputs/{config['env-name']}_maml_at_{curr_time}_seed_{args.seed}"
    metalearner_updates_folder = os.path.join(output_folder, 'metalearner_updates')
    os.makedirs(metalearner_updates_folder)
        
    # define filenames
    policy_filename = os.path.join(output_folder, 'policy.th')
    config_filename = os.path.join(output_folder, 'config.json')
    metalearner_updates_filename = os.path.join(metalearner_updates_folder, 'metalearner_updates')
    returns_graph_filename = os.path.join(output_folder, 'returns.png')
    train_returns_log_filename = os.path.join(output_folder, 'train_returns.log')
    valid_returns_log_filename = os.path.join(output_folder, 'valid_returns.log')

    # Save config
    with open(config_filename, 'w') as f:
        config.update(vars(args))
        json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config.get('env-kwargs', {}),
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    reward_fns = config['reward_fns']
    train_returns = {key: [] for key in reward_fns}
    valid_returns = {key: [] for key in reward_fns}
    env.unwrapped.register_reward_fns(reward_fns)
    
    # create plots
    fig, axs = plotter.create_fig(len(reward_fns), config['env-name'])
    plotter.set_titles(axs, reward_fns)
    lines_train, lines_valid = plotter.create_lines(axs)
    plt.show(block=False)

    num_iterations = 0

    for batch in trange(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=get_returns(train_episodes[0]),
                    valid_returns=get_returns(valid_episodes))
        
        for task, train_return, valid_return in zip(logs['tasks'], logs['train_returns'], logs['valid_returns']):
            train_returns[task].append(train_return[0])
            valid_returns[task].append(valid_return[0])

        # update plots and render
        plotter.set_lines_data(lines_train, train_returns, reward_fns)
        plotter.set_lines_data(lines_valid, valid_returns, reward_fns)
        plotter.render(axs)
        
        # Save policy
        if output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)

        # Save picture
        plt.savefig(returns_graph_filename)

        # Save train returns, valid returns text data
        with open(train_returns_log_filename, 'a') as t, open(valid_returns_log_filename, 'a') as v:
            for task, train_return, valid_return in zip(logs['tasks'], logs['train_returns'], logs['valid_returns']):
                t.write(f"{task} {train_return[0]}\n")
                v.write(f"{task} {valid_return[0]}\n")

        # Save metalearner updates
        with open(f'{metalearner_updates_filename}_{batch:04d}.th', 'wb') as f:
            torch.save(logs, f)

        


if __name__ == '__main__':
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')

    args = parser.parse_args()
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')

    main(args)

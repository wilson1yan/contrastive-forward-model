import argparse
import json
import math
import time
import os
from os.path import join, exists
import itertools
from tqdm import tqdm
import numpy as np
import imageio
import multiprocessing as mp
import sys

from cfm.env.dm_control_env import DMControlEnv


def worker(worker_id, start, end):
    np.random.seed(worker_id+1)
    # Initialize environment
    env = DMControlEnv(**env_args)
    if worker_id == 0:
        pbar = tqdm(total=end - start)

    for i in range(start, end):
        str_i = str(i)
        run_folder = join(root, 'run{}'.format(str_i.zfill(5)))
        if not exists(run_folder):
            os.makedirs(run_folder)

        o = env.reset()
        imageio.imwrite(join(run_folder, 'img_{}_{}.png'.format('0'.zfill(2), '0'.zfill(3))), o.pixels.astype('uint8'))
        actions = []
        env_states = [env.get_state()]
        for t in itertools.count(start=1):
            saved_state = env.get_state(ignore_step=False)
            str_t = str(t)
            actions_t = []

            a = env.action_space.sample()
            actions_t.insert(0, np.concatenate((o.location[:2], a)))
            o, _, terminal, info = env.step(a)
            env_states.append(env.get_state())

            imageio.imwrite(join(run_folder, 'img_{}_{}.png'.format(str_t.zfill(2), '0'.zfill(3))), o.pixels.astype('uint8'))

            actions.append(np.stack(actions_t, axis=0))
            if terminal or info.traj_done:
                break
        env_states = np.stack(env_states, axis=0)
        actions = np.stack(actions, axis=0)
        np.save(join(run_folder, 'actions.npy'), actions)
        np.save(join(run_folder, 'env_states.npy'), env_states)

        if worker_id == 0:
            pbar.update(1)
    if worker_id == 0:
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_dr', action='store_true', help='Add domain randomization in the simulation')
    parser.add_argument('--domain', type=str, default='rope', help='rope|cloth (default: rope)')
    parser.add_argument('--n_samples', type=int, default=200000,
                        help='Number of simulation samples (i.e. observations / transitions) to collect')
    parser.add_argument('--traj_length', type=int, default=10, help='Trajectory length for each episode')
    args = parser.parse_args()

    assert args.domain in ['rope', 'cloth'], f'Invalid domain: {args.domain}'

    start = time.time()
    name = f'{args.domain}_dr_{args.use_dr}'
    root = join('data', name)
    if not exists(root):
        os.makedirs(root)

    n_trajectories = math.ceil(args.n_samples / (args.traj_length + 1))

    env_args = dict(
        domain=f'{args.domain}_dr',
        task='easy',
        max_path_length=args.traj_length,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        task_kwargs=dict(random_pick=True, use_dr=args.use_dr, init_flat=False)
    )

    with open(join(root, 'env_args.json'), 'w') as f:
        json.dump(env_args, f)

    n_chunks = mp.cpu_count()
    partition_size = math.ceil(n_trajectories / n_chunks)
    args_list = []
    for i in range(n_chunks):
        args_list.append((i, i * partition_size, min((i + 1) * partition_size, n_trajectories)))
    print('args', args_list)

    ps = [mp.Process(target=worker, args=args) for args in args_list]
    [p.start() for p in ps]
    [p.join() for p in ps]

    elapsed = time.time() - start
    print('Finished in {:.2f} min'.format(elapsed / 60))

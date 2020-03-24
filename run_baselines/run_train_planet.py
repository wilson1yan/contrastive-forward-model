import argparse
import math
import multiprocessing as mp
import shlex
import subprocess
import os
from cfm.utils import construct_run_command, construct_variants

def worker(gpu_id, max_per_gpu, exps):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    processes = []
    for exp in exps:
        exp['id'] = 'planet_' + exp['id']
        args = construct_run_command('cfm/baselines/train_planet.py', exp)
        print('Running', args)
        args = shlex.split(args)
        processes.append(subprocess.Popen(args, env=env))

        if len(processes) >= max_per_gpu:
            [p.wait() for p in processes]
            processes = []

    [p.wait() for p in processes]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--max_per_gpu', type=int, default=2)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    exps = [{'root': ['data/rope'],
             'state-size': [8], 'chunk-size': [10], 'overshooting-distance': [10],
             'overshooting-kl-beta': [0], 'global-kl-beta': [0], 'trans_type': ['mlp'],
             'embedding-size': [512], 'hidden-size': [128]
            },
            {'seed': [0, 1, 2, 3]}]

    exps = construct_variants(exps, name_key='id')

    if args.end == -1:
        args.end = len(exps)
    print(f'Running {args.end - args.start} / {len(exps)} experiments')
    exps = exps[args.start:args.end]
    n_exps = len(exps)
    chunk_size = math.ceil(n_exps / args.n_gpus)
    worker_args = []
    for i in range(args.n_gpus):
        start, end = chunk_size * i, min(chunk_size * (i + 1), n_exps)
        worker_args.append((i, args.max_per_gpu, exps[start:end]))
    workers = [mp.Process(target=worker, args=arg) for arg in worker_args]
    [w.start() for w in workers]
    [w.join() for w in workers]

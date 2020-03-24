import argparse
import math
import multiprocessing as mp
import shlex
import subprocess
import os
from cpc_util import construct_run_command, construct_variants


def worker(gpu_id, max_per_gpu, exps):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    processes = []
    for exp in exps:
        exp['name'] = 'autoencoder_' + exp['name']
        args = construct_run_command('cfm/baselines/train_autoencoder.py', exp)
        print('Running', args)
        args = shlex.split(args)
        processes.append(subprocess.Popen(args, env=env))

        if len(processes) >= max_per_gpu:
            [p.wait() for p in processes]
            processes = []

    [p.wait() for p in processes]


def get_z(folder):
    if 'rope' in folder or 'cloth' in folder:
        return 8
    else:
        raise Exception(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=4)
    parser.add_argument('--max_per_gpu', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    folders = ['data/rope']
    exps = [dict(root=folders, z_dim=[get_z(f) for f in folders]),
            dict(trans_type=['reparam_w_tanh'], batch_size=[128], lr=[1e-3], epochs=[20]),
            dict(seed=[0, 1, 2, 3])]

    exps = construct_variants(exps, name_key='name')

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

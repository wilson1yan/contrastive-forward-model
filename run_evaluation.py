import itertools
import torch
import sys
import math
import multiprocessing as mp
import shlex
import subprocess
import os
from cfm.utils import construct_variants, construct_run_command

def worker(gpu_id, exps):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    processes = []
    for exp in exps:
        args = construct_run_command('cfm/evaluate_planning.py', exp)
        print('Running', args)
        args = shlex.split(args)
        processes.append(subprocess.Popen(args, env=env))

        if len(processes) >= 1:
            [p.wait() for p in processes]
            processes = []


    [p.wait() for p in processes]


def get_mode(folder):
    if 'rope' in folder:
        mode = 'rope'
    elif 'cloth' in folder:
        mode = 'cloth'
    else:
        raise Exception(folder)
    return mode


def get_n_actions(folder):
    if 'rope' in folder:
        n_actions = 20
    elif 'cloth' in folder:
        n_actions = 40
    else:
        raise Exception(folder)
    return n_actions



if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    folders = sys.argv[1:]
    exps = [dict(folder=folders, mode=[get_mode(f) for f in folders],
                 n_actions=[get_n_actions(f) for f in folders]),
            dict(goal_type=['random', 'flat']), dict(n_trials=[1000])]
    exps = construct_variants(exps, name_key=None)

    chunk_size = math.ceil(len(exps) / n_gpus)
    worker_args = []
    for i in range(n_gpus):
        start, end = chunk_size * i, min(chunk_size * (i + 1), len(exps))
        worker_args.append((i, exps[start:end]))
    workers = [mp.Process(target=worker, args=arg) for arg in worker_args]
    [w.start() for w in workers]
    [w.join() for w in workers]

import sys
from os.path import join
import json
import glob
import numpy as np

roots = sys.argv[1:]
for root in roots:
    folders = glob.glob(join(root, '*', 'eval', '*'))
    stats = dict()
    for folder in folders:
        with open(join(folder, 'eval_results.json')) as f:
            results = json.load(f)
        val = results['geom']
        key = results['goal_type']
        if key not in stats:
            stats[key] = []
        stats[key].append(val)

    print(root)
    for k, v in stats.items():
        min, max = np.min(v), np.max(v)
        std = np.std(v)
        mean = np.mean(v)

        print(f'\t{k}: {min:.3f}/{max:.3f}/{mean:.3f}/{std:.3f} (Min/Max/Mean/Std)')
    print()

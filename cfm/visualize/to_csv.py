import json
import glob
from os.path import join
import pandas as pd
import sys

roots = sys.argv[1:]
metrics = ['xor', 'iou', 'geom']
for root in roots:
    folders = glob.glob(join(root,'*', 'eval', '*'))
    for folder in folders:
        with open(join(folder, 'eval_results.json'), 'r') as f:
            results = json.load(f)
        df_data = dict()
        for metric in metrics:
            df_data[metric] = [results[metric]] * 10
            del results[metric]

        with open(join(folder, 'params.json'), 'w') as f:
            json.dump(results, f)

        progress = pd.DataFrame(df_data)
        progress.to_csv(join(folder, 'progress.csv'), index=False)


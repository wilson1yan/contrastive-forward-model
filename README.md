# Contrastive Forward Model

This is code to reproduce experiments for the paper [Learning Predictive Representations for Deformable Objects Using Contrastive Estimation](https://arxiv.org/abs/2003.05436).

## Installation
This project was run using Python 3.7.6 with PyTorch 1.4.0. You will also need to install a [custom dm_control package](https://github.com/wilson1yan/dm_control/tree/cfm) with the relevant rope and cloth environments. 

You will also need to install the repo as a pip package: `cd contrastive-forward-model; pip install -e .`

## Running
The steps to collect and run data are as follows. You may use the `-h` flag to show more customizable options.
1. Collect data by running `python sample_trajectories.py`
2. Process the data using `python process_dataset.py data/rope`
3. Train CFM with `python run_train.py`. You can customize your own flags to run it with different hyperparameters. The output is stored int the `out/` folder
4. Run the evaluations with `python run_evaluation.py out/*`, which will generate json files and store them in `out/<exp_name>/eval/<eval_name>/eval_results.json`

## Visualization
There are two ways to visualize your results. If you group up your result folders by seed, and store them in a single file, i.e. `tmp`, you may call `python cfm/visualize/print_evaluation_stats.py tmp`, which will print out the results in a formatted manner, with standard statistics across seds.

If you are performing hyperparmeters tuning, it may be easier to run `python cfm/visualize/to_csv.py out`, which sill generate `progress.csv` and `params.json` files in each eval directory. Then, you can use the [rllab viskit](https://github.com/vitchyr/viskit) library to view: `python <path to viskit>/viskit/frontend.py out`, where you can split by different hyperparameters and average over seeds.


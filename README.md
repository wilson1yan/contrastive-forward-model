# Contrastive Forward Model

This is code to reproduce experiments for the paper [Learning Predictive Representations for Deformable Objects Using Contrastive Estimation](https://arxiv.org/abs/2003.05436).

## Installation
This project was run using Python 3.7.6. All the dependencies are in the `requirements.txt` file and we recommend creating a virtual environment and then installing by `pip install -r requirements.txt`.

You will also need to install a custom [dm_env package](https://github.com/wilson1yan/dm_env) and a custom [dm_control package](https://github.com/wilson1yan/dm_control/tree/cfm) which has the relevant rope and cloth environments. You **must** use the **cfm** branch in the custom `dm_control` repo. Note that `dm_control` requires the Mujoco simulator to use.  Finally, you will need to install this repo as a pip package: `cd contrastive-forward-model; pip install -e .`

## Running
The steps to collect and run data are as follows. You may use the `-h` flag to show more customizable options.
1. Collect data by running `python sample_trajectories.py`
2. Process the data using `python process_dataset.py data/rope`
3. Train CFM with `python run_train.py`. You can customize your own flags to run it with different hyperparameters. The output is stored in the `out/` folder
4. Run the evaluations with `python run_evaluation.py out/*`, which will generate json files and store them in `out/<exp_name>/eval/<eval_name>/eval_results.json`

## Visualization
There are two ways to visualize your results. If you group up your result folders by seed, and store them in a single file, i.e. `tmp`, you may call `python cfm/visualize/print_evaluation_stats.py tmp`, which will print out the results in a formatted manner, with standard statistics across seeds.

If you are performing hyperparmeters tuning, it may be easier to run `python cfm/visualize/to_csv.py out`, which will generate `progress.csv` and `params.json` files in each eval directory. Then, you can use the [rllab viskit](https://github.com/vitchyr/viskit) library to view: `python <path to viskit>/viskit/frontend.py out`, where you can split by different hyperparameters and average over seeds.

## Baselines
You can run the baselines by executing `python run_baselines/run_train_<baseline_name>.py` for step 3 instead of the CFM script. The rest of the steps are identical.

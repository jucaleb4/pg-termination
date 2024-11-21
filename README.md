# Policy Gradient Termination
We have a simple script to automatically generate settings, run, and plot.

## Experiements for strongly-polynomial PMD
```
python script/2024_10_15/exp_0.py --setup --mode full # use --mode work to make sure it runs with 1 seed
python script/2024_10_15/exp_0.py --run # -use --sub_runs x,y to run experiments [x,y).
```

The data is saved in the folder `logs/2024_10_15/exp_0`, and there should be 63 total experiments.
You can see `settings/2024_10_15/exp_0` for the configuration of each experiment.

## Step size tuning for stochastic PMD
For gamma=0.9, run
```
python script/2024_10_19/exp_0.py --setup
python script/2024_10_19/exp_0.py --run
```

For **gridworld**, a constant stepsize with eta=1 works best.
For **taxi**, a constant stepsize with eta=0.1 works best.

You can repeat with gamma=0.99, 
```
python script/2024_10_19/exp_2.py --setup
python script/2024_10_19/exp_2.py --run
```
although we found the same step sizes from gamma=0.9 work well here.

## Run experiments
For gamma=0.9
```
python script/2024_10_19/exp_1.py --setup --mode full # change full->test to get it working
python script/2024_10_19/exp_1.py --run
```

For gamma=0.99
```
python script/2024_10_19/exp_2.py --setup --mode full # change full->test to get it working
python script/2024_10_19/exp_2.py --run
```

## Printing and plotting results
To create the plots (for online validation analysis):
```
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.9 --plot
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.99 --plot
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.9 --plot
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.99 --plot
```

To create the tables (for offline validation analysis):
```
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.9
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.99
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.9
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.99
```

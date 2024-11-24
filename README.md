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

The table below shows the different environments, discount factors, and tuned
step size.  You can run this yourself by replacing the "0" in exp_0 from the
script above with the "exp_id" given in the table.

We selected the step size based on the final policy's offline validation
analysis step.  Our report is given below:

| Gamma | Environment | Stepsize | eta | Score | exp_id | 
| ----- | ----------- | -------- | --- | ----- | ------ | 
| 0.9   | Gridworld   | Constant |   1 |  5.19 |      0 | 
| 0.9   | Taxi        | Constant | 0.1 | -3.09 |      0 |  
| 0.95  | Gridworld   | Constant |   1 |  7.94 |      4 | 
| 0.95  | Taxi        | Constant | 0.1 | -10.3 |      4 | 
| 0.99  | Gridworld   | Constant | 0.1 |  38.6 |      2 | 
| 0.99  | Taxi        | Constant | 0.1 |  70.7 |      2 | 

## Run experiments
For gamma=0.9
```
python script/2024_10_19/exp_1.py --setup --mode full 
python script/2024_10_19/exp_1.py --run
```
If you only want to see if the code runs, change the `--mode` flag to "test".
If you have parallel processors, add a `--parallel` flag after the `--run` flag.

You can repeat the same experiment for different discount factors by changing the exp_1 to:
- exp_3: gamma=0.99
- exp_5: gamma=0.95

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
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.9 --print
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.95 --print
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.99 --print
```

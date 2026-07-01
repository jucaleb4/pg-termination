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
python script/2024_10_19/exp_1.py --setup --mode full # change full->test to get it working
python script/2024_10_19/exp_1.py --run
```

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
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.9
python plot/parse_2024_10_19.py --env_name gridworld_small --gamma 0.99
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.9
python plot/parse_2024_10_19.py --env_name taxi --gamma 0.99
```

# Autoexploration

## Overview of experiments

- `2025_12_05/exp_0.py`: initial testing of fixed, estimated, dynamic MC, and CTD on 5x5 GridWorld
- `2025_12_11/exp_0.py`: setup simple BatteryModel
- `2025_12_11/exp_1.py`: successive hyper-tuning (one with noisy oracle, one with exact oracle)
- `2025_12_11/exp_2.py`: running with best fixed exploration time from `exp_1.py` above
- `2025_12_24/exp_0.py`: testing CTD on discretized MountainCar

- `2026_04_08/exp_0.py`: tune SPMD for GARNET (5-100)
- `2026_04_08/exp_1.py`: opt on GARNET
- `2026_04_08/exp_2.py`: full SPMD on GARNET (*DEPRECEATED* use 2026_05_05/exp_0.py)
- `2026_04_09/exp_0.py`: tune SPMD for GridWorld *DEPRECEATED*
- `2026_04_09/exp_1.py`: GridWorld OPT *DEPRECEATED*
- `2026_04_09/exp_2.py`: full SPMD on GridWorld *DEPRECEATED*
- `2026_04_13/exp_0.py`: Initial ppo_clean
- `2026_04_14/exp_0.py`: tune SPMD+CTD on GridWorld *DEPRECEATED*
- `2026_04_15/exp_0.py`: tune SPMD on GridWorld 
- `2026_04_15/exp_1.py`: GridWorld OPT
- `2026_04_15/exp_2.py`: full SPMD on GridWorld 
- `2026_04_16/exp_0.py`: tune Q-learn on GridWorld (*DEPRECEATED* since no more Q-learn)
- `2026_04_16/exp_1.py`: opt on GridWorld (*DEPRECEATED* since no more Q-learn)
- `2026_04_16/exp_2.py`: full Q-learn on GridWorld (*DEPRECEATED* since no more Q-learn)
- `2026_04_25/exp_0.py`: tune SARSA on Garnet (*DEPRECEATED* since duplicate)
- `2026_04_25/exp_2.py`: SARSA on Garnet (*DEPRECEATED* since duplicate)
- `2026_05_05/exp_0.py`: SPMD full run on Garnet (w/ min sample count) (*DEPRECEATED* since duplicate)
- `2026_05_05/exp_1.py`: SARSA full run on Garnet (w/ min sample count) (*DEPRECEATED* since duplicate)
- `2026_05_05/exp_2.py`: SPMD full run on GridWorld (w/ min sample count) (*DEPRECEATED* since duplicate)
- `2026_05_05/exp_3.py`: SARSA full run on GridWorld (w/ min sample count) (*DEPRECEATED* since duplicate)
- `2026_05_06/exp_0.py`: tune SARSA on Garnet
- `2026_05_06/exp_1.py`: full SARSA run on Garnet
- `2026_05_06/exp_2.py`: tune SARSA on GridWorld
- `2026_05_06/exp_3.py`: full SARSA run on GridWorld
- `2026_05_06/exp_4.py`: full SARSA run on GridWorld-loop
- `2026_05_18/exp_0.py`: tune SPMD+CTD on GridWorld 
- `2026_05_18/exp_1.py`: full run SPMD+CTD on GridWorld 
- `2026_05_18/exp_2.py`: full run SPMD+CTD on GridWorld (aggressive parameters) 
- `2026_05_19/exp_0.py`: tune SPMD on GARNET with Tsallis
- `2026_05_19/exp_1.py`: tune SPMD on GridWorld with Tsallis
- `2026_05_19/exp_2.py`: full SPMD on GARNET with Tsallis
- `2026_05_19/exp_3.py`: full SPMD on GridWorld with Tsallis
- `2026_05_19/exp_4.py`: full SPMD on GridWorld-loop with Tsallis
- `2026_05_19/exp_5.py`: Refined tuning-v1 SPMD-Dyn on GARNET
- `2026_05_19/exp_6.py`: Refined tuning-v1 SPMD-Dyn on GridWorld
- `2026_05_19/exp_7.py`: Full-run SPMD-Dyn on GARNET from Refined tuning-v1
- `2026_05_19/exp_8.py`: Full-run SPMD-Dyn on GridWorld from Refined tuning-v1
- `2026_05_20/exp_0.py`: GARNET OPT
- `2026_05_20/exp_1.py`: GridWorld OPT (TBD)
- `2026_05_31/exp_0.py`: Full-tune of SPMD+CTD on GARNET
- `2026_05_31/exp_1.py`: Full-tune of SPMD+CTD on GridWorld
- `2026_05_31/exp_2.py`: Full-tune of SPMD+CTD on GridWorld-loop
- `2026_05_31/exp_3.py`: Full-tune of SPMD+CTD on GARNET - more feature tuning
- `2026_05_31/exp_4.py`: Full-tune of SPMD+CTD on GridWorld - more feature tuning
- `2026_05_31/exp_5.py`: full (via enhanced tuned) SPMD+CTD on GARNET 
- `2026_05_31/exp_6.py`: full (via enhanced tuned) SPMD+CTD on GridWorld
- `2026_05_31/exp_7.py`: shorter full (via enhanced tuned) SPMD+CTD on GARNET (*DEPRECEATED* since unsure what we showed)
- `2026_05_31/exp_8.py`: shorter full (via enhanced tuned) SPMD+CTD on GridWorld (*DEPRECEATED* since unsure what we showed)
- `2026_05_31/exp_9.py`: full (via enhanced tuned from exp_6) SPMD+CTD on GridWorld-loop
- `2026_05_31/exp_10.py`: Full tune of SPMD+CTD on GARNET - enhanced-v2 (new multipliers, mini-batch)
- `2026_05_31/exp_11.py`: Full tune of spmd+ctd on gridworld - enhanced-v2 (new multipliers, mini-batch)
- `2026_05_31/exp_12.py`: Full-run SPMD+CTD on GARNET using enhanced-v2 
- `2026_05_31/exp_13.py`: Full-run SPMD+CTD on Gridworld using  enhanced-v2 
- `2026_05_31/exp_14.py`: Refined tuning-v1 SPMD+CTD on GARNET 
- `2026_05_31/exp_15.py`: Refined tuning-v1 SPMD+CTD on gridworld 
- `2026_05_31/exp_16.py`: Full-run SPMD+CTD on GARNET from refined tuning-v1
- `2026_05_31/exp_17.py`: Full-run SPMD+CTD on GridWorld from refined tuning-v1
- `2026_05_31/exp_18.py`: Refined tuning-v2 SPMD+CTD on GARNET 
- `2026_05_31/exp_19.py`: Refined tuning-v2 SPMD+CTD on gridworld 
- `2026_05_31/exp_20.py`: Full-run SPMD+CTD on GARNET from refined tuning-v2
- `2026_05_31/exp_21.py`: Full-run SPMD+CTD on GridWorld from refined tuning-v2
- `2026_06_01/exp_0.py`: Baselines (uniform random policy) for GARNET, GW, and GridWorld-lodim
- `2026_06_07/exp_0.py`: tune SPMD on GridWorld-lowdim
- `2026_06_07/exp_1.py`: tune SARSA on GridWorld-lowdim
- `2026_06_07/exp_2.py`: Full tune SPMD+CTD on GridWorld-lowdim - enhanced-v2 
- `2026_06_12/exp_0.py`: Refined tuning-v3 SPMD+CTD on GARNET (zero func-err)
- `2026_06_12/exp_1.py`: Refined tuning-v3 SPMD+CTD on GridWorld (zero func-err)
- `2026_06_12/exp_2.py`: Full run of SPMD+CTD on GARNET with refined tuning-v3
- `2026_06_12/exp_3.py`: Full run of SPMD+CTD on GridWorld with refined tuning-v3
- `2026_06_12/exp_4.py`: Manual full run of GARNET (derivative of exp_2.py)
- `2026_06_14/exp_0.py`: Tuning SPMD+CTD comparisons on large GARNETs (s_origin = 'reset')
- `2026_06_14/exp_1.py`: Tuning SPMD+CTD comparisons on large GARNETs (s_origin = None)
- `2026_06_14/exp_2.py`: Full run SPMD+CTD-estimate on large GARNETs (includes estimate and reset)
- `2026_06_15/exp_0.py`: Full run of SPMD+CTD-estimate on GARNET with refined tuning-v3
- `2026_06_15/exp_1.py`: Full run of SPMD+CTD-estimate on GridWorld with refined tuning-v3
- `2026_06_30/exp_0.py`: Tune SPMD MC-Dyn on Discretized-Inventory 
- `2026_06_30/exp_1.py`: Tune SPMD MC+CTD-Dyn on Discretized-Inventory 
- `2026_06_30/exp_2.py`: Tune PPO on Discretized-Inventory 
- `2026_07_01/exp_0.py`: Tune SPMD MC-Dyn on Discretized-CartPole
- `2026_07_01/exp_1.py`: Tune SPMD MC+CTD-Dyn on Discretized-CartPole
- `2026_07_01/exp_2.py`: Tune PPO on Discretized-CartPole

## TODOs
- Implement CTD (general state)
- Implement PPO, DQN

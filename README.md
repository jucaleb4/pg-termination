# Stochastic policy mirror descent experiments
This code provides reproductable experiments in a scripting-based manner for (stochastic) policy mirror papers.
The two papers included so far are:

1. [Strongly-polynomial](#strong-polynomial-runtime)
2. [Auto-explore](#auto-exploration)

## Setup
We use pip. For essentials installation, run
```
pip install -r requirements.txt
```

Note that this package requires an older version of NumPy due to depreceation of the `reshape` function (see this [GitHub issues](https://github.com/google/meridian/issues/1427)).
The `requirements.txt` includes a working version, but beware if you are using your own version.

Experiment for [inventory](#inventory) requires a separate installation.
We defer the details therein.

## Strong-polynomial runtime 

### Tuning Step size tuning for stochastic PMD
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


### Run experiments
For gamma=0.9
```
python script/2024_10_19/exp_1.py --setup --mode full # change full->test to get it working
python script/2024_10_19/exp_1.py --run
```
The data is saved in the folder `logs/2024_10_15/exp_0`, and there should be 63 total experiments.
You can see `settings/2024_10_15/exp_0` for the configuration of each experiment.

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

### Source and Citation
Paper DOI: https://doi.org/10.1007/s10107-026-02356-y

This code is freely available for everyone to use. If you would like to acknowledge the usage of this code, please cite the following paper:

Ju, Caleb, and Guanghui Lan. "Strongly-polynomial time and validation analysis of policy gradient methods." Mathematical Programming (2026): 1-45.

## Auto-exploration 

### Re-create results from paper
We ran on four environments: GARNET, GridWorld, Inventory (newsvendor), and CartPole.

#### GARNET
The following creates scripts, runs them, and saves the logs for GARNET.
```
# SARSA
python script/2026_05_06/exp_1.py --setup --mode full
python script/2026_05_06/exp_1.py --run

# SPMD MC-Est
python script/2026_05_19/exp_2.py --setup --mode full
python script/2026_05_19/exp_2.py --run

# SPMD MC-Dyn
python script/2026_05_19/exp_7.py --setup --mode full
python script/2026_05_19/exp_7.py --run

# SPMD+CTD-Est
python script/2026_06_15/exp_0.py --setup --mode full
python script/2026_06_15/exp_0.py --run

# SPMD+CTD-Dyn
python script/2026_05_31/exp_16.py --setup --mode full
python script/2026_05_31/exp_16.py --run

# SPMD+CTD-Dyn (extra experiments)
python script/2026_05_31/exp_20.py --setup --mode full
python script/2026_05_31/exp_20.py --run
```

All experiments are repeated over 10 different seeds.
If you prefer just one seed, then remove the `--mode full` flag.
Beware this will cause some issues with the plotting code later.
Additionally, when running over 10 seeds (and access to 10+ cores), you can parallelize by adding a `--parallel` flag after `--run`, such as:

```
python script/2026_05_06/exp_1 --run --parallel
```

#### GridWorld
```
# SARSA
python script/2026_05_06/exp_3.py --setup --mode full
python script/2026_05_06/exp_3.py --run

# SPMD MC-Est
python script/2026_05_19/exp_3.py --setup --mode full
python script/2026_05_19/exp_3.py --run

# SPMD MC-Dyn
python script/2026_05_19/exp_8.py --setup --mode full
python script/2026_05_19/exp_8.py --run

# SPMD+CTD-Est
python script/2026_06_15/exp_1.py --setup --mode full
python script/2026_06_15/exp_1.py --run

# SPMD+CTD-Dyn
python script/2026_05_31/exp_17.py --setup --mode full
python script/2026_05_31/exp_17.py --run
```

#### Inventory
Before you can setup the Inventory environment, you first need Gymnasium.
See the instructions from [CartPole](#cartPole).
Once Gymnasium is installed, you will need to install a custom Gymnasium package.
This is straightforward:
1. Clone our customized [gym_examples repo](https://github.com/jucaleb4/gym-examples#setting-up-batteryenv-with-gymnasium). More info in the README therein, which can be safely skipped for now.
2. After cloning, go into the root directory of `gym_examples` and run `pip install -e .`.

After installing gym_examples, you can run the following scripts.
```
# SARSA
python script/2026_07_02/exp_0.py --setup --mode full
python script/2026_07_02/exp_0.py --run

# SPMD+CTD-Est
python script/2026_07_02/exp_1.py --setup --mode full
python script/2026_07_02/exp_1.py --run

# SPMD MC-Dyn
python script/2026_07_02/exp_2.py --setup --mode full
python script/2026_07_02/exp_2.py --run
```

#### CartPole
To run [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), you first need to install Gymnasium. This is straightforward with pip and the following steps:
1. Run `pip install "gymnasium[classic-control]"`
See the [Gymnasium documentation](https://gymnasium.farama.org) for more info.

After installing Gymnasium, you can run the following scripts.

```
# SARSA
python script/2026_07_03/exp_0.py --setup --mode full
python script/2026_07_03/exp_0.py --run

# SPMD+CTD-Est
python script/2026_07_03/exp_1.py --setup --mode full
python script/2026_07_03/exp_1.py --run

# SPMD MC-Dyn
python script/2026_07_03/exp_2.py --setup --mode full
python script/2026_07_03/exp_2.py --run
```

#### Baslines
This code provides baselines for a uniform random policy to compare against.
Make sure you follow the same installations as the subsections above for Inventory and CartPole.
```
# GARNET and GridWorld
python script/2026_06_01/exp_0.py --setup --mode full
python script/2026_06_01/exp_0.py --run

# Inventory
python script/2026_06_01/exp_1.py --setup --mode full
python script/2026_06_01/exp_1.py --run

# CartPole
python script/2026_06_01/exp_2.py --setup --mode full
python script/2026_06_01/exp_2.py --run
```

### Plotting
We also included Jupyter notebooks to create plots in the paper after completing the experiments.

1. GARNET: `plot/2026_06_10.ipynb`
1. GridWorld: `plot/2026_06_10.ipynb`
1. Inventory: `plot/2026_07_02.ipynb`
1. CartPole: `plot/2026_07_03.ipynb`

Figures are saved in separate folders in the `plot` folder.

### Source and Citation
Paper link: https://arxiv.org/abs/2512.06244 

This code is freely available for everyone to use. If you would like to acknowledge the usage of this code, please cite the following paper:

Ju, Caleb, and Guanghui Lan. "Auto-exploration for online reinforcement learning." arXiv preprint arXiv:2512.06244 (2025).

# All experiments (internal use)
See ALLEXPERIMENTS.txt.
This is to track experiments (including failed experiments) and more for internal use.
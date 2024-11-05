import os
import sys

import argparse
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, Normalize

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pg_termination import wbmdp 

DATE = "2024_10_15"
EXP_ID = 0
N_RUNS = 63
N_SEEDS = 10

def print_worse_case_complexity(env_name, gamma):
    env = wbmdp.get_env(env_name, 0.9, 0)
    N = np.ceil(4./(1-gamma))
    T = np.ceil(np.log(env.n_states**3*env.n_actions/(1.-gamma)**2/np.log(2)))+1
    print("Worst case iteration complexity: %d" % (env.n_states*(env.n_actions-1)*N*T))

def read_alg_performance(fname):
    """ Read seed=%d.csv files """
    if not os.path.exists(fname):
        return None

    df = pd.read_csv(fname, header="infer")
    data = df.to_numpy()
    return data

def read_all_final_alg_performances():
    """ 
    Returns 3D tensor, where 1st index is run_id, 2nd is seed_id, and third
    contains data for: 
    [iter,average value,final advantage gap,used greedy for final]
    """

    folder = os.path.join("logs", DATE, "exp_%d" % EXP_ID)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    data = np.zeros((N_RUNS, N_SEEDS, 4), dtype=float)
    for run_id in range(N_RUNS):
        missing_files = False
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "seed=%d.csv" % seed_id)
            exp_data = read_alg_performance(fname)
            if exp_data is None:
                missing_files = True
            elif exp_data.shape[1] == 3:
                # policyiter only uses greedy
                data[run_id, seed_id,:3] = exp_data[-1]
                data[run_id, seed_id,3] = True
            else:
                # pmd uses either randomized or greedy to termiante
                use_greedy = exp_data[-1,2] >= exp_data[-1,3]
                data[run_id, seed_id,:2] = exp_data[-1,:2]
                data[run_id, seed_id,2] = exp_data[-1,3 if use_greedy else 2]
                data[run_id, seed_id,3] = use_greedy

        if missing_files:
            print("Missing some seed files for run_id=%d" % run_id)

    return data

def print_final_convergence_result(env_name):
    data = read_all_final_alg_performances()

    env_name_arr = [
        "gridworld_small", 
        "gridworld_hill_small", 
        "taxi",
        "random",
        "chain",
    ]

    alg_name_arr = [
        "pmd-eu-basic",
        "pmd-eu-agg",
        "policyiter",
    ]
    gamma_arr = [0.9, 0.99, 0.999]

    found_env_name = False
    for i, name in enumerate(env_name_arr):
        if env_name == name:
            data = data[9*i:9*(i+1)]
            found_env_name = True

    if not found_env_name:
        print("Unknown env_name=%s" % env_name)
        return

    exp_metadata = ["Alg", "Iter", "f(pi)", "gap", "#final greedy"]
    row_format ="{:>15}|" + "{:>25}|" * (len(exp_metadata)-2) + "{:>10}"
    dashed_msg = "-" * (15+25*(len(exp_metadata)-2) + 10 + len(exp_metadata) - 1)
    z = 1.96
    ct = 0

    min_iter_arr  = np.min(data[:,:,0],axis=1)
    max_iter_arr  = np.max(data[:,:,0], axis=1)
    mean_f_pi_arr = np.mean(data[:,:,1],axis=1)
    std_f_pi_arr  = np.std(data[:,:,1], axis=1)
    mean_gap_arr  = np.mean(data[:,:,2],axis=1)
    std_gap_arr   = np.std(data[:,:,2], axis=1)
    num_greedy_arr = np.sum(data[:,:,3],axis=1) 

    for gamma in gamma_arr:
        print("")
        print("Env %s with gamma: %.4f" % (env_name, gamma))
        print_worse_case_complexity(env_name, gamma)
        print("")
        # print(dashed_msg)
        print(row_format.format(*exp_metadata))
        print(dashed_msg)

        for i in range(3):
            print(row_format.format(
                alg_name_arr[i], 
                "[%d,%d]" % (min_iter_arr[ct], max_iter_arr[ct]),
                "%.2e +/- %.2e" % (mean_f_pi_arr[ct], std_f_pi_arr[ct]),
                "%.2e +/- %.2e" % (mean_gap_arr[ct], std_gap_arr[ct]),
                "%d" % num_greedy_arr[ct],
            ))

            # find non-optimal
            if np.sum(data[ct,:,2] > 1e-10):
                where_nonopt = np.where(data[ct,:,2] > 1e-10)[0]
                print("\t- Non-optimal: seeds {}".format(where_nonopt))
                print("\t- final-gap:{}".format(data[ct,where_nonopt,2]))

            ct += 1
        print(dashed_msg)

def get_all_alg_performances():
    """ 
    Returns 3D tensor, where 1st index is run_id, 2nd is seed_id, and third
    contains data for: 
    [iter,average value,final advantage gap,used greedy for final]
    """

    folder = os.path.join("logs", DATE, "exp_%d" % EXP_ID)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    for run_id in range(N_RUNS):
        missing_files = False
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "seed=%d.csv" % seed_id)
            exp_data = read_alg_performance(fname)
            if exp_data is None:
                missing_files = True
            elif exp_data.shape[1] == 3:
                # policyiter only uses greedy
                data[run_id, seed_id,:3] = exp_data[-1]
                data[run_id, seed_id,3] = True
            else:
                # pmd uses either randomized or greedy to termiante
                use_greedy = exp_data[-1,2] >= exp_data[-1,3]
                data[run_id, seed_id,:2] = exp_data[-1,:2]
                data[run_id, seed_id,2] = exp_data[-1,3 if use_greedy else 2]
                data[run_id, seed_id,3] = use_greedy

        if missing_files:
            print("Missing some seed files for run_id=%d" % run_id)

    return data

def plot_fgap_vs_advantage_gap(env_name, seed_id, gamma=0.9, fig_fname=None):
    folder = os.path.join("logs", DATE, "exp_%d" % EXP_ID)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    env_name_arr = [
        "gridworld_small", 
        "gridworld_hill_small", 
        "taxi",
        "random",
        "chain",
    ]

    alg_name_arr = [
        "pmd-kl",
        "pmd-eu",
        "policyiter",
    ]

    found_env_name = False
    for i, name in enumerate(env_name_arr):
        if env_name == name:
            run_id = 9*i
            found_env_name = True

    if not found_env_name:
        print("Unknown env_name=%s" % env_name)
        return

    gamma_arr = np.array([0.9, 0.99, 0.999])
    i = np.where(gamma_arr == gamma)[0]
    if len(i) == 0: 
        print("Did not find gamma %.3f" % gamma)
        return 

    gamma_id = i[0]
    run_id += 3*gamma_id

    fname = os.path.join(folder, "run_%d" % run_id, "seed=%d.csv" % seed_id)
    pmd_kl = pd.read_csv(fname, header="infer").to_numpy()

    fname = os.path.join(folder, "run_%d" % (run_id+1), "seed=%d.csv" % seed_id)
    pmd_eu = pd.read_csv(fname, header="infer").to_numpy()

    fname = os.path.join(folder, "run_%d" % (run_id+2), "seed=%d.csv" % seed_id)
    piter = pd.read_csv(fname, header="infer").to_numpy()

    plt.style.use('ggplot')
    fig, axes = plt.subplots(nrows=2)
    z = 1.96

    xs = pmd_kl[:,0]
    axes[0].plot(xs, pmd_kl[:,1], label="PMD (KL)", linestyle="solid", color="red")
    axes[1].plot(xs, pmd_kl[:,3], linestyle="solid", color="red")

    xs = pmd_eu[:,0]
    axes[0].plot(xs, pmd_eu[:,1], label="PMD (EU)", linestyle="dashed", color="blue")
    axes[1].plot(xs, pmd_eu[:,3], linestyle="dashed", color="blue")

    xs = piter[:,0]
    axes[0].plot(xs, piter[:,1], label="PI", linestyle="dotted", color="black")
    axes[1].plot(xs, piter[:,2], linestyle="dotted", color="black")

    axes[0].legend()
    axes[0].set(
        title="Function and advantage gap convergence\nin %s (gamma=%.3f, seed=%d)" % (env_name, gamma, seed_id),
        ylabel=r"$f(\pi)-f^*$", 
        yscale= "log",
        xlabel="",
    )
    axes[0].set_xticklabels([])
    axes[1].set(
        ylabel=r"$\max_{s \in S} ~g^{\hat{\pi}_k}(s)$", 
        yscale= "log",
        xlabel="Iteration",
    )

    plt.tight_layout()
    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=240)
    else:
        plt.show()

def read_final_dist(fname):
    """ Read pi_seed=%d.csv files """
    if not os.path.exists(fname):
        return None

    df = pd.read_csv(fname, header=None)
    data = df.to_numpy()
    return data

def read_gridworld_target(fname):
    """ Read gridworld_target_seed=%d.csv to find target """
    if not os.path.exists(fname):
        return None

    df = pd.read_csv(fname, header="infer")
    data = df.to_numpy()
    return data

def read_all_final_rho_and_target(env_name):
    """ 
    TODO: 
    """
    env_name_arr = [
        "gridworld_small", 
        "gridworld_hill_small", 
        "taxi",
        "random",
        "chain",
    ]

    folder = os.path.join("logs", DATE, "exp_%d" % EXP_ID)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    found_env_name = False
    run_id_arr = []
    for i, name in enumerate(env_name_arr):
        if env_name == name:
            run_id_arr = list(range(1+9*i,1+9*(i+1),3))
            found_env_name = True

    dist_data = None
    target_arr = np.zeros((3, N_SEEDS), dtype=int)

    for i, run_id in enumerate(run_id_arr):
        missing_files = False
        missing_files_arr = []
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "rho_seed=%d.csv" % seed_id)
            exp_data = read_final_dist(fname)
            if exp_data is None:
                missing_files = True
                missing_files_arr += [seed_id]
            elif dist_data is None:
                dist_data = np.zeros((len(run_id_arr), N_SEEDS, len(exp_data)))
                dist_data[i, seed_id] = np.squeeze(exp_data)
            else:
                dist_data[i, seed_id] = np.squeeze(exp_data)

            # gridworld_target_seed=0.csv
            fname = os.path.join(folder, "run_%d" % run_id, "gridworld_target_seed=%d.csv" % seed_id)
            target_data = read_gridworld_target(fname)
            target_arr[i, seed_id] = target_data

        if missing_files:
            print("Missing some seed files for run_id=%d {}".format(missing_files_arr) % run_id)

    return dist_data, target_arr

def plot_stationary_distribution_heatmap(env_name, gamma=0.9, fig_fname=None):
    assert env_name in ["gridworld_small", "gridworld_hill_small"], "%s not supported, only gridworld supported" % env_name
    dist_data_arr, target_data_arr = read_all_final_rho_and_target(env_name)

    gamma_arr = np.array([0.9, 0.99, 0.999])
    i = np.where(gamma_arr == gamma)[0]
    if len(i) == 0: 
        print("Did not find gamma %.3f" % gamma)
        return 
    i = i[0]

    rho = None
    target = -1
    seed_id = -1
    ct = 0
    for seed in range(N_SEEDS):
        if abs(1. - np.sum(dist_data_arr[i, seed])) < 1e-2:
            ct += 1
            if ct < 2:
                continue
            rho = dist_data_arr[i, seed]
            target = target_data_arr[i, seed]
            seed_id = seed
            print(seed)
            print("Using seed %d" % seed_id)
            break

    if rho is None:
        print("Did not find nonzero distribution")
        exit(0)

    # convert 1d -> 2d
    n = int(len(rho)**0.5)
    rho = np.reshape(rho, newshape=(n,n))

    print("target: (%d,%d)" % (target // n, target % n))
    print("min rho(s)=%.6e" % np.min(rho))

    rho = np.maximum(1e-16, rho)

    # https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap
    ax = sns.heatmap(rho, linewidth=0.5, norm=LogNorm())
    ax.set(
        title="Stationary distribution of pi* in %s\n(target=[%d,%d], gamma=%.3f, seed=%d)" % (
            env_name, 
            target // n,
            target % n, 
            gamma, 
            seed_id,
        ),
        xlabel="x-coordinate",
        ylabel="y-coordinate",
    )
    plt.tight_layout()
    if fig_fname is not None:
        plt.savefig(fig_fname, dpi=240)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=str, default="final")
    parser.add_argument("--env_name", type=str, required=True)
    args = parser.parse_args()

    if args.plot == "final":
        print_final_convergence_result(args.env_name)
    elif args.plot == "gap":
        # seed_id=5 from running `plot_stationary_distribution_heatmap`
        plot_fgap_vs_advantage_gap(args.env_name, gamma=0.999, seed_id=7, fig_fname="gridworld_hill_gap.png")
    elif args.plot == "dist":
        plot_stationary_distribution_heatmap(args.env_name, gamma=0.999, fig_fname="gridworld_hill_rho.png")
    else:
        print("Unknown plot type %s" % args.plot)

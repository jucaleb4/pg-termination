import os
import sys

import argparse
import numpy as np
import pandas as pd

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
        "gridworld_large",  # not ready yet
        "gridworld_hill_small", 
        "gridworld_hill_large", # not ready yet
        "taxi",
        "random",
        "chain",
    ]
    alg_name_arr = [
        "pmd-kl",
        "pmd-eu",
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
            ct += 1
        print(dashed_msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    args = parser.parse_args()
    print_final_convergence_result(args.env_name)

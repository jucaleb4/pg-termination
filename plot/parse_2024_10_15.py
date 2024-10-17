import os
import numpy as np
import pandas as pd

DATE = "2024_10_15"
EXP_ID = 0
N_RUNS = 64
N_SEEDS = 10

def read_alg_performance(fname):
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
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "seed=%d.csv" % seed_id)
            exp_data = read_alg_performance(fname)
            if exp_data.shape[1] == 3:
                # policyiter only uses greedy
                data[run_id, seed_id,:3] = exp_data[-1]
                data[run_id, seed_id,3] = True
            else:
                # pmd uses either randomized or greedy to termiante
                use_greedy = exp_data[-1,2] >= exp_data[-1,3]
                data[run_id, seed_id,:2] = exp_data[-1,:2]
                data[run_id, seed_id,3] = exp_data[-1,3 if use_greedy else 2]
                data[run_id, seed_id,3] = use_greedy

    return data

if __name__ == "__main__":
    read_all_final_alg_performances()

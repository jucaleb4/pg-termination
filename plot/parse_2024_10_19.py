import os
import sys
import argparse
import numpy as np
import pandas as pd

import matplotlib.pylab as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

DATE = "2024_10_19"
N_SEEDS = 10
N_SAMPLES_PER_STATE_PER_ITER = 50
# first two are for gamma=0.9 (gridworld and taxi), next two are 0.99
opt_arr = [[5.108817, -3.2235],[23.37597, -73.35445]]

# TODO: Show sparse? 
env_name_map = {
    "gridworld_small": 0, 
    "gridworld_small_sparse": 1,
    "taxi": 2,
    "taxi_sparse": 3,
}
# Recall that remove sparse so that 2->1
env_name_map = {
    "gridworld_small": 0, 
    "taxi": 1,
}
gamma_to_exp_id_map = {
    "0.9": 1,
    "0.99": 3,
}

"""
GW Traps (0.9) OPT: 5.108817
Taxi (0.9) OPT: -40.52865
"""

def read_alg_seed_performance(fname):
    """ Read *_seed=%d.csv files """
    if not os.path.exists(fname):
        return None

    df = pd.read_csv(fname, header="infer")
    data = df.to_numpy()
    return data

def _read_all_alg_seed_performances(n_runs, exp_id):
    """ 
    Returns 4D tensor, where: 
        - 1st index is run_id 
        - 2nd index is seed_id
        - 3rd index is iteration
        - 4th index contains data for:
    [
        iter,
        point value,
        point opt_lb,
        point uni_opt_lb,
        agg value,
        agg opt_lb,
        agg uni_opt_lb,
        true value,
        true opt_lb,
        true uni_opt_lb
    ]
    """

    folder = os.path.join("logs", DATE, "exp_%d" % exp_id)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    data = None
    for run_id in range(n_runs):
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "seed=%d.csv" % seed_id)
            exp_data = read_alg_seed_performance(fname)
            if data is None:
                data = np.zeros((n_runs, N_SEEDS, exp_data.shape[0], exp_data.shape[1]), dtype=float)
            data[run_id, seed_id,:,:] = exp_data

    return data

def _read_validation_seed_performances(n_runs, exp_id):
    """ 
    Returns 3D tensor, where: 
        - 1st index is run_id 
        - 2nd index is seed_id
        - 3rd index contains data for:
    [
        agg value
        agg opt_lb
        agg uni_opt_lb
        true value
        true opt_lb
        true uni_opt_lb
        agg V_err
        avg total_V_err
    ]
    """

    folder = os.path.join("logs", DATE, "exp_%d" % exp_id)
    if not os.path.exists(folder):
        print("Folder %s cannot be found. Exitting")
        return

    data = np.zeros((n_runs, N_SEEDS, 8), dtype=float)
    for run_id in range(n_runs):
        for seed_id in range(N_SEEDS):
            fname = os.path.join(folder, "run_%d" % run_id, "validation_seed=%d.csv" % seed_id)
            exp_data = read_alg_seed_performance(fname)
            data[run_id, seed_id,:] = exp_data

    return data

def read_all_alg_seed_performances(exp_id):
    if exp_id == 1:
        return _read_all_alg_seed_performances(4, exp_id)[::2]
    return _read_all_alg_seed_performances(2, exp_id)

def read_validation_seed_performances(exp_id):
    if exp_id == 1:
        return _read_validation_seed_performances(4, exp_id)[::2]
    return _read_validation_seed_performances(2, exp_id)

def plot_ub_lb_convergence(env_name, gamma, fig_fname=None):
    """
    Plots convergence of ub and lb over algorithm's convergence
    """

    # gridworld is a 10x10 grid, taxi is a 5x5x5x4=500 state space
    # seee: https://gymnasium.farama.org/environments/toy_text/taxi/
    n_states_arr = [100, 100, 500, 500]

    if env_name not in env_name_map: 
        print("Env %s not registered" % env_name)
        return
    env_id = env_name_map[env_name]
    exp_id = gamma_to_exp_id_map[str(gamma)]

    print("gamma: %.3f" % gamma)
    print("env, exp: ", env_id, exp_id)

    data = read_all_alg_seed_performances(exp_id)

    plt.style.use('ggplot')
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(10,5)

    n_samples_per_iter = N_SAMPLES_PER_STATE_PER_ITER * n_states_arr[env_id]
    xs = n_samples_per_iter * np.arange(data.shape[2])
    xs = np.arange(data.shape[2])
    z = 1.96
    
    # Plot 1:
    ub_color_arr=["blue","red","green","yellow"]
    lb_color_arr=["blue","red","green","yellow"]
    lss_arr = ["dashed", "dotted", "dashdot", (1,(3,5,1,5,1,5))]
    label_arr = ["point", "adapt aggregate", "true", "univ. aggregate (only lb)"]

    # plot OPT
    for i in range(2):
        axes[i].hlines(opt_arr[exp_id//2][env_id], xs[0], xs[-1], color="black", label="OPT")

    # point, aggregated, true estimates with conservative lb
    for i in range(3):
        mu = np.mean(data[env_id,:,:,1+3*i], axis=0)
        sig = np.std(data[env_id,:,:,1+3*i], axis=0)
        axes[0].plot(xs, mu, linestyle=lss_arr[i], color=ub_color_arr[i], label=label_arr[i])
        axes[0].fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=ub_color_arr[i])
        mu = np.mean(data[env_id,:,:,2+3*i], axis=0)
        sig = np.std(data[env_id,:,:,2+3*i], axis=0)
        axes[0].plot(xs, mu, linestyle=lss_arr[i], color=lb_color_arr[i])
        axes[0].fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=lb_color_arr[i])

    mu = np.mean(data[env_id,:,:,3+3*1], axis=0)
    sig = np.std(data[env_id,:,:,3+3*1], axis=0)
    axes[0].plot(xs, mu, linestyle=lss_arr[3], color=ub_color_arr[3], label=label_arr[3])
    axes[0].fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=ub_color_arr[3])

    # worst-case, aggregated, apriori
    """
    Recall the last index in data has the following values
    [ iter, point value, point opt_lb, point uni_opt_lb, agg value,
      agg opt_lb, agg uni_opt_lb, true value, true opt_lb, true uni_opt_lb
    ]
    """
    # we only want upper and lower bounds for the three labels
    data2 = np.zeros((data.shape[1:3] + (6,)), dtype=float)
    label_arr = ["worst-case (lb only)", "adapt aggregate", "apriori (lb only)"]
    barQ = 10
    logA = np.log(4)/np.log(2)
    agg_V_ub = data[env_id,:,:,1+3*1]
    agg_V_lb = data[env_id,:,:,2+3*1]
    
    # give all experiments the same upper bound
    for i in range(len(label_arr)):
        data2[:,:,1+2*i] = agg_V_ub
    data2[:,:,0] = agg_V_ub - np.outer(
        np.ones(N_SEEDS), 
        2*np.sqrt(np.divide(logA*barQ**2, (1.-gamma)**2*np.arange(1, data.shape[2]+1)).astype('float'))
    )

    # aggregate
    data2[:,:,2] = agg_V_lb

    # worst-case
    if env_id == 0:
        # gridworld
        data2[:,:,4] = (1.-gamma**1)/(1.-gamma**2)
    else:
        # taxi
        data2[:,:,4] = ((1.-gamma**2)/(1-gamma)-20*gamma**3)/(1.-gamma**3)

    for i in range(3):
        if i == 1:
            mu = np.mean(data2[:,:,1+2*i], axis=0)
            sig = np.std(data2[:,:,1+2*i], axis=0)
            axes[1].plot(xs, mu, linestyle=lss_arr[i], color=ub_color_arr[i])
            axes[1].fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=ub_color_arr[i])
        mu = np.mean(data2[:,:,2*i], axis=0)
        sig = np.std(data2[:,:,2*i], axis=0)
        axes[1].plot(xs, mu, linestyle=lss_arr[i], color=lb_color_arr[i], label=label_arr[i])
        axes[1].fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=lb_color_arr[i])

    i_star = 0
    title_arr = ["various adv. gap", "hueristic adv. gap"]
    if exp_id == 1:
        y_lim = (-20, 10) if env_id == 0 else (-75, 20)
    else:
        y_lim = (-150, 80) if env_id == 0 else (-10_000, 1_000)
    for i in range(2):
        axes[i].legend(loc=3)
        axes[i].set(
            ylim=y_lim,
            # xlabel="Total samples",
            xlabel="Iteration",
            ylabel=r"Estimate of $f_\rho(\pi_k)$ and $f_\rho(\pi^*)$ with $\rho=\mathrm{Uni}(\mathcal{S})$" if i==0 else "",
            title="Estimates of f^* with %s in\n%s with gamma=%.3f" % (title_arr[i], env_name, gamma),
        )
    plt.tight_layout()

    if fig_fname is None:
        plt.show()
    else:
        plt.savefig(fig_fname, dpi=240)

def plot_ub_lb_convergence2(env_name, gamma, fig_fname=None):
    """
    This is similar to `plot_ub_lb_convergence`, but only shows one plot with 
    OPT, worst-case, adapt aggreagte, univ. aggregate, and apriori (per
    Goerge's suggestion).
    """

    # gridworld is a 10x10 grid, taxi is a 5x5x5x4=500 state space
    # seee: https://gymnasium.farama.org/environments/toy_text/taxi/
    n_states_arr = [100, 100, 500, 500]

    if env_name not in env_name_map: 
        print("Env %s not registered" % env_name)
        return
    env_id = env_name_map[env_name]
    exp_id = gamma_to_exp_id_map[str(gamma)]

    print("gamma: %.3f" % gamma)
    print("env, exp: ", env_id, exp_id)

    data = read_all_alg_seed_performances(exp_id)

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(6,5)

    n_samples_per_iter = N_SAMPLES_PER_STATE_PER_ITER * n_states_arr[env_id]
    xs = n_samples_per_iter * np.arange(data.shape[2])
    xs = np.arange(data.shape[2])
    z = 1.96
    
    # Plot 1:
    ub_color_arr=["blue","red","green","yellow"]
    lb_color_arr=["blue","red","green","yellow"]
    lss_arr = ["dashed", "dotted", "dashdot", (1,(3,5,1,5,1,5))]

    # plot OPT
    ax.hlines(opt_arr[exp_id//2][env_id], xs[0], xs[-1], color="black", label="OPT")

    # worst-case, aggregated, apriori
    """
    Recall the last index in data has the following values
    [ iter, point value, point opt_lb, point uni_opt_lb, agg value,
      agg opt_lb, agg uni_opt_lb, true value, true opt_lb, true uni_opt_lb
    ]
    """
    # we only want upper and lower bounds for the three labels
    label_arr = ["adapt aggregate", "univ. aggregate (only lb)", "worst-case (only lb)", "apriori (only lb)"]
    data2 = np.zeros((data.shape[1:3] + (8,)), dtype=float)
    barQ = 10
    logA = np.log(4)/np.log(2)
    agg_V_ub = data[env_id,:,:,1+3*1]
    agg_V_lb = data[env_id,:,:,2+3*1]
    agg_V_univ_lb = data[env_id,:,:,3+3*1]
    
    # give all experiments the same upper bound
    for i in range(len(label_arr)):
        data2[:,:,1+2*i] = agg_V_ub

    # worst-case
    data2[:,:,4] = agg_V_ub - np.outer(
        np.ones(N_SEEDS), 
        2*np.sqrt(np.divide(logA*barQ**2, (1.-gamma)**2*np.arange(1, data.shape[2]+1)).astype('float'))
    )

    # aggregate
    data2[:,:,0] = agg_V_lb
    data2[:,:,2] = agg_V_univ_lb

    # worst-case
    if env_id == 0:
        # gridworld
        data2[:,:,6] = (1.-gamma**1)/(1.-gamma**2)
    else:
        # taxi
        data2[:,:,6] = ((1.-gamma**2)/(1-gamma)-20*gamma**3)/(1.-gamma**3)

    for i in range(4):
        if i == 0:
            mu = np.mean(data2[:,:,1+2*i], axis=0)
            sig = np.std(data2[:,:,1+2*i], axis=0)
            ax.plot(xs, mu, linestyle=lss_arr[i], color=ub_color_arr[i])
            ax.fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=ub_color_arr[i])
        mu = np.mean(data2[:,:,2*i], axis=0)
        sig = np.std(data2[:,:,2*i], axis=0)
        ax.plot(xs, mu, linestyle=lss_arr[i], color=lb_color_arr[i], label=label_arr[i])
        ax.fill_between(xs, mu-z*sig, mu+z*sig, alpha=0.2, color=lb_color_arr[i])

    if exp_id == 1:
        y_lim = (-25, 15) if env_id == 0 else (-75, 20)
    else:
        y_lim = (-200, 100) if env_id == 0 else (-10_000, 1_000)
    ax.legend(loc=3)
    ax.set(
        xlim=(0,len(xs)),
        ylim=y_lim,
        # xlabel="Total samples",
        xlabel="Iteration (k)",
        ylabel=r"Estimate of $f_\rho(\pi_k)$ and $f_\rho(\pi^*)$",
        title=r"Estimates of $f_\rho(\pi_k)$ ($\rho=\mathrm{Uni}(\mathcal{S})$) in" + "\n%s with gamma=%.3f" % (env_name, gamma),
    )

    plt.tight_layout()

    if fig_fname is None:
        plt.show()
    else:
        plt.savefig(fig_fname, dpi=240)

def print_validation_results(env_name, gamma):
    """
    Prints out validation analysis results
    """
    if env_name not in env_name_map: 
        print("Env %s not registered" % env_name)
        return
    env_id = env_name_map[env_name]
    exp_id = gamma_to_exp_id_map[str(gamma)]

    alg_data = read_all_alg_seed_performances(exp_id)
    data = read_validation_seed_performances(exp_id)

    exp_metadata = ["Method", "Expected Value", "Standard Deviation"]
    row_format ="{:>15}|{:>20}|{:>20}"
    dashed_msg = "-" * (15+20*(len(exp_metadata)-1) + len(exp_metadata) - 1)

    print("")
    print("Env %s with gamma: %.4f" % (env_name, gamma))
    print("OPT: %.6f" % opt_arr[exp_id//2][env_id])
    print(row_format.format(*exp_metadata))
    print(dashed_msg)

    print(row_format.format("online_V_agg", "%.6f" % np.mean(alg_data[env_id,:,-1,1]), "%.6f" % np.std(alg_data[env_id,:,-1,1])))
    print(row_format.format("offline_V_agg", "%.6f" % np.mean(data[env_id,:,0]), "%.6f" % np.std(data[env_id,:,0])))
    print(row_format.format("offline_V_true", "%.6f" % np.mean(data[env_id,:,3]), "%.6f" % np.std(data[env_id,:,3])))
    print(row_format.format("online_V*_agg", "%.6f" % np.mean(alg_data[env_id,:,-1,5]), "%.6f" % np.std(alg_data[env_id,:,-1,5])))
    print(row_format.format("offline_V*_agg", "%.6f" % np.mean(data[env_id,:,1]), "%.6f" % np.std(data[env_id,:,1])))
    print(row_format.format("offline_V*_true", "%.6f" % np.mean(data[env_id,:,4]), "%.6f" % np.std(data[env_id,:,4])))

    print("")
    exp_metadata = ["Metric", "Agg V_err", "Avg Total V_err"]
    print(row_format.format(*exp_metadata))
    print(dashed_msg)
    print(row_format.format("E|V-tilde(V)|", "%.6f" % np.mean(data[env_id,:,6]), "%.6f" % np.mean(data[env_id,:,7])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--env_name", type=str, required=True, choices=["gridworld_small", "taxi"])
    parser.add_argument("--gamma", type=float, required=True, choices=[0.9, 0.99])
    args = parser.parse_args()

    env_name = args.env_name
    gamma = args.gamma
    if args.plot:
        # plot_ub_lb_convergence(env_name, gamma, "%s_convergence_%.3f.png" % (env_name, gamma))
        plot_ub_lb_convergence2(env_name, gamma, "%s_convergence_%.3f.png" % (env_name, gamma))
    else:
        print_validation_results(env_name, gamma)

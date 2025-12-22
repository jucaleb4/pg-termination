import os
import sys
import itertools
import argparse
import yaml
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from pg_termination import pmd

MAX_RUNS = 4
DATE = "2025_12_11"
EXP_ID  = 0

def parse_sub_runs(sub_runs):
    start_run_id, end_run_id = 0, MAX_RUNS
    if (sub_runs is not None):
        try:
            start_run_id, end_run_id = sub_runs.split(",")
            start_run_id = int(start_run_id)
            end_run_id = int(end_run_id)
            assert 0 <= start_run_id <= end_run_id <= MAX_RUNS, "sub_runs id must be in [0,%s]" % (MAX_RUNS-1)
            
        except:
            raise Exception("Invalid sub_runs id. Must be two integers split between [0,%s] split by a single comma with no space" % (MAX_RUNS-1))

    return start_run_id, end_run_id

def get_parameter_settings(seed_0, n_seeds, n_iters, print_info=False):
    od = dict([
        ("seed_0", seed_0), 
        ("n_seeds", n_seeds), 
        ("n_iters", n_iters),
        ("alg", "spmd"),
        ("eps", math.exp(-10)),
        ("delta", 1e-2),
        ("stepsize_rule", int(pmd.StepSize.SUBLINEAR)), 
        ("update_rule", int(pmd.Update.KL_UPDATE)),
        ("estimate_Q", "online"),
        ("env_name", "battery"),
        ("gamma", 0.9),
        ("N", 1), # number of replications in estimation
        ("T", 1000), # approximately log(5)/log(1/0.99)
        ("validation_k", 0), # skip post-validation if use conservative estimated values
        ("pi_threshold_mult", 1.0),
        ("eta", 0.01),
        ("linear_learning_rate", "constant"), # constant, optimal
        ("linear_eta0", 1e-2), # 1e-2, 1e-3
        ("linear_max_iter", 10), # 1000, 100, 10
        ("linear_alpha", 1e-4), # 1e-3, 1e-4
        ("skip_true_model", False),
        ("ctd_feature_size", 100),
        ("ctd_N_alt", 1000),
        ("ctd_iota_alt", 1e-1),
        ("tune_exploration", False),
        ("successive_half_trials", 16),
        ("min_T", 100),
        ("max_T", 2e6),
    ])

    od_info = [
        ("seed_0", "start seed"), 
        ("n_seeds", "num seeds"), 
        ("n_iters", "num SPMD iters"),
        ("alg", "which algorithm to run ('pmd', 'spmd', 'policyiter')"),
        ("eps", "acc tolerance. Used for dynamic mixing time and CTD"),
        ("delta", "failure rate (only used by CTD for robust est)"),
        ("stepsize_rule", "stepsize rule (const, decr). See pmd.StepSizeint enum"),
        ("update_rule", "pmd update (euc, kl, tsallis)"),
        ("estimate_Q", "how to estimate Q (gen, Monte Carlo, CTD)"),
        ("env_name", "environment"),
        ("gamma", "discount factor"),
        ("N", "number of replications in estimating Q (only for gen)"),
        ("T", "Monte Carlo estimation length (only for gen, non-dynamic mc)"), 
        ("validation_k", "offline validation step replication amt"),
        ("pi_threshold_mult", "constant factor in cut-off for sub-opt actions"),
        ("eta", "base step size"),
        ("linear_learning_rate", "sklearn linear stepsize rule ('constant', 'optimal')"), 
        ("linear_eta0", "sklearn linear starting stepsize"),
        ("linear_max_iter", "sklearn linear iterations (may early terminate)"), 
        ("linear_alpha", "sklearn regularization strength"), 
        ("skip_true_model", "skips validation on true model - only works for non-gym envs"),
        ("ctd_feature_size", "feature size for CTD (only)"),
        ("ctd_N_alt", "User-chosen fixed CTD iterations (set to 0 to use theory)"),
        ("ctd_iota_alt", "User-chosen fixed CTD stepsize (set to 0 to use theory)"),
        ("tune_exploration", "Tune exploration time in Monte Carlo"),
        ("successive_half_trials", "Number of trials in successive halving tuning"),
        ("min_T", "Minimum exploration time in tuning"),
        ("max_T", "Maximum exploration time in tuning"),
    ]

    if print_info:
        exp_metadata = ["setting", "description"]
        row_format ="{:<20}|{:<60}"
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (80+len(exp_metadata)-1))
        for name, description in od_info:
            print(row_format.format(name, description))
        print("-" * (80+len(exp_metadata)-1))

    return od

def setup_setting_files(seed_0, n_seeds, n_iters, print_info):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, print_info)

    n_iters_estimator_T_arr = [
        (1000, "online_mc_fixed", 500),
        (20, "online_mc_estimate", 0),
        (225, "online_mc_dynamic", 0),
        (250, "ctd", 0),
    ]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "n_iters", "estimator", "T"]
    row_format ="{:>10}|{:>10}|{:>20}|{:>10}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (50+len(exp_metadata)-1))

    ct = 0
    for ((n_iters, estimator, T),) in itertools.product(n_iters_estimator_T_arr):
        od["n_iters"] = n_iters
        od["estimate_Q"] = estimator
        od["T"] = T

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        print(row_format.format(ct, od["n_iters"], od["estimate_Q"], od["T"]))

        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w') as f:
            # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
            yaml.dump(od, f, default_flow_style=False, sort_keys=False)
        ct += 1

    assert ct == MAX_RUNS, "Number of created exps (%i) does not match MAX_RUNS (%i)" % (ct, MAX_RUNS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments (if not passed, no setup)")
    parser.add_argument("--run", action="store_true", help="Runs experiments (applied after setup)")
    parser.add_argument("--print_info", action="store_true", help="Prints description of all settings")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="work", 
        choices=["work"],
        help="Set up number of trials and max_step for various testing reasons"
    )
    parser.add_argument(
        "--sub_runs", 
        type=str, 
        help="Which experiments to run. Must be given as two integers separate by a comma with no space"
    )
    parser.add_argument("--parallel", action="store_true", help="Run seeds in parallel")

    args = parser.parse_args()
    seed_0 = 0
    n_seeds = 1
    n_iters = 500

    if args.setup:
        setup_setting_files(seed_0, n_seeds, n_iters, args.print_info)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id):
            settings_file = os.path.join(folder_name, "run_%i.yaml" % i)
            os.system('echo "Running exp id %d"' % i)
            os.system("python run.py --settings %s%s" % (
                settings_file, 
                " --parallel" if args.parallel else "",
            ))
    else:
        print("Neither setup nor run passed. Shutting down...")

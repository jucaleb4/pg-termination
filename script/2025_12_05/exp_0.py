import os
import sys
import itertools
import argparse
import yaml
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."+".", "."+"."))
sys.path.insert(0, parent_dir)

from pg_termination import pmd
from script.helper import get_parameter_settings, parse_sub_runs

DATE = "2025_12_05"
EXP_ID  = 0
ABOUT = "Auto-Explore with Monte-Carlo on Tiny Gridworld" 

def setup_setting_files(seed_0, n_seeds, n_iters, print_info, skip_save=False):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, print_info, ABOUT)
    od["env_name"] = "gridworld_footnote"
    od["linear_learning_rate"] = "constant" # constant, optimal
    od["linear_eta0"] = 1e-2 # 1e-2, 1e-3
    od["linear_max_iter"] = 10 # 1000, 100, 10
    od["linear_alpha"] = 1e-4 # 1e-3, 1e-4
    od["skip_true_model"] = False
    od["ctd_feature_size"] = 100
    od["min_T"] = 100
    od["max_T"] = 2e6

    n_iters_estimator_T_arr = [
        (1000, "online_mc_fixed", 500),
        (50, "online_mc_estimate", 0),
        (200, "online_mc_dynamic", 0),
        (200, "ctd", 0),
    ]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

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
        od["T_mc"] = T

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        print(row_format.format(ct, od["n_iters"], od["estimate_Q"], od["T_mc"]))

        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w') as f:
            # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
            yaml.dump(od, f, default_flow_style=False, sort_keys=False)
        ct += 1

    return ct

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
        max_runs = setup_setting_files(seed_0, n_seeds, n_iters, args.print_info, True)
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs, max_runs)
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

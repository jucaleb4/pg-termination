import os
import sys
import itertools
import argparse
import yaml
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."+".", "."+"."))
sys.path.insert(0, parent_dir)

from pg_termination import pmd
from script.helper import get_parameter_settings, parse_sub_runs

DATE =  os.path.dirname(__file__).split("/")[-1] # "2025_12_24"
EXP_ID = int(re.search(r'\d+', os.path.splitext(os.path.basename(__file__))[0]).group()) # 0
ABOUT = "SPMD full experiment on Garnet with Tsallis update"

def setup_setting_files(seed_0, n_seeds, n_iters, print_info, skip_save=False):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, False, ABOUT)

    od["alg"] = "spmd"
    od["n_iters"] = n_iters
    od["skip_true_model"] = True
    od["validation_mode"] = "random_reset"
    od["validation_k"] = 30
    od["update_rule"] = int(pmd.Update.TSALLIS_UPDATE)

    estimator_arr = ["online_mc_fixed", "online_mc_estimate", "online_mc_dynamic"]
    env_gamma_T_eta_maxobs_arr = [
        ("garnet_50", 0.9, 2_000, 0.5, 5e6),
        ("garnet_50", 0.99, 400, 0.02, 5e6), 
        ("garnet_50", 0.995, 400, 0.02, 5e6), 
        ("garnet_200", 0.9, 400, 0.02, 1e7), 
        ("garnet_200", 0.99, 400, 0.005, 1e7), 
        ("garnet_200", 0.995, 400, 0.005, 1e7), 
        ("garnet_1000", 0.9, 400, 0.005, 2e7), 
        ("garnet_1000", 0.99, 400, 0.005, 2e7), 
        ("garnet_1000", 0.995, 400, 0.005, 2e7), 
    ]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "Env name", "gamma", "n_iters", "estimator", "eta"]
    row_format ="{:>10}|{:>15}|{:>10}|{:>10}|{:>20}|{:>10}"
    if not skip_save:
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (75+len(exp_metadata)-1))

    ct = 0
    for ((env_name, gamma, T_mc, eta, max_obs), estimator) in itertools.product(env_gamma_T_eta_maxobs_arr, estimator_arr):
        od["env_name"] = env_name
        od["gamma"] = gamma
        od["T_mc"] = T_mc
        od["eta"] = eta
        od["estimate_Q"] = estimator
        od["max_obs"] = max_obs
        od["min_obs"] = int(max_obs*0.9)

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        if not skip_save:
            print(row_format.format(ct, od["env_name"], od["gamma"], od["n_iters"], od["estimate_Q"], od["eta"]))

            if not(os.path.exists(od["log_folder"])):
                os.makedirs(od["log_folder"])
            with open(setting_fname, 'w') as f:
                # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
                yaml.dump(od, f, default_flow_style=False, sort_keys=False)
        ct += 1

    return ct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--run", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--print_info", action="store_true", help="Prints description of all settings")
    parser.add_argument(
        "--mode", 
        type=str, 
        default="work", 
        choices=["full", "work"],
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
    n_iters = 1_000

    if args.setup:
        if args.mode == "full":
            n_seeds = 10
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

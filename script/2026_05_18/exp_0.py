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
ABOUT = "Tune SPMD+CTD on GridWorld. We set tune limit based on # samples and not iterations (sensitive to hyperparameters)"

def setup_setting_files(seed_0, n_seeds, n_iters, print_info, skip_save=False):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, False, ABOUT)

    od["estimate_Q"] = "ctd"
    od["skip_true_model"] = True
    total_samples_base = 20_000
    # use validation to tune...
    od["validation_mode"] = "random_reset"
    od["validation_k"] = 10
    od["max_runtime_in_sec"] = 600

    env_name_ctd_size_arr = [
        # "gridworld_footnote", 
        ("gridworld_small", 1.),
        ("gridworld_large", 0.5),
    ]
    s_origin_arr = [None, 'rand']
    gamma_arr = [0.9, 0.99, 0.995]
    eta_arr = [5e-3, 2e-2, 5e-1]
    ukappa_arr = [1e0,1e0/(5**0.25),1e0/(10**0.25)] # [1e0, 2e-1]
    for i in range(len(ukappa_arr)):
        ukappa_arr[i] = int(1e3*ukappa_arr[i])/1e3
    ctd_iota_arr = [5e-3, 5e-1]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "Env name", "gamma", "feat frac", "s_orig", "eta", "iota", "ukappa"]
    row_format ="{:>10}|{:>20}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}"
    if not skip_save:
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (90+len(exp_metadata)-1))

    ct = 0
    for ((env_name, ratio), gamma, s_origin, eta, iota, ukappa) in itertools.product(
            env_name_ctd_size_arr, gamma_arr, s_origin_arr, eta_arr, ctd_iota_arr, ukappa_arr
    ):
        od["env_name"] = env_name
        od["ctd_feature_size_ratio"] = ratio
        od["gamma"] = gamma
        od["max_obs"] = total_samples_base/(1.-gamma)
        od["s_origin"] = s_origin
        od["eta"] = eta
        od["iota"] = iota
        od["ukappa"] = ukappa

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        if not skip_save:
            print(row_format.format(ct, od["env_name"], od["gamma"], od["ctd_feature_size_ratio"], 
            od["s_origin"] if od["s_origin"] is not None else "none", 
            od["eta"], od["iota"], od["ukappa"]))

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
    n_iters = 40

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

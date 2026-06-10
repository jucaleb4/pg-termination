import os
import sys
import itertools
import argparse
import yaml
import re
import math

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."+".", "."+"."))
sys.path.insert(0, parent_dir)

from pg_termination import pmd
from script.helper import get_parameter_settings, parse_sub_runs

DATE =  os.path.dirname(__file__).split("/")[-1] # "2025_12_24"
EXP_ID = int(re.search(r'\d+', os.path.splitext(os.path.basename(__file__))[0]).group()) # 0
ABOUT = "Refined tuning-v2 of SPMD+CTD on GridWorld (understand stepsize with func approx error)"

def setup_setting_files(seed_0, n_seeds, n_iters, print_info, skip_save=False):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, False, ABOUT)

    od["estimate_Q"] = "ctd"
    od["skip_true_model"] = True
    od["validation_mode"] = "random_reset"
    od["validation_k"] = 30
    od["max_runtime_in_sec"] = 3600
    od["max_obs"] = math.inf
    od["s_origin"] = None
    od["ukappa"] = 1.0

    TS = int(pmd.Update.TSALLIS_UPDATE)
    KL = int(pmd.Update.KL_UPDATE)

    # fixed parameters
    od["update_rule"] = KL
    od["n_batches"] = 1
    od["ctd_feat_type"] = "Gaussian"
    od["ctd_feat_size"] = 1000
    od["ctd_burn_in"] = False
    od["ctd_N_mult"] = 1.

    # tuning parameters
    env_name_arr = ["gridworld_tiny", "gridworld_small", "gridworld_large"]
    gamma_arr = [0.9, 0.99]

    eta_arr = [1e4, 1e2, 1e0] 
    iota_mult_arr = [1e4, 1e2, 1e0]
    uLam_mult_arr = [1, -1./2] 
    for i in range(len(uLam_mult_arr)):
        uLam_mult_arr[i] = int(1e3*uLam_mult_arr[i])/1e3

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "Env name", "gamma", "eta", "iota_mult", "uLam"]
    row_format ="{:>10}|{:>15}|{:>10}|{:>10}|{:>10}|{:>10}"
    if not skip_save:
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (65+len(exp_metadata)-1))

    ct = 0
    for (env_name, gamma, eta, iota_mult, uLam_mult) in itertools.product(
            env_name_arr, gamma_arr, eta_arr, iota_mult_arr, uLam_mult_arr, 
    ):
        od["env_name"] = env_name
        od["gamma"] = gamma
        od["eta"] = eta
        od["ctd_iota_mult"] = iota_mult
        od["ctd_uLam_mult"] = uLam_mult if uLam_mult > 0 else (1e-3*int(1e3*(1-gamma)**(uLam_mult)))

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        if not skip_save:
            print(row_format.format(ct, od["env_name"], od["gamma"], od["eta"], 
                od["ctd_iota_mult"], od["ctd_uLam_mult"], 
            ))

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

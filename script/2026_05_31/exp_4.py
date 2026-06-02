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
ABOUT = "Full-run tune SPMD+CTD on GridWorld. More parameters (as of June 2nd, 2026)"

def setup_setting_files(seed_0, n_seeds, n_iters, print_info, skip_save=False):
    od = get_parameter_settings(seed_0, n_seeds, n_iters, False, ABOUT)

    od["estimate_Q"] = "ctd"
    od["skip_true_model"] = True
    od["validation_mode"] = "random_reset"
    od["validation_k"] = 30
    od["max_runtime_in_sec"] = 3600
    # od["ctd_reg_ratio"] = 1.0
    # od["ctd_feature_size_ratio"] = 1.0
    od["max_obs"] = math.inf
    od["s_origin"] = None
    od["ctd_reg_val"] = 0
    od["ctd_estimate_Phi_sigs"] = False

    env_name_max_obs_arr = [
        ("gridworld_small", int(1e6)),
        ("gridworld_large", int(2e6)),
    ]
    ctd_feat_type_arr = ['Gaussian', 'rff']
    update_type_ctd_feat_size_arr = [(int(pmd.Update.TSALLIS_UPDATE), -1), (int(pmd.Update.KL_UPDATE), 100)]
    gamma_arr = [0.9, 0.99, 0.995]
    eta_arr = [5e-1, 1e-3]
    ukappa_arr = [1e0,1e0/(10**0.25)] # [1e0, 2e-1]
    for i in range(len(ukappa_arr)):
        ukappa_arr[i] = int(1e3*ukappa_arr[i])/1e3
    iota_mult_arr = [5e-1, 1e-3]
    burn_in_arr = [False, True]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not skip_save:
        if not(os.path.exists(log_folder_base)):
            os.makedirs(log_folder_base)
        if not(os.path.exists(setting_folder_base)):
            os.makedirs(setting_folder_base)
        print("Saving setting files to %s" % setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "Env name", "gamma", "feat type", "feat size", "update", "eta", "iota_mult", "ukappa", "burn_in"]
    row_format ="{:>10}|{:>15}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}|{:>10}"
    if not skip_save:
        print("")
        print(row_format.format(*exp_metadata))
        print("-" * (95+len(exp_metadata)-1))

    ct = 0
    for ((env_name, max_obs), gamma, (update, feat_size), eta, feat_type, iota_mult, ukappa, burn_in) in itertools.product(
            env_name_max_obs_arr, gamma_arr, update_type_ctd_feat_size_arr, eta_arr, 
            ctd_feat_type_arr, iota_mult_arr, ukappa_arr, burn_in_arr,
    ):
        od["env_name"] = env_name
        od["max_obs"] = max_obs
        od["gamma"] = gamma
        od["update_rule"] = update
        od["eta"] = eta
        od["ctd_feat_type"] = feat_type
        od["ctd_feat_size"] = feat_size
        od["ctd_iota_mult"] = iota_mult
        od["ukappa"] = ukappa
        od["ctd_burn_in"] = burn_in
        od["ctd_N_mult"] = 1.-gamma

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        if not skip_save:
            print(row_format.format(ct, od["env_name"], od["gamma"], 
                od["ctd_feat_type"], od["ctd_feat_size"], 
                pmd.Update(od["update_rule"]).name[:7],
                od["eta"], od["ctd_iota_mult"], od["ukappa"], od["ctd_burn_in"])
            )

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

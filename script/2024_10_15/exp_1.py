import os
import sys
import itertools
import argparse
import yaml

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, parent_dir)

from pg_termination import pmd

MAX_RUNS = 12
DATE = "2024_10_15"
EXP_ID  = 1

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

def setup_setting_files(seed_0, n_seeds, n_iters):
    od = dict([
        ("seed_0", seed_0), 
        ("n_seeds", n_seeds), 
    ])

    env_name_arr = [
        "gridworld_large", 
        "gridworld_hill_large", 
    ]

    gamma_arr = [0.9, 0.99]

    alg_setting_arr = [
        ("pmd", pmd.StepSize.KL_LINEAR_GEOMETRIC, pmd.Update.KL_UPDATE),
        ("pmd", pmd.StepSize.EUCLIDEAN_LINEAR_ADAPTIVE, pmd.Update.EUCLIDEAN_UPDATE),
        ("policyiter", 0, 0),
    ]

    log_folder_base = os.path.join("logs", DATE, "exp_%s" % EXP_ID)
    setting_folder_base = os.path.join("settings", DATE, "exp_%s" % EXP_ID)

    if not(os.path.exists(log_folder_base)):
        os.makedirs(log_folder_base)
    if not(os.path.exists(setting_folder_base)):
        os.makedirs(setting_folder_base)

    # https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    exp_metadata = ["Exp id", "Env name", "gamma", "Algm", "Stepsize", "Update"]
    row_format ="{:>10}|{:>25}|" + "{:>15}|" * (len(exp_metadata)-3) + "{:>15}"
    print("")
    print(row_format.format(*exp_metadata))
    print("-" * (10+25+15*(len(exp_metadata)-3) + 15+len(exp_metadata)-1))

    ct = 0
    for (env_name, gamma, alg_setting) in itertools.product(env_name_arr, gamma_arr, alg_setting_arr):
        od["env_name"] = env_name
        od["gamma"] = gamma
        od["alg"] = alg_setting[0]
        od["stepsize_rule"] = int(alg_setting[1])
        od["update_rule"] = int(alg_setting[2])
        od["n_iters"] = n_iters 

        setting_fname = os.path.join(setting_folder_base,  "run_%s.yaml" % ct)
        od["log_folder"] = os.path.join(log_folder_base, "run_%s" % ct)

        print(row_format.format(ct, od["env_name"], od["gamma"], od["alg"], od["stepsize_rule"], od["update_rule"]))

        if not(os.path.exists(od["log_folder"])):
            os.makedirs(od["log_folder"])
        with open(setting_fname, 'w') as f:
            # https://stackoverflow.com/questions/42518067/how-to-use-ordereddict-as-an-input-in-yaml-dump-or-yaml-safe-dump
            yaml.dump(od, f, default_flow_style=False, sort_keys=False)
        ct += 1

    assert ct == MAX_RUNS, "Number of created exps (%i) does not match MAX_RUNS (%i)" % (ct, MAX_RUNS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--setup", action="store_true", help="Setup environments. Otherwise we run the experiments")
    parser.add_argument("--run", action="store_true", help="Setup environments. Otherwise we run the experiments")
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
    n_iters = 10_000

    if args.setup:
        if args.mode == "full":
            n_seeds = 10
        setup_setting_files(seed_0, n_seeds, n_iters)
    elif args.run:
        start_run_id, end_run_id = parse_sub_runs(args.sub_runs)
        folder_name = os.path.join("settings", DATE, "exp_%i" % EXP_ID)

        for i in range(start_run_id, end_run_id):
            settings_file = os.path.join(folder_name, "run_%i.yaml" % i)
            os.system('echo "Running exp id %d"' % i)
            # print("python run.py --settings %s%s" % (
            #     settings_file, 
            #     " --parallel" if args.parallel else "",
            # ))

            os.system("python run.py --settings %s%s" % (
                settings_file, 
                " --parallel" if args.parallel else "",
            ))
    else:
        print("Neither setup nor run passed. Shutting down...")

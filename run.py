import argparse
import yaml

from pg_termination import pmd, policyiter, spmd, reinforce, trpo, qlearn, sarsa
from script.helper import get_parameter_settings

def check_key_settings_diff(settings):
    base_settings = get_parameter_settings(settings['seed_0'], settings['n_seeds'], settings['n_iters'], False, "")

    diff_keys = list(set(settings) - set(base_settings))

    if len(diff_keys) > 0:
        print(">>> Recieved %d extra keys in settings: %s" % (len(diff_keys), diff_keys))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    parser.add_argument("--parallel", action="store_true", help="Run seeds in parallel")
    args = parser.parse_args()

    with open(args.settings) as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    check_key_settings_diff(settings)
    settings["parallel"] = args.parallel

    if settings['alg'] == 'pmd':
        pmd.train(settings)
    elif settings['alg'] == 'spmd':
        spmd.train(settings)
    elif settings['alg'] == 'policyiter':
        policyiter.train(settings)
    elif settings['alg'] == 'reinforce':
        reinforce.train(settings)
    elif settings['alg'] == 'trpo':
        trpo.train(settings)
    elif settings['alg'] == 'qlearn':
        qlearn.train(settings)
    elif settings['alg'] == 'sarsa':
        sarsa.train(settings)
    else:
        print("Unknown alg %s" % settings['alg'])

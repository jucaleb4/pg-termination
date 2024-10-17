import argparse
import yaml

from pg_termination import pmd, policyiter, spmd

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

    settings["parallel"] = args.parallel

    if settings['alg'] == 'pmd':
        pmd.train(settings)
    if settings['alg'] == 'spmd':
        spmd.train(settings)
    elif settings['alg'] == 'policyiter':
        policyiter.train(settings)
    else:
        print("Unknown alg %s" % settings['alg'])

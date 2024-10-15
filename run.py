import argparse
import yaml

from pg_termination import pmd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    args = parser.parse_args()

    with open(args.settings) as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    pmd.train(settings)

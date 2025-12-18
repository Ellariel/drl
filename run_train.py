import os
import time
import argparse

from subprocess import Popen as new, CREATE_NEW_CONSOLE


def run_scenarios(approach, sample, base_dir, windows=False):
    threads = []
    idxs = list(range(10))
    params = dict() if not windows else dict(creationflags=CREATE_NEW_CONSOLE)
    print("\n<-----running scenario----->")
    for idx in idxs:
        threads.append(
            new(
                f"uv run {os.path.join(base_dir, 'train.py')} --approach {approach} --sample {sample} --idx {idx}",
                **params,
            )
        )
        time.sleep(5)
    [i.wait() for i in threads]


parser = argparse.ArgumentParser()
parser.add_argument("--approach", default="PPO", type=str)
parser.add_argument("--sample", default=50, type=int)
parser.add_argument("--dir", default=None, type=str)
args = parser.parse_args()
base_dir = os.path.dirname(__file__) if args.dir is None else args.dir

run_scenarios(args.approach, args.sample, base_dir)

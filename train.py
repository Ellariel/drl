import os
import time
import pickle
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.env_util import make_vec_env
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

methods = {
    "PPO": PPO,
    "A2C": A2C,
    "TD3": TD3,
    "SAC": SAC,
    "DDPG": DDPG,
}

parser = argparse.ArgumentParser()
parser.add_argument("--approach", default="PPO", type=str)
parser.add_argument("--n_envs", default=8, type=int)
parser.add_argument("--env", default="env", type=str)

parser.add_argument("--sample", default=50, type=int)
parser.add_argument("--idx", default=0, type=int)

parser.add_argument("--attempts", default=100, type=int)
parser.add_argument("--epochs", default=1000, type=int)
parser.add_argument("--timesteps", default=1e5, type=int)

args = parser.parse_args()

idx = args.idx
n_envs = args.n_envs
timesteps = args.timesteps
approach = args.approach
epochs = args.epochs
sample = args.sample
attempts = args.attempts

if args.env == "env":
    version = "env"
    from env import LNEnv


def max_neighbors(G):
    def neighbors_count(G, id):
        return len(list(G.neighbors(id)))

    max_neighbors = 0
    for id in G.nodes:
        max_neighbors = max(max_neighbors, neighbors_count(G, id))
    return max_neighbors


def test_path(u, v, amount=100):
    E_.subset = [(u, v, amount)]
    obs, _ = E_.reset()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = E_.step(action)
    if E_.check_path():
        return E_.get_path()


def mmm(alist: list):
    if len(alist):
        return f"{np.min(alist):.1f}", f"{np.mean(alist):.1f}", f"{np.max(alist):.1f}"
    return "-", "-", "-"


base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "data"))
weights_dir = os.path.abspath(os.path.join(base_dir, "weights"))
results_dir = os.path.abspath(os.path.join(base_dir, "results"))
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)


with open(os.path.join(data_dir, "samples.pkl"), "rb") as f:
    samples = pickle.load(f)
    print(f"available samples: {', '.join([str(i) for i in samples.keys()])}")

S = samples[sample]
print(f"{sample}:")
for id, g in enumerate(S["g"]):
    print(
        f"idx: {id}, n: {len(g.nodes)}, e: {len(g.edges)}, max_neighbors: {max_neighbors(g)}"
    )

G = S["g"][idx]
T = S["t"][idx]

train_size = int(len(T) * 0.6)
test_size = int(len(T) * 0.2)
train_set = T[:train_size]
test_set = T[train_size : train_size + test_size]
valid_set = T[train_size + test_size :]

print(
    f"subgraph, n: {len(G.nodes)}, e: {len(G.edges)}, max neighbors: {max_neighbors(G)}, sample idx: {idx}"
)
print(
    f"transations count: {len(T)}, train_set: {len(train_set)}, test_set: {len(test_set)}, valid_set: {len(valid_set)}"
)
file_mask = f"{approach}-{version}-{n_envs}-{sample}-{idx}"

learning_rate = 0.000001

for a in range(attempts):
    print(
        f"approach: {approach}, env: {version}, n_envs: {n_envs}, sample: {sample}, idx: {idx}"
    )
    print(f"train: {file_mask}")

    E_ = LNEnv(G, [], train=False)
    E = make_vec_env(lambda: LNEnv(G, train_set), n_envs=n_envs)

    lf = os.path.join(results_dir, f"{file_mask}.log.zip")
    log = pd.read_csv(lf, sep=";", compression="zip") if os.path.exists(lf) else None
    f = os.path.join(weights_dir, f"{file_mask}.sav")
    model_class = methods[approach]

    if os.path.exists(f) and model_class:
        try:
            model = model_class.load(
                f, E, force_reset=False, verbose=0, learning_rate=learning_rate
            )
            print(f"model is loaded {approach}: {f}")
        except:  # noqa: E722
            model = model_class.load(
                f + ".tmp", E, force_reset=False, verbose=0, learning_rate=learning_rate
            )
            print(f"model is loaded {approach}: {f + '.tmp'}")
    else:
        print(f"did not find {approach}: {f}")
        model = model_class("MlpPolicy", E, verbose=0, learning_rate=learning_rate)
    for epoch in range(1, epochs + 1):
        model.learn(total_timesteps=timesteps, progress_bar=True)

        train_score = 0
        train_total_pathlen = []
        for tx in tqdm(train_set, leave=False):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                train_score += 1
                train_total_pathlen += [len(r)]
        train_score = train_score / len(train_set)

        test_score = 0
        test_total_pathlen = []
        for tx in tqdm(test_set, leave=False):
            r = test_path(tx[0], tx[1], tx[2])
            if r:
                test_score += 1
                test_total_pathlen += [len(r)]
        test_score = test_score / len(test_set)

        reward = E.env_method("get_reward")
        mean_reward = np.mean(reward, axis=1)
        max_mean_reward = np.max(mean_reward)

        test_total_pathlen_ = mmm(test_total_pathlen)
        train_total_pathlen_ = mmm(train_total_pathlen)
        print(f"n_envs: {n_envs}, epoch: {epoch}/{epochs}, attempt: {a}/{attempts}")
        print(
            f"test score: {test_score}, pathlen min: {test_total_pathlen_[0]} average: {test_total_pathlen_[1]} max: {test_total_pathlen_[2]}"
        )
        print(
            f"train score: {train_score}, pathlen min: {train_total_pathlen_[0]} average: {train_total_pathlen_[1]} max: {train_total_pathlen_[2]}"
        )
        print(f"max mean reward: {max_mean_reward:.3f}~{mean_reward}")

        if os.path.exists(f):
            shutil.move(f, f + ".tmp")
        model.save(f)

        if max(train_score, test_score) > 0.5:
            model.save(f + f"-{train_score:.3f}-{test_score:.3f}")
            print("saved:", f + f"-{train_score:.3f}-{test_score:.3f}")

        total_pathlen = test_total_pathlen + train_total_pathlen
        total_pathlen_ = mmm(total_pathlen)

        log = pd.concat(
            [
                log,
                pd.DataFrame.from_dict(
                    {
                        "time": time.time(),
                        "approach": approach,
                        "version": version,
                        "n_envs": n_envs,
                        "sample": sample,
                        "idx": idx,
                        "max_mean_reward": max_mean_reward,
                        "mean_reward": mean_reward,
                        "test_score": test_score,
                        "train_score": train_score,
                        "epoch": epoch,
                        "epochs": epochs,
                        "attempt": a,
                        "total_timesteps": timesteps,
                        "filename": f,
                        "n": len(G.nodes),
                        "e": len(G.edges),
                        "max_neighbors": max_neighbors(G),
                        "min_pathlen": total_pathlen_[0],
                        "max_pathlen": total_pathlen_[2],
                        "avg_pathlen": total_pathlen_[1],
                    },
                    orient="index",
                ).T,
            ],
            ignore_index=True,
        )
        log.to_csv(lf, sep=";", index=False, compression="zip")

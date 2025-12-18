import os
import copy
import pickle
import random
import networkx as nx
from tqdm import tqdm
from littleballoffur import ForestFireSampler


from proto import (
    gen_txset,
    random_amount,
    set_seed,
    MIN_AGE,
    MAX_AGE,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def sample_graph(g, sampler=None):
    g = nx.convert_node_labels_to_integers(g, label_attribute="old_id")
    s = sampler.sample(g)
    mapping = {k: g.nodes[k]["old_id"] for k in s.nodes()}
    return nx.relabel_nodes(s, mapping)


base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "data"))
results_dir = os.path.abspath(os.path.join(base_dir, "results"))
os.makedirs(results_dir, exist_ok=True)

print("data_dir:", data_dir)
print("results_dir:", results_dir)

seed = set_seed(13)

if not os.path.exists(os.path.join(data_dir, "snapshot.gml")):
    g = nx.read_gml(os.path.join(data_dir, "20230716.gml.geo")).to_undirected()
    print("graph preparation...")
    for e in g.edges:
        e = g.edges[e]
        e["fee_base_sat"] = float(e["fee_base_msat"]) / 1000
        e["fee_rate_sat"] = float(e["fee_proportional_millionths"]) / 10**6
        if "htlc_maximum_msat" not in e:
            e["htlc_maximum_msat"] = random_amount() * 1000 + float(
                e["htlc_minimim_msat"]
            )
        else:
            e["htlc_maximum_msat"] = float(e["htlc_maximum_msat"])
        e["capacity_sat"] = e["htlc_maximum_msat"] / 1000
        e["delay"] = float(e["cltv_expiry_delta"])
        e["age"] = random.randrange(MIN_AGE, MAX_AGE)
    nx.write_gml(g, os.path.join(data_dir, "snapshot.gml"))
    print("graph preparation is done.")
g = nx.read_gml(os.path.join(data_dir, "snapshot.gml")).to_undirected()
print(g)

if not os.path.exists(os.path.join(data_dir, "samples.pkl")):
    samples = {}
    for size in [50, 100, 300, 500, 1000]:
        samples[size] = {"g": [], "t": []}
        g_ = copy.deepcopy(g)
        seed = set_seed(13)
        sampler = ForestFireSampler(number_of_nodes=size, seed=seed)
        for i in tqdm(range(10), desc=f"n={size}", leave=False):
            s: nx.Graph = sample_graph(g_, sampler)
            t = gen_txset(s, seed=seed)
            samples[size]["g"] += [s]
            samples[size]["t"] += [t]

    with open(os.path.join(data_dir, "samples.pkl"), "wb") as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

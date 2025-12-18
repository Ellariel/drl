import random
import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.stats import entropy
import requests


def set_seed(seed=13):
    random.seed(seed)
    np.random.seed(seed)
    return seed


def normalize(x, min, max):
    x = float(x)
    if x <= min:
        return 0.0
    if x > max:
        return 0.99999
    return (x - min) / (max - min)


def average_degree(g, method=np.mean):
    try:
        return method(list(dict(g.degree).values()))
    except Exception as e:
        print(e)


def degree_distribution_entropy(g, base=2):
    try:
        degrees = np.array(g.degree)[:, 1].astype("float64")
        values, counts = np.unique(degrees, return_counts=True)
        pk = counts / counts.sum()
        h = entropy(pk, base=base)
        k = len(values)
        h_norm = h / np.log(k) * np.log(base)
        return h, h_norm
    except Exception as e:
        print(e)


# Retrieves current block height from API
# in case of fail, will return a default block height
def getBlockHeight(default=True):
    if default:
        return 697000
    API_URL = "https://api.blockcypher.com/v1/btc/main"
    try:
        CBR = requests.get(API_URL).json()["height"]
        print("Block height used:", CBR)
        return CBR
    except:  # noqa: E722
        print("Block height not found, using default 697000")
        return 697000


### GENERAL
BASE_TIMESTAMP = 1681234596.2736187
BLOCK_HEIGHT = getBlockHeight()
### LND
LND_RISK_FACTOR = 0.000000015
A_PRIORI_PROB = 0.6
### ECL
MIN_AGE = 505149
MAX_AGE = BLOCK_HEIGHT
MIN_DELAY = 9
MAX_DELAY = 2016
MIN_CAP = 1
MAX_CAP = 100000000
DELAY_RATIO = 0.15
CAPACITY_RATIO = 0.5
AGE_RATIO = 0.35
### CLN
C_RISK_FACTOR = 10
RISK_BIAS = 1
DEFAULT_FUZZ = 0.05
FUZZ = random.uniform(-1, 1)
LOG_SPACE = np.logspace(0, 7, 10**6)


def cost_function(G, u, v, amount, proto_type="LND"):
    fee = G.edges[u, v]["fee_base_sat"] + amount * G.edges[u, v]["fee_rate_sat"]
    if proto_type == "LND":
        cost = (amount + fee) * G.edges[u, v][
            "delay"
        ] * LND_RISK_FACTOR + fee  # + calc_bias(G.edges[u, v]['last_failure'])*1e6
        # we don't consider failure heuristic at this point
    elif proto_type == "ECL":
        n_capacity = 1 - (normalize(G.edges[u, v]["capacity_sat"], MIN_CAP, MAX_CAP))
        n_age = normalize(BLOCK_HEIGHT - G.edges[u, v]["age"], MIN_AGE, MAX_AGE)
        n_delay = normalize(G.edges[u, v]["delay"], MIN_DELAY, MAX_DELAY)
        cost = fee * (
            n_delay * DELAY_RATIO + n_capacity * CAPACITY_RATIO + n_age * AGE_RATIO
        )

    elif proto_type == "CLN":
        fee = fee * (1 + DEFAULT_FUZZ * FUZZ)
        cost = (amount + fee) * G.edges[u, v]["delay"] * C_RISK_FACTOR + RISK_BIAS

    else:
        cost = 1
    cost = 0 if cost < 0 else cost
    return cost


def get_shortest_path(G, u, v, amount, proto_type="LND"):
    def weight_function(u, v, d):
        return cost_function(G, u, v, amount, proto_type=proto_type)

    # return nx.shortest_path(G, u, v, weight=weight_function)
    try:
        return nx.shortest_path(G, u, v, weight=weight_function)
    except:  # noqa: E722
        return []


def random_amount():  # SAT
    # Возвращает массив значений от 10^0 = 1 до 10^7, равномерно распределенных на логарифмической шкале
    # https://coingate.com/blog/post/lightning-network-bitcoin-stats-progress
    # The highest transaction processed is 0.03967739 BTC, while the lowest is 0.000001 BTC. The average payment size is 0.00508484 BTC;
    # highest: 3967739.0 SAT
    # average: 508484.0 SAT
    # lowest: 100.0 SAT
    return LOG_SPACE[random.randrange(0, 10**6)]


def gen_txset(G, transacitons_count=1000, seed=47):
    def shortest_path_len(u, v):
        path_len = 0
        try:
            path_len = nx.shortest_path_length(G, u, v)
        except:  # noqa: E722
            pass
        return path_len

    random.seed(seed)
    np.random.seed(seed)

    tx_set = []
    nodes = list(G.nodes)
    max_path_length = 0
    for _ in tqdm(range(1, transacitons_count + 1), leave=False):
        while True:
            u = nodes[random.randrange(0, len(nodes))]
            v = nodes[random.randrange(0, len(nodes))]
            p = shortest_path_len(u, v)
            max_path_length = max(max_path_length, p)
            if v != u and p >= 2 and (u, v) not in tx_set and (v, u) not in tx_set:
                break
        tx_set.append((u, v))
    tx_set = [(tx[0], tx[1], random_amount() + 100) for tx in tx_set]
    # print(f'max_path_length: {max_path_length}')
    return tx_set

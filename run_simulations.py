import logging
import multiprocessing as mp
from itertools import product
from pathlib import Path

import pandas as pd
import networkx as nx

from simulations.diffusion_of_innovation_on_graph import diffusion_of_innovation_model_on_graph

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

PATH_BASEDIR = Path('results')

# CONFIGURATION

P = [0.5]
F = [0.1]
Q = [3, 4]
H = [0.01]

T = 1000
N_INDEPENDENT_RUNS = 100

GRAPHS = {
    "lattice": {
        "generator": nx.grid_2d_graph,
        "args": {
            "n": 100,
            "m": 100
        }
    },
    "complete": {
        "generator": nx.complete_graph,
        "args": {
            "n": 100
        }
    }
}


def calculate(graph, p, f, q, h, t, n_independent_runs):
    current_graph_path = f"{graph}/p-{p}/f-{f}/q-{q}/h-{h}"
    logging.info(current_graph_path)

    current_path = Path(PATH_BASEDIR).joinpath(current_graph_path)
    current_path.mkdir(parents=True, exist_ok=True)

    for i in range(n_independent_runs):
        generator = GRAPHS[graph]['generator']
        args = GRAPHS[graph]['args']

        g = generator(**args)
        results = diffusion_of_innovation_model_on_graph(g, p, f, q, h, t)
        results = pd.DataFrame(results)
        results_path = current_path.joinpath(f'{i}.csv')
        results.to_csv(results_path, index=False)


if __name__ == '__main__':
    PATH_BASEDIR.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool(processes=mp.cpu_count())

    for graph, p, f, q, h in product(GRAPHS, P, F, Q, H):
        pool.apply_async(calculate, args=(graph, p, f, q, h, T, N_INDEPENDENT_RUNS))

    pool.close()
    pool.join()

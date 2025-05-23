from mesa.batchrunner import batch_run
from model import MisinformationNetwork
import numpy as np
import json
import random

seeds = [random.randint(1, 1000) for _ in range(100)]

params = {"seed": seeds,
          "num_residents": 100,
        #   "avg_node_degree": [5, 10, 15, 20],
        #   "seed_mode": ['high_betweenness', 'high_degree', 'peripheral_betweenness', 'peripheral_degree', 'staff'], 
        #   "network_type": ['random weight'], 
        #   "alpha_dl": [1, 2, 3, 4, 5],
        #   "alpha_cognitive": [1, 2, 3, 4, 5],
        #   "fact_checking_prob":  np.arange(0.05, 0.3, 0.05),
        #   "confidence_deprecation_rate": np.arange(0.1, 1.0, 0.1)
        }

if __name__ == '__main__':
    results = batch_run(
        MisinformationNetwork,
        parameters=params,
        iterations=1,
        max_steps=1000,
        number_processes=None,
        data_collection_period=1,
        display_progress=True,
    )
    with open("model_output.json", "w") as f:
        json.dump(results, f, indent=4)
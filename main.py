""" Run experiments."""
import sys

import ray
import ray.tune

from base_experiment import Experiment



if __name__ == "__main__":
    GPUS = 1
    RUN_NAME = 'MNIST_Depth_Comparison_with_relu'

    possible_dense_layers = [
        ((5, 20), (1, 100), (1, 100)),
        ((1, 100), (5, 20), (1, 100)),
        ((1, 100), (1, 100), (5, 20)),
    ]

    CONFIG = {
        "eval_metric": "class/accuracy",
        "batch_size": 128,
        "m_steps": ray.tune.grid_search([10, 20, 30]),
        "optimizer": 'RMSPropOptimizer',
        "sample_size": 8,
        "learning_rate": 0.001,
        "dense_layers": ray.tune.grid_search(possible_dense_layers),
        "sampled_evals": True,
    }

    train_spec = {
        "run": RUN_NAME,
        "trial_resources": {'cpu': 4, 'gpu': GPUS},
        "stop": {"training_iteration": 200},
        "config": CONFIG,
    }

    ray.tune.register_trainable(RUN_NAME, Experiment)
    if len(sys.argv) > 1:
        ray.init(redis_address=sys.argv[1])
    else:
        ray.init()
    ray.tune.run_experiments({RUN_NAME: train_spec})

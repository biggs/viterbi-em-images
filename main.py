""" Run simple experiment."""
import ray
import ray.tune

from base_experiment import Experiment

def single_ray_experiment(name, config, gpus):
    train_spec = {
        "run": name,
        "trial_resources": {'cpu': 4, 'gpu': gpus},
        "stop": {"training_iteration": 100},
        "config": config,
    }
    ray.tune.register_trainable(name, Experiment)
    ray.init()
    ray.tune.run_experiments({name: train_spec})



if __name__ == "__main__":
    GPUS = 1

    RUN_NAME = 'MNIST_Depth_Comparison'

    possible_dense_layers = [
        ((1, 100), (1, 100), (1, 100)),
        ((1, 50), (1, 100), (1, 100)),
        ((1, 100), (1, 50), (1, 100)),
        ((1, 100), (1, 100), (1, 50)),
    ]

    CONFIG = {
        "eval_metric": "class/accuracy",
        "batch_size": 128,
        "m_steps": ray.tune.grid_search([20, 30, 40, 50]),
        "optimizer": 'RMSPropOptimizer',
        "sample_size": 10,
        "learning_rate": ray.tune.grid_search([0.001, 0.005]),
        "dense_layers": ray.tune.grid_search(possible_dense_layers),
        "sampled_evals": True,
    }


    single_ray_experiment(RUN_NAME, CONFIG, GPUS)

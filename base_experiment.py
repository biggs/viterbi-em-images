""" Classes for running and logging an experiment."""
import numpy as np
import tensorflow as tf

from ray.tune import Trainable
from ray.tune import TrainingResult

import base_model
from load_data import MnistLoader
from utils import AttrDict


class Experiment(Trainable):

    def _setup(self):
        config = AttrDict(self.config)
        self.data = MnistLoader("~/data", config.batch_size)

        self.model = getattr(base_model, "DenseImageModel")
        self.model = self.model(self.data, config)

        self.logger = Logger(self.logdir)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.dict_train = self.data.get_dict_train(self.sess)

        self.num_complete_e = 0

    def _evaluations_valid(self, evaluations):
        dict_valid = self.data.get_dict_valid(self.sess)
        evaluations_sum, proportions_sum = self.sess.run(evaluations, dict_valid)
        proportions_sum = [np.array(p) for p in proportions_sum]
        count = 1
        while True:
            try:
                evaluation, proportions = self.sess.run(evaluations, dict_valid)
                for key, val in evaluation.items():
                    evaluations_sum[key] += val
                proportions_sum = [s + np.array(p) for s, p in
                                   zip(proportions_sum, proportions)]
                count += 1
            except tf.errors.OutOfRangeError:
                break
        evals = {key: val / count for key, val in evaluations_sum.items()}
        proportions = [p / count for p in proportions_sum]
        return evals, proportions

    def _train_steps(self, steps):
        """ Trains for steps, returning evaluations from the final train step."""
        # Train alternating E and M steps, completing with an M step.
        m_steps = self.config["m_steps"]
        for i in range(steps - 1):
            step = self.model.e_step if i % m_steps == 0 else self.model.m_step
            self.sess.run(step, self.dict_train)

        evals, _ = self.sess.run([self.model.mode_evaluations, self.model.m_step],
                                 self.dict_train)
        return evals

    def _complete_e_step(self):
        dict_train_complete = self.data.get_dict_train_complete(self.sess)
        while True:
            try:
                self.sess.run(self.model.e_step, dict_train_complete)
            except tf.errors.OutOfRangeError:
                break

    def _train(self):
        train_steps = self.data.train_epoch_size

        # Complete E Step if needed.
        if self._iteration % self.config["epochs_per_complete_e"] == 0:
            print('Complete E Step')
            self.num_complete_e += 1
            self._complete_e_step()

        # Run training and get evaluation.
        evals_train, proportions_train = self._train_steps(train_steps)

        # Run Validation Evaluations.
        evals_valid = (self.model.evaluations if self.config["sampled_evals"]
                       else self.model.mode_evaluations)
        evals_valid, proportions_valid = self._evaluations_valid(evals_valid)

        # Write logs.
        steps = self._timesteps_total + train_steps
        self.logger.log_all(evals_train, steps, True)
        self.logger.log_all(evals_valid, steps)
        valid_modular_selection = [p for p in proportions_valid  # only for no_modules > 1.
                                   if np.prod(p.shape) > 1]
        self.logger.log_selections(valid_modular_selection)

        info = {"config": self.config,
                "evaluations_train": evals_train,
                "evaluations_valid": evals_valid}
        return TrainingResult(timesteps_this_iter=train_steps, info=info)

    def _stop(self):
        self.sess.close()

    def _save(self, checkpoint_dir):
        path = checkpoint_dir + "/save"
        return self.saver.save(self.sess, path, global_step=self._iteration)

    def _restore(self, checkpoint_path):
        return self.saver.restore(self.sess, checkpoint_path)





class Logger(object):
    """Logging in tensorboard without tensorflow ops.

    https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer_valid = tf.summary.FileWriter(log_dir + '/valid')
        self.writer_train = tf.summary.FileWriter(log_dir + '/train')
        self.selections_valid = False
        self.selections_file = log_dir + '/selections_valid.txt'

    def log_all(self, evaluations, step, train_step=False):
        values = [tf.Summary.Value(tag=tag, simple_value=value)
                  for tag, value in evaluations.items()]
        summary = tf.Summary(value=values)
        writer = self.writer_train if train_step else self.writer_valid
        writer.add_summary(summary, step)

    def log_selections(self, values):
        if not len(values) > 0:
            return
        values = values[0]  # only plot first modular layer.
        print('Test selections:', values)
        if self.selections_valid is False:
            self.selections_valid = values
        else:
            self.selections_valid = np.concatenate((self.selections_valid, values))
        np.savetxt(self.selections_file, self.selections_valid)




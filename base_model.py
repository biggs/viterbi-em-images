import tensorflow as tf
from tensorflow.contrib import distributions as tfd

from modular import ModularContext
from modular import ModularMode
from modular import modularize_target
from modular import modular_layer
from modular import average_over_samples
from modular import max_over_samples
from create_modules import create_dense_modules

from decorators import template
from decorators import define_scope
from utils import get_shape



class BaseImageModel(object):

    def __init__(self, data, config):
        # Define the model interface.
        self.inputs = data.inputs
        self.labels = data.labels
        self.data_indices = data.data_indices
        self.num_labels = data.num_labels
        self.dataset_size = data.dataset_size
        self.is_training = data.is_training
        self.config = config

        self.e_step, self.m_step, self.evaluations  # pylint: disable=pointless-statement

    @define_scope
    def e_step(self):
        context = ModularContext(ModularMode.E_STEP, self.data_indices,
                                 self.dataset_size, self.config.sample_size)
        # batch_size * sample_size
        llh = self.llh(tfd.Categorical(self.logits(context)), context)
        logprob = context.selection_logprob() + llh
        logprob = tf.reshape(logprob, [self.config.sample_size, -1])
        best_selection_indices = tf.stop_gradient(tf.argmax(logprob, axis=0))
        return context.update_best_selection(best_selection_indices)

    @define_scope
    def m_step(self):
        context = ModularContext(
            ModularMode.M_STEP, self.data_indices, self.dataset_size)
        objective = self.llh(tfd.Categorical(self.logits(context)), context)
        selection_logprob = context.selection_logprob()

        ctrl_objective = -tf.reduce_mean(selection_logprob)
        module_objective = -tf.reduce_mean(objective)
        joint_objective = ctrl_objective + module_objective

        optimizer = getattr(tf.train, self.config.optimizer)
        optimizer = optimizer(self.config.learning_rate)
        return optimizer.minimize(joint_objective)

    @template
    def llh(self, outputs, context: ModularContext):
        target = modularize_target(self.labels, context)
        return outputs.log_prob(target)

    @template
    def accuracy(self, outputs, context: ModularContext):
        target = modularize_target(self.labels, context)
        is_correct = tf.equal(outputs.mode(), target)
        return tf.reduce_mean(tf.cast(is_correct, tf.float32))


    @define_scope
    def mode_evaluations(self):
        context = ModularContext(ModularMode.MODE_EVALUATION)
        outputs = tfd.Categorical(logits=self.logits(context))
        evaluations = {
            "loglikelihood/mode": tf.reduce_mean(self.llh(outputs, context)),
            "accuracy/mode": self.accuracy(outputs, context),
            "entropy/selection": context.selection_entropy(),
            "entropy/batch": context.batch_selection_entropy(),
        }
        module_proportions = context.module_proportions()
        return evaluations, module_proportions

    @template
    def sampled_evaluations(self, reducing_function):
        context = ModularContext(ModularMode.SAMPLES_EVALUATION)
        logits = self.logits(context)
        reduced_probs = reducing_function(tf.nn.softmax(logits), context)
        outputs = tfd.Categorical(probs=reduced_probs)
        llh = tf.reduce_mean(self.llh(outputs, context))
        acc = self.accuracy(outputs, context)
        return llh, acc

    @define_scope
    def bayesian_evaluations(self):
        llh, acc = self.sampled_evaluations(average_over_samples)
        return {"loglikelihood/ensemb": llh, "accuracy/ensemb": acc}

    @define_scope
    def max_evaluations(self):
        llh, acc = self.sampled_evaluations(max_over_samples)
        return {"loglikelihood/max": llh, "accuracy/max": acc}

    @define_scope
    def evaluations(self):
        mode_evals, proportions = self.mode_evaluations
        return ({**mode_evals, **self.bayesian_evaluations,
                **self.max_evaluations}, proportions)

    def logits(self, context: ModularContext):
        pass




class DenseImageModel(BaseImageModel):

    @template
    def logits(self, context: ModularContext):
        x = tf.layers.flatten(self.inputs)
        for modules, units in self.config.dense_layers:
            x = _dense_modular_layer(x, modules, units, context)
            x = tf.nn.relu(x)
        return tf.layers.dense(x, self.num_labels)


def _dense_modular_layer(x, module_count: int, units: int,
                         context: ModularContext):
    modules = create_dense_modules(x, module_count, units)
    return modular_layer(x, modules, 1, context)

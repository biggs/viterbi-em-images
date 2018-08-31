""" Manage and run Viterbi-EM modular layers."""
from collections import namedtuple
from enum import Enum

import tensorflow as tf
from tensorflow.contrib import distributions as tfd

from utils import gather_each
from utils import get_shape

ModularMode = Enum('ModularMode', 'E_STEP M_STEP MODE_EVALUATION SAMPLES_EVALUATION')
ModularLayerAttributes = namedtuple(
    'ModularLayerAttributes', ['selection', 'best_selection', 'controller'])
ModulePool = namedtuple(
    'ModulePool', ['module_count', 'module_fnc', 'output_shape'])



class ModularContext:
    """ Manager for modular layers. One per mode."""

    def __init__(self, mode: ModularMode, data_indices=None,
                 dataset_size: int = None, sample_size: int = 1):
        self.mode = mode
        self.data_indices = data_indices
        self.dataset_size = dataset_size
        self.sample_size = sample_size
        self.samples = False
        self.layers = []

    def begin_modular(self, inputs):
        sampled = (self.mode == ModularMode.E_STEP or
                   self.mode == ModularMode.SAMPLES_EVALUATION)
        if sampled and not self.samples:
            self.samples = True
            rank = inputs.shape.ndims
            return tf.tile(inputs, [self.sample_size] + [1] * (rank - 1))
        return inputs

    def selection_entropy(self):
        return tf.reduce_mean([tf.reduce_mean(layer.controller.entropy())
                               for layer in self.layers])

    def batch_selection_entropy(self):
        def layer_entropy(layer):
            probs = tf.reduce_mean(layer.controller.probs, axis=0)
            return -tf.reduce_sum(probs * tf.log(probs + 1e-30), axis=-1)
        return tf.reduce_mean([layer_entropy(layer) for layer in self.layers])

    def module_proportions(self):
        def layer_proportion(layer):
            ctrl = layer.controller
            selection = tf.one_hot(ctrl.mode(), ctrl.event_size)
            return tf.reduce_mean(selection, axis=0)
        return [layer_proportion(layer) for layer in self.layers]

    def selection_logprob(self):
        x = [tf.reduce_sum(layer.controller.log_prob(layer.selection), axis=-1)
             for layer in self.layers]
        return tf.reduce_sum(x, axis=0)

    def update_best_selection(self, best_selection_indices):
        def update(layer):
            selection = tf.reshape(layer.selection, [self.sample_size, -1] \
                                   + layer.selection.shape[1:].as_list())
            new_best_selection = gather_each(selection, best_selection_indices)
            return tf.scatter_update(layer.best_selection, self.data_indices,
                                     new_best_selection)
        return tf.group(*(update(layer) for layer in self.layers))



def run_modules(inputs, selection, module_fnc, output_shape):
    batch_size = tf.shape(inputs)[0]
    if output_shape is not None:
        output_shape = [batch_size] + output_shape
    else:
        # This is the only way I am aware of to get the output shape easily.
        dummy = module_fnc(inputs, 0)
        output_shape = [batch_size] + dummy.shape[1:].as_list()

    used_modules, _ = tf.unique(tf.reshape(selection, (-1,)))

    def compute_module(accum, module):
        mask = tf.equal(module, selection)
        reduced_mask = tf.reduce_any(mask, axis=-1)
        indices = tf.where(reduced_mask)
        affected_inp = tf.boolean_mask(inputs, reduced_mask)
        output = module_fnc(affected_inp, module)
        return accum + tf.scatter_nd(
            indices, output, tf.cast(output_shape, tf.int64))

    # Go through all used_modules and compute, masking unused data.
    output = tf.scan(compute_module, used_modules,
                     initializer=tf.zeros(output_shape))[-1]
    return output



def get_ctrl(inputs, module_count, parallel_count):
    """ Control distribution for modular layer."""
    flat_inputs = tf.layers.flatten(inputs)
    logits = tf.layers.dense(flat_inputs, module_count * parallel_count)
    logits = tf.reshape(logits, [-1, parallel_count, module_count])
    return tfd.Categorical(logits)

def get_best_selection_persistent(module_count, parallel_count, dataset_size):
    """ Variable to store persistent best module choice for layer."""
    init = tf.random_uniform_initializer(maxval=module_count, dtype=tf.int32)
    shape = [dataset_size, parallel_count]
    return tf.get_variable('best_selection', shape, tf.int32, init)


def modular_layer(inputs, modules: ModulePool, parallel_count: int,
                  context: ModularContext):
    """ Create a modular layer and add to context."""

    with tf.variable_scope(None, 'modular_layer'):
        inputs = context.begin_modular(inputs)  # tile inputs on first layer.
        ctrl = get_ctrl(inputs, modules.module_count, parallel_count)
        saved_selections = get_best_selection_persistent(
            modules.module_count, parallel_count, context.dataset_size)

        if context.mode == ModularMode.E_STEP:
            best_selection = tf.gather(saved_selections, context.data_indices)
            # sample_size x batch_size x 1
            sampled_selection = tf.reshape(
                ctrl.sample(), [context.sample_size, -1, parallel_count])
            selection = tf.concat(
                [best_selection[tf.newaxis], sampled_selection[1:]], axis=0)
            selection = tf.reshape(selection, [-1, parallel_count])
        elif context.mode == ModularMode.M_STEP:
            selection = tf.gather(saved_selections, context.data_indices)
        elif context.mode == ModularMode.MODE_EVALUATION:
            selection = ctrl.mode()
        elif context.mode == ModularMode.SAMPLES_EVALUATION:
            selection = ctrl.sample()
        else:
            raise ValueError('Invalid modular mode')

        attrs = ModularLayerAttributes(selection, saved_selections, ctrl)
        context.layers.append(attrs)

        return run_modules(inputs, selection, modules.module_fnc, modules.output_shape)



def modularize_target(target, context: ModularContext):
    """ Tile labels in E Step for comparison of likelihoods."""
    if context.mode == ModularMode.E_STEP:
        rank = target.shape.ndims
        return tf.tile(target, [context.sample_size] + [1] * (rank - 1))
    return target

def average_over_samples(inputs, context: ModularContext):
    """ Average the value of inputs over samples."""
    if context.mode == ModularMode.SAMPLES_EVALUATION:
        shape = get_shape(inputs)
        inputs = tf.reshape(inputs, [context.sample_size, -1] + shape[1:])
        return tf.reduce_mean(inputs, axis=0)
    return inputs

def max_over_samples(inputs, context: ModularContext):
    """ Maximise the value of inputs over samples."""
    if context.mode == ModularMode.SAMPLES_EVALUATION:
        shape = get_shape(inputs)
        inputs = tf.reshape(inputs, [context.sample_size, -1] + shape[1:])
        return tf.reduce_max(inputs, axis=0)
    return inputs

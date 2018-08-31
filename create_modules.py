from collections import namedtuple
import tensorflow as tf
from modular import ModulePool


def create_dense_modules(inputs_or_shape, module_count: int, units: int = None):
    with tf.variable_scope(None, 'dense_modules'):
        if hasattr(inputs_or_shape, 'shape') and units is not None:
            weights_shape = [module_count, inputs_or_shape.shape[-1].value, units]
        else:
            weights_shape = [module_count] + inputs_or_shape
            units = inputs_or_shape[-1]
        weights = tf.get_variable('weights', weights_shape)
        biases_shape = [module_count, units]
        biases = tf.get_variable(
            'biases', biases_shape, initializer=tf.zeros_initializer())

        def module_fnc(x, a):
            return tf.matmul(x, weights[a]) + biases[a]

        return ModulePool(module_count, module_fnc, output_shape=[units])


def create_conv_modules(shape, module_count: int, strides, padding='SAME'):
    with tf.variable_scope(None, 'conv_modules'):
        filter_shape = [module_count] + list(shape)
        filter = tf.get_variable('filter', filter_shape)
        biases_shape = [module_count, shape[-1]]
        biases = tf.get_variable('biases', biases_shape,
                                 initializer=tf.zeros_initializer())

        def module_fnc(x, a):
            return tf.nn.conv2d(x, filter[a], strides, padding) + biases[a]

        return ModulePool(module_count, module_fnc, output_shape=None)


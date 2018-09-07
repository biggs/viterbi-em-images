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



def create_fused_modules(inputs, conv_shape, module_units: int, module_count: int):
    """ Convolution plus dense modules. Module_units is output size."""
    # Fixed strides and padding to allow easy flattened_size calculation.
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    flattened_size = inputs.shape[1] * inputs.shape[2] * conv_shape[-1]

    filter_shape = [module_count] + list(conv_shape)
    biases_conv_shape = [module_count, conv_shape[-1]]
    weights_shape = [module_count, flattened_size, module_units]
    biases_dense_shape = [module_count, module_units]

    with tf.variable_scope(None, 'fused_modules'):
        # Convolutional
        filter = tf.get_variable('filter', filter_shape)
        biases_conv = tf.get_variable('biases_conv', biases_conv_shape,
                                      initializer=tf.zeros_initializer())

        # Dense
        weights = tf.get_variable('weights', weights_shape)
        biases_flat = tf.get_variable('biases_dense', biases_dense_shape,
                                      initializer=tf.zeros_initializer())

        def module_fnc(x, a):
            x = tf.nn.conv2d(x, filter[a], strides, padding) + biases_conv[a]
            x = tf.reshape(x, [-1, flattened_size])
            x = tf.matmul(x, weights[a]) + biases_flat[a]
            return x

        return ModulePool(module_count, module_fnc, output_shape=[module_units])

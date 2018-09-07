import tensorflow as tf
from tensorflow.contrib import distributions as tfd

def get_ctrl(ctrl_name):
    ctrl_dict = {
        'standard': ctrl_standard,
        'filter_mean': ctrl_filter_mean,
        'filter_mean_squared': ctrl_filter_mean_squared,
        'filter_mean_squared_nonlin': ctrl_filter_mean_squared_nonlin
        }
    return ctrl_dict[ctrl_name]


def ctrl(processed_inputs, module_count: int, parallel_count: int):
    """ Standard wrapper for all controls."""
    logits = tf.layers.dense(processed_inputs, module_count * parallel_count)
    logits = tf.reshape(logits, [-1, parallel_count, module_count])
    return tfd.Categorical(logits)


def ctrl_standard(inputs, module_count: int, parallel_count: int):
    """ Basic controller (for convolution or not)."""
    flat_inputs = tf.layers.flatten(inputs)
    return ctrl(flat_inputs, module_count, parallel_count)

def ctrl_filter_mean(inputs, module_count: int, parallel_count: int):
    """ Use the mean of filter values as the controller input."""
    filter_mean = tf.reduce_mean(inputs, axis=(1, 2))
    return ctrl(filter_mean, module_count, parallel_count)



# Allowing access to squared mean as well as mean
def concat_filter_mean_square_mean(inputs):
    """ Concatenate the mean and squared mean of filter values."""
    filter_mean = tf.reduce_mean(inputs, axis=(1, 2))
    filter_square_mean = tf.reduce_mean(tf.square(inputs), axis=(1, 2))
    processed_inputs = tf.concat([filter_mean, filter_square_mean], axis=1)
    return processed_inputs


def ctrl_filter_mean_squared(inputs, module_count: int, parallel_count: int):
    """ Should simulate something like allowing access to variance."""
    processed_inputs = concat_filter_mean_square_mean(inputs)
    return ctrl(processed_inputs, module_count, parallel_count)

def ctrl_filter_mean_squared_nonlin(inputs, module_count: int, parallel_count: int):
    """ Add an additional regularized nonlinearity to ctrl_filter_mean."""
    x = concat_filter_mean_square_mean(inputs)
    x = tf.layers.dense(x, 64, tf.nn.relu)
    return ctrl(x, module_count, parallel_count)

""" Basic utils."""
import tensorflow as tf
import tensorflow.contrib.distributions as tfd


def get_shape(tensor):
    """ Returns the shape of a tf.Tensor as a list.

    Args:
        tensor: A tf.Tensor.

    Returns:
        A list of the dimensions of the tensor, with static
        values where available and dynamic where not.
    """
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
            for s in zip(static_shape, dynamic_shape)]
    return dims



class AttrDict(dict):
    """ Simple dot-addressable dict."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__



def gather_each(params, indices):
    count = tf.cast(tf.shape(indices)[0], indices.dtype)
    each = tf.range(count, dtype=indices.dtype)
    indices = tf.stack([indices, each], axis=1)
    return tf.gather_nd(params, indices)

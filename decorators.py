""" Decorators for managing scopes and properties in tensorflow models."""
import functools
import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    if 'default_name' in kwargs:
        name = None
    else:
        name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


@doublewrap
def define_scope_fnc(function, scope=None, *scope_args, **scope_kwargs):
    if 'default_name' in scope_kwargs:
        name = None
    else:
        name = scope or function.__name__

    @functools.lru_cache(None)
    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        with tf.variable_scope(name, *scope_args, **scope_kwargs):
            return function(self, *args, **kwargs)
    return decorator


@doublewrap
def template(function, *template_args, **template_kwargs):

    attribute = '_template_' + function.__name__

    @functools.wraps(function)
    def decorator(self, *args, **kwargs):
        if not hasattr(self, attribute):
            template = tf.make_template(function.__name__, function,
                                        *template_args, **template_kwargs)
            setattr(self, attribute, template)
        else:
            template = getattr(self, attribute)
        return template(self, *args, **kwargs)

    return decorator

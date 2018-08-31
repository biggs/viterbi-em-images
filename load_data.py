""" load_data.py

Load datasets used for em-viterbi-images (each a wrapped
tf.DataSet with valid and train feeds).
"""
from os.path import join
from abc import ABC, abstractmethod

import numpy as np
import observations as obs
import tensorflow as tf


def _make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle



class DataLoader(object):
    """ Abstract base class. Implement _setup and _proprocess
    Public Attributes:
        self.train_epoch_size
        self.num_labels
        self.data_indices
        self.inputs
        self.labels
        self.is_training
    """
    def _load(self, data, batch_size):
        (x_train, y_train), (x_valid, y_valid) = data
        self.dataset_size = x_train.shape[0]
        self.num_labels = np.max(y_train) + 1
        self.train_epoch_size = int(self.dataset_size/batch_size)

        # Train. Random ordering.
        train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train = train._enumerate().repeat().shuffle(self.dataset_size)
        self._train = train.batch(batch_size)

        # Validation.
        valid_dummy_indices = tf.zeros([x_valid.shape[0]], dtype=tf.int64)
        y_valid = y_valid.astype(np.uint8)    # Fix datatype.
        self._valid = tf.data.Dataset.from_tensor_slices(
            (valid_dummy_indices, (x_valid, y_valid))).batch(1000)

        self._handle = tf.placeholder(tf.string, [])
        itr = tf.data.Iterator.from_string_handle(
            self._handle, self._train.output_types, self._train.output_shapes)

        self.data_indices, (inputs, labels) = itr.get_next()
        self.inputs, self.labels = self._preprocess(inputs, labels)
        self.is_training = tf.placeholder(tf.bool)

    def get_dict_train(self, sess):
        return {self._handle: _make_handle(sess, self._train),
                self.is_training: True}

    def get_dict_valid(self, sess):
        return {self._handle: _make_handle(sess, self._valid),
                self.is_training: True}

    def _preprocess(inputs, labels):
        pass




class MnistLoader(DataLoader):

    def __init__(self, datadir, batch_size):
        data = obs.mnist(join(datadir, "MNIST"))
        self._load(data, batch_size)

    def _preprocess(self, inputs, labels):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        inputs = tf.cast(inputs, tf.float32) / 255.0
        labels = tf.cast(labels, tf.int32)
        return inputs, labels



class CifarLoader(DataLoader):

    def _preprocess(self, inputs, labels):
        """ [B H W D] inputs scaled to [0,1] and labels processing."""
        inputs = tf.cast(inputs, tf.float32) / 255.0
        inputs = tf.transpose(inputs, perm=(0, 2, 3, 1))
        labels = tf.cast(labels, tf.int32)
        return inputs, labels

    def _load(self, datadir):
        return obs.cifar10(join(datadir, "cifar10"))

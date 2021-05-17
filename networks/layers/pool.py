import tensorflow as tf
import numpy as np


class LosslessPool(tf.keras.layers.Layer):
    """Reduce the feature map dimensions by putting spatial neighbors
    into depth channel.

    """
    def __init__(self, kernel_size):
        super(LosslessPool, self).__init__()
        self._ks = kernel_size

    def call(self, batch, training=None):
        C = tf.shape(batch)[-1]
        # Create a (ks, ks, C, ks * ks * C) kernel
        # (ks*ks*C, ks*ks*C)
        kernel = tf.eye(self._ks * self._ks * C)
        # (ks, ks, C, ks*ks*C)
        kernel = tf.reshape(kernel, (self._ks, self._ks, C, -1))

        return tf.nn.conv2d(
            input=batch,
            filters=kernel,
            strides=self._ks,
            padding='SAME',
        )


class InvLosslessPool(tf.keras.layers.Layer):
    """Increase the feature map dimensions by putting depth channel
    into spatial neighbors.

    """
    def __init__(self, kernel_size):
        super(InvLosslessPool, self).__init__()
        self._ks = kernel_size

    def call(self, batch, training=None):
        C = tf.shape(batch)[-1]
        # number of channels must be multiple of ks*ks
        out_c = C // (self._ks * self._ks)
        kernel = tf.eye(self._ks * self._ks * out_c)
        # (ks, ks, out_C, C)
        kernel = tf.reshape(kernel, (self._ks, self._ks, out_c, -1))

        B = tf.shape(batch)[0]
        H = tf.shape(batch)[1]
        W = tf.shape(batch)[2]
        return tf.nn.conv2d_transpose(
            input=batch,
            filters=kernel,
            output_shape=(B, H * self._ks, W * self._ks, out_c),
            strides=self._ks,
        )

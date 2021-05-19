"""AE using im decoder."""

import tensorflow as tf

from dl_playground.networks.layers.interface import BatchLayer
from dl_playground.networks.autoencoder.im_decoder import IMDecoder
from dl_playground.networks.layers.cnn import ConvNet, LayerConfig


class CNNIM(tf.keras.layers.Layer, BatchLayer):
    def __init__(
        self,
        encoder_configs,
        decoder_kwargs,
    ):
        super(CNNIM, self).__init__()
        self._enc = ConvNet(encoder_configs)
        # TODO: make it a list so I can have several resolutions
        self._dec = IMDecoder(**decoder_kwargs)

    def call(self, batch, training=None):
        code = self._enc(batch, training=training)

        H = tf.shape(batch)[1]
        W = tf.shape(batch)[2]

        decoded = self.decode(
            code=code, im_shape=(H, W), training=training
        )
        return {
            'code': code,
            'decoded': decoded,
        }

    def decode(self, code, im_shape, training=None):
        # Create the coordinates
        H, W = im_shape

        # (H, W, 2)
        coord = tf.cast(tf.stack(
            tf.meshgrid(tf.range(H), tf.range(W), indexing='ij'),
            axis=-1
        ), tf.float32)

        # make it between -0.5 and 0.5
        coord = coord / tf.cast([H, W], tf.float32) - 0.5
        B = tf.shape(code)[0]
        # (B, H, W, 2)
        coord = tf.tile(coord[tf.newaxis], (B, 1, 1, 1))

        # The whole code (including spatially) will be used as if
        # the resolution is reduced to 1x1
        new_batch = {
            'code': tf.reshape(code, (B, -1)),
            'coord': tf.reshape(coord, (B, -1, 2)),
        }

        # (B, H*W)
        decoded = self._dec(new_batch, training=training)
        decoded = tf.reshape(decoded, (B, H, W, 1))

        return decoded

    def loss_fn(self, batch, prediction, step):
        label = tf.cast(batch > 0.5, tf.float32)  # binarize
        diff = label - prediction['decoded']
        dist = diff * diff
        loss = tf.reduce_mean(dist, axis=(1, 2, 3))
        return {'loss': loss}

    def train_callback(self):
        pass

    def summary(self, writer, batch, step, training=None):
        pred = self.call(batch, training=training)
        recons = pred['decoded']
        with writer.as_default():
            tf.summary.image("input", batch, step=step)
            tf.summary.image("recons", recons, step=step)

    @classmethod
    def from_config(cls, config):
        encoder_configs = [
            LayerConfig(**layer_config)
            for layer_config in config['encoder_configs']
        ]
        layer = cls(
            encoder_configs=encoder_configs,
            decoder_kwargs=config['decoder_kwargs'],
        )
        return layer

"""Convolutional Pose Machines

https://arxiv.org/pdf/1602.00134.pdf

"""

import tensorflow as tf
import tensorflow_probability as tfp

from dl_playground.networks.layers.interface import BatchLayer
from dl_playground.networks.layers.cnn import LayerConfig, ConvNet


VGG19CONV44_PATH = (
    '/home/data/anna/models/hand_kps/vgg19/conv4_4/pretrained-2'
)

VGG19CONV44_CONFIGS = [
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=64,
        activation='relu',
        name='conv1_1',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=64,
        activation='relu',
        name='conv1_2',
    ),
    LayerConfig(
        type='MaxPool2D',
        kernel_size=2,
        strides=2,
        padding='VALID',
        name='pool1',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=128,
        activation='relu',
        name='conv2_1',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=128,
        activation='relu',
        name='conv2_2',
    ),
    LayerConfig(
        type='MaxPool2D',
        kernel_size=2,
        strides=2,
        padding='VALID',
        name='pool2',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=256,
        activation='relu',
        name='conv3_1',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=256,
        activation='relu',
        name='conv3_2',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=256,
        activation='relu',
        name='conv3_3',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=256,
        activation='relu',
        name='conv3_4',
    ),
    LayerConfig(
        type='MaxPool2D',
        kernel_size=2,
        strides=2,
        padding='VALID',
        name='pool3',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=512,
        activation='relu',
        name='conv4_1',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=512,
        activation='relu',
        name='conv4_2',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=512,
        activation='relu',
        name='conv4_3',
    ),
    LayerConfig(
        type='Conv2D',
        kernel_size=3,
        strides=1,
        padding='SAME',
        filters=512,
        activation='relu',
        name='conv4_4',
    ),

]


class VGG19Conv44(tf.keras.layers.Layer):
    """VGG-19 architecture up to conv4_4."""
    def __init__(self, load_path=VGG19CONV44_PATH):
        super(VGG19Conv44, self).__init__()
        self._cnn = ConvNet(layer_configs=VGG19CONV44_CONFIGS)

        if load_path is not None:
            self._cnn.warm_start(input_shape=(8, 8, 3))

            layers = {}
            for layer in self._cnn.local_layers:
                layers[layer.name] = layer

            ckpt = tf.train.Checkpoint(
                **layers
            )
            ckpt.restore(load_path).assert_consumed()

            self._ckpt = ckpt

    def call(self, batch, training=None):
        return self._cnn.call(batch, training=training)


class CPM(tf.keras.layers.Layer, BatchLayer):
    """
    Parameters
    ----------
    feat_extractor_configs : [LayerConfig]
    predictor_configs : [LayerConfig]
    n_keypoints : int

    """
    def __init__(
        self,
        feat_extractor_configs,
        predictor_configs,
        n_keypoints,
        n_stages=6,
        predictor_share_weights=False,
        loss_gaussian_scale=2.0,
    ):
        super(CPM, self).__init__()

        self._vgg = VGG19Conv44()
        self._feat_ext = ConvNet(feat_extractor_configs)

        self._predictors = []
        for stage_id in range(n_stages):
            if stage_id == 0:
                pred = ConvNet(predictor_configs)
                self._predictors.append(pred)
            else:
                if predictor_share_weights:
                    self._predictors.append(self._predictors[0])
                else:
                    pred = ConvNet(predictor_configs)
                    self._predictors.append(pred)

        self._n_keypoints = n_keypoints
        self._n_stages = n_stages
        self._predictor_share_weights = predictor_share_weights
        self._loss_gaussian_scale = loss_gaussian_scale

    def call(self, batch, training=None):
        x = batch['image']
        # (B, H/8, W/8, 512)
        x = self._vgg(x, training=training)
        # (B, H/8, W/8, F)
        f = self._feat_ext(x, training=training)

        predictions = []
        pred = self._predictors[0].call(f, training=training)
        predictions.append(pred)

        for stage_id in range(1, self._n_stages):
            pred_input = tf.concat([pred, f], axis=-1)
            pred = self._predictors[stage_id].call(
                pred_input, training=training
            )
            predictions.append(pred)

        return predictions

    def loss_fn(self, batch, pred, step):
        # batch['keypoints'] : (B, P, 3)
        # (B, P, 2)
        keypoints = batch['keypoints'][..., :2]
        # (B, P)
        conf = batch['keypoints'][..., 2]

        H = tf.shape(batch['image'])[1]
        W = tf.shape(batch['image'])[2]

        Hp = tf.shape(pred[0])[1]
        Wp = tf.shape(pred[0])[2]
        pred_scale = tf.cast(Hp, tf.float32) / tf.cast(H, tf.float32)

        # the keypoint locations needs to be scaled accordingly
        gaussian_pdf = tfp.distributions.MultivariateNormalDiag(
            loc=keypoints[:, tf.newaxis, tf.newaxis, :] * pred_scale,
            scale_diag=[
                self._loss_gaussian_scale, self._loss_gaussian_scale
            ],
        )

        # (Hp, Wp, 2)
        grids = tf.cast(tf.stack(tf.meshgrid(
            tf.range(Hp), tf.range(Wp), indexing='ij',
        ), axis=-1), tf.float32)

        # (B, Hp, Wp, P)
        probs = gaussian_pdf.prob(
            grids[tf.newaxis, :, :, tf.newaxis]
        )
        max_p = tf.reduce_max(probs)
        probs = probs / max_p  # make the max 1

        B = tf.shape(batch['image'])[0]
        pred_losses = []
        loss = tf.zeros(B, dtype=tf.float32)
        for p in pred:
            # each p is (B, Hp, Wp, P)
            diff = p - probs
            dist = diff * diff
            weighted_dist = dist * conf[:, tf.newaxis, tf.newaxis]
            pred_losses.append(weighted_dist)

            stage_loss = tf.reduce_sum(weighted_dist, axis=(1, 2, 3))
            loss += stage_loss

        return {
            'loss': loss,
            'pred_losses': pred_losses,
            'label_map': probs,
        }

    def train_callback(self):
        pass

    def predict(self, batch):
        out = self.call(batch, training=False)
        pred = out[-1]

        # resize the output
        orig_h = tf.shape(batch['image'])[1]
        orig_w = tf.shape(batch['image'])[2]
        output_map = tf.image.resize(
            pred, [orig_h, orig_w], method='bicubic'
        )

        # get the keypoints
        B = tf.shape(pred)[0]
        P = tf.shape(pred)[-1]
        # (B, H*W, P)
        flat_map = tf.reshape(output_map, (B, -1, P))
        # (B, P)
        keypoints = tf.cast(tf.math.argmax(flat_map, axis=1), tf.int32)
        keypoints_x = tf.math.floormod(keypoints, orig_w)
        keypoints_y = tf.math.floordiv(keypoints, orig_w)
        # (B, P, 2)
        keypoints = tf.stack([keypoints_y, keypoints_x], axis=-1)

        return {
            'heat_map': output_map,
            'keypoints': keypoints,
        }

    @classmethod
    def from_config(cls, config):
        feat_ext_configs = config['feature_extractor_configs']
        feat_ext_configs = [
            LayerConfig(**layer_config)
            for layer_config in feat_ext_configs
        ]
        predictor_configs = config['predictor_configs']
        predictor_configs = [
            LayerConfig(**layer_config)
            for layer_config in predictor_configs
        ]
        layer = cls(
            feat_extractor_configs=feat_ext_configs,
            predictor_configs=predictor_configs,
            **config['cpm'],
        )
        return layer

    def summary(self, writer, batch, step, training=None):
        # [(B, Hp, Wp, P)]
        pred = self.call(batch, training=training)
        loss = self.loss_fn(batch, pred, step)

        # only visualize the last stage for now
        with writer.as_default():
            tf.summary.image('input', batch['image'], step=step)
            for p in range(self._n_keypoints):
                tf.summary.image(
                    'joint{}'.format(p),
                    pred[-1][..., p][..., tf.newaxis],
                    step=step
                )
                tf.summary.image(
                    'loss/joint{}'.format(p),
                    loss['pred_losses'][-1][..., p][..., tf.newaxis],
                    step=step,
                )
                tf.summary.image(
                    'loss/label_map{}'.format(p),
                    loss['label_map'][..., p][..., tf.newaxis],
                    step=step,
                )

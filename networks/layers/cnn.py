from collections import namedtuple
import tensorflow as tf
import numpy as np

from artistcritic.networks.utils import (
    config_to_regularizer,
    ActivityMonitor,
    activation_function,
)
from artistcritic.networks.layers.pool import (
    LosslessPool,
    InvLosslessPool,
)


class ConvNet(tf.keras.layers.Layer):
    """A Convolution Neural Network trunk.

    Parameters
    ----------
    layer_configs : [LayerConfig]
    trainable : bool | None
        If not None, overwrites the `trainable` of all layers

    """
    def __init__(self, layer_configs, trainable=None):
        super(ConvNet, self).__init__()
        self._local_layers = []
        self._layer_groups = [] # this will be list of lists
        self._layer_channels = []
        for config in layer_configs:
            layers = config_to_layers(config, trainable=trainable)
            self._local_layers.extend(layers)
            self._layer_groups.append(layers)

            # Record the number of channels per layer
            for layer in layers:
                if type(layer) == tf.keras.layers.Conv2D:
                    self._layer_channels.append(layer.filters)
                elif type(layer) == tf.keras.layers.Conv2DTranspose:
                    self._layer_channels.append(layer.filters)
                elif len(self._layer_channels) == 0:
                    self._layer_channels.append(None)
                else:
                    self._layer_channels.append(
                        self._layer_channels[-1]
                    )

        self._last_acts = []
        self._act_monitor = ActivityMonitor()

    def warm_start(self, input_shape):
        self.call(np.zeros((
            1,
            input_shape[0],
            input_shape[1],
            input_shape[2]
        )), training=False)

    def call(
        self,
        inputs,
        training=None,
        masks=None,
        external_inputs=None,
    ):
        """Runs the network.

        Parameters
        ----------
        inputs : tf.Tensor | [tf.Tensor] | dict
        training : bool | None
        masks : [tf.Tensor] | None
        external_inputs : dict | None
            Maps layer name to additional input to concat to the
            existing input from the previous layer.

        Returns
        -------
        output : tf.Tensor

        Side effects
        ------------
        last_acts : `self._last_acts` is set to the latest activation

        """
        x = inputs
        acts = []
        for idx, layer_group in enumerate(self._layer_groups):
            group_name = layer_group[0].name
            if (
                external_inputs is not None and
                group_name in external_inputs
            ):
                ext_in = external_inputs[group_name]
                x = tf.concat([x, ext_in], axis=-1)

            for layer in layer_group:
                x = layer(x, training=training)

            if masks is not None and masks[idx] is not None:
                x = x * masks[idx]
            acts.append(x)

        self._last_acts = acts
        return x

    def update_activity_monitor(self):
        layer_names = [layer.name for layer in self._local_layers]
        self._act_monitor.update_counters(
            layer_names, self._last_acts
        )

    @property
    def local_layers(self):
        return self._local_layers

    @property
    def last_acts(self):
        return self._last_acts

    @classmethod
    def from_config(cls, configs):
        configs = [
            LayerConfig(**layer_config)
            for layer_config in configs
        ]
        return cls(configs)


LayerConfig = namedtuple('LayerConfig', [
    'type',  #  one of 'Conv2D', 'MaxPool2D', 'Conv2DTranspose',
             #  'UpSampling2D', 'LosslessPool', 'InvLosslessPool'
    'kernel_size',
    'strides',
    'padding',
    'filters',
    'activation',
    'activity_regularizer',  # (type, weight)
    'use_batchnorm',
    'use_bias',
    'trainable',
    'name',
], defaults=[
    None,  # type
    None,  # kernel_size
    None,  # strides
    None,  # padding
    None,  # filters
    None,  # activation
    None,  # activity_reg
    False,  # use_batchnorm
    True,  # use_bias
    True,  # trainable
    None,  # name
])


def config_to_layers(config, trainable=None):
    """Convert LayerConfig to keras layer(s).

    Parameters
    ----------
    config : LayerConfig
    trainable : bool | None
        If not None, overwrite `trainable` in config

    Returns
    -------
    layers : [tf.keras.layers.Layer]
        The constructed layers. When batch norm is used, two layers
        are returned (the second is a BatchNorm layer). Otherwise,
        a single element list is returned.

    """
    # config : LayerConfig
    act_regularizer = config_to_regularizer(
        config.activity_regularizer
    )

    if trainable is None:
        trainable = config.trainable

    if config.type == 'Conv2D':
        final_act_fn = activation_function(config.activation)
        act_fn = 'linear' if config.use_batchnorm else final_act_fn
        layers = [tf.keras.layers.Conv2D(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding=config.padding,
            activation=act_fn,
            use_bias=config.use_bias and (not config.use_batchnorm),
            activity_regularizer=act_regularizer,
            trainable=trainable,
            name=config.name,
        )]
        if config.use_batchnorm:
            if config.name is not None:
                bn_name = config.name + '_bn'
                act_name = config.name + '_act'
            else:
                bn_name = None
                act_name = None
            blayer = tf.keras.layers.BatchNormalization(
                trainable=trainable,
                name=bn_name,
            )
            layers.append(blayer)
            act_layer = tf.keras.layers.Activation(
                final_act_fn,
                name=act_name,
            )
            layers.append(act_layer)
        return layers
    if config.type == 'MaxPool2D':
        return [tf.keras.layers.MaxPool2D(
            pool_size=config.kernel_size,
            strides=config.strides,
            padding=config.padding,
            name=config.name,
        )]
    if config.type == 'LosslessPool':
        return [LosslessPool(
            kernel_size=config.kernel_size, name=config.name
        )]
    if config.type == 'InvLosslessPool':
        return [InvLosslessPool(
            kernel_size=config.kernel_size, name=config.name
        )]
    if config.type == 'Conv2DTranspose':
        final_act_fn = activation_function(config.activation)
        act_fn = 'linear' if config.use_batchnorm else final_act_fn
        layers = [tf.keras.layers.Conv2DTranspose(
            filters=config.filters,
            kernel_size=config.kernel_size,
            strides=config.strides,
            padding=config.padding,
            activation=act_fn,
            use_bias=config.use_bias and (not config.use_batchnorm),
            activity_regularizer=act_regularizer,
            trainable=trainable,
            name=config.name,
        )]
        if config.use_batchnorm:
            if config.name is not None:
                bn_name = config.name + '_bn'
                act_name = config.name + '_act'
            else:
                bn_name = None
                act_name = None
            blayer = tf.keras.layers.BatchNormalization(
                trainable=trainable,
                name=bn_name,
            )
            layers.append(blayer)
            act_layer = tf.keras.layers.Activation(
                final_act_fn,
                name=act_name,
            )
            layers.append(act_layer)
        return layers
    if config.type == 'UpSampling2D':
        return [tf.keras.layers.UpSampling2D(
            size=config.kernel_size,
            name=config.name,
        )]
    raise ValueError("Unrecognized layer {}".format(config.type))

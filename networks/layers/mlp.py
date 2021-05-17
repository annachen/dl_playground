import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from collections.abc import Iterable

from artistcritic.networks.utils import (
    config_to_regularizer,
    ActivityMonitor,
    activation_function,
)


class MLP(layers.Layer):
    """Multi-layer perceptron.

    Parameters
    ----------
    filters : [int]
        Number of filters for the layers of the MLP.
    act_fn : str | [str]
        Activation function to use for intermediate layers
    last_layer_act_fn : str
        The activation function of the last layer. All other layers
        use (leaky) ReLU.
    act_fn_kwargs : dict | None
        kwargs to `act_fn` and `last_layer_act_fn`
    use_bias : bool
        Whether the dense layers use biases.
    use_leaky : bool
        Whether to use leaky ReLU for intermediate layer activations.
    activity_regularizer : (str, float) | None
    normalize_weights : bool
        Whether to normalize the filter weights
    normalize_activation : bool
    clip_weights_norm : float | None
    n_input_channels : int | None
        Number of inputs. Only needed when `normalize_weights` is True
    trainable : bool
        Default to True

    """
    def __init__(
        self,
        filters,
        act_fn='relu',
        last_layer_act_fn='relu',
        act_fn_overrides=None,
        act_fn_kwargs=None,
        use_bias=True,
        use_leaky=False,
        activity_regularizer=None,
        normalize_weights=False,
        normalize_activation=False,
        clip_weights_norm=None,
        n_input_channels=None,
        trainable=True,
    ):
        super(MLP, self).__init__(trainable=trainable)
        self._filters = filters
        if type(act_fn) == str:
            self._act_fn = activation_function(act_fn, act_fn_kwargs)
        else:
            self._act_fn = [
                activation_function(af, act_fn_kwargs)
                for af in act_fn
            ]
        self._last_layer_act_fn_str = last_layer_act_fn
        self._last_layer_act_fn = activation_function(
            last_layer_act_fn, act_fn_kwargs
        )
        self._use_bias = use_bias
        self._use_leaky = use_leaky
        self._normalize_weights = normalize_weights
        self._normalize_activation = normalize_activation
        self._clip_weights_norm = clip_weights_norm
        self._n_input_channels = n_input_channels
        self._act_reg = config_to_regularizer(activity_regularizer)

        self._local_layers = []
        for n_filters in filters:
            self._local_layers.append(
                layers.Dense(
                    n_filters,
                    activation='linear',
                    activity_regularizer=self._act_reg,
                    use_bias=use_bias,
                    trainable=trainable,
                )
            )

        # normalize initial weights if needed to
        if (
            self._normalize_weights or
            (self._clip_weights_norm is not None)
        ):
            # initialize the weights
            self.call(tf.zeros((1, self._n_input_channels)))

            if self._normalize_weights:
                self.normalize_weights()
                assert self._clip_weights_norm is None
            else:
                self.clip_weights_norm()

        self._last_acts = []
        self._last_pre_acts = []

        # initialize counters
        self._act_monitor = ActivityMonitor()

    def warm_start(self):
        assert self._n_input_channels is not None
        self.call(tf.zeros((1, self._n_input_channels)))

    def call(self, inputs, masks=None, training=None):
        x = inputs
        acts = []
        pre_acts = []
        for idx, l in enumerate(self._local_layers):
            x = l(x)
            if masks is not None and masks[idx] is not None:
                x = x * tf.cast(masks[idx], tf.float32)

            pre_acts.append(x)

            if idx != len(self._local_layers) - 1:
                if self._use_leaky:
                    x = tf.nn.leaky_relu(x)
                else:
                    if isinstance(self._act_fn, Iterable):
                        x = self._act_fn[idx](x)
                    else:
                        x = self._act_fn(x)
            else:
                x = self._last_layer_act_fn(x)

            if self._normalize_activation:
                x, _ = tf.linalg.normalize(x, axis=-1)

            acts.append(x)

        self._last_acts = acts
        self._last_pre_acts = pre_acts
        return x

    @property
    def last_acts(self):
        return self._last_acts

    @property
    def last_pre_acts(self):
        return self._last_pre_acts

    @property
    def local_layers(self):
        return self._local_layers

    def update_activity_monitor(self):
        layer_names = [layer.name for layer in self._local_layers]
        self._act_monitor.update_counters(
            layer_names, self._last_acts
        )

    def normalize_weights(self):
        for v in self.weights:
            if 'kernel' in v.name:
                # normalize for each filter node
                norm = tf.norm(v, axis=0, keepdims=True)
                v.assign(v / norm)

    def clip_weights_norm(self):
        for v in self.weights:
            if 'kernel' in v.name:
                clipped = tf.clip_by_norm(
                    v, clip_norm=self._clip_weights_norm, axes=0
                )
                v.assign(clipped)

    def train_callback(self):
        if self._normalize_weights:
            self.normalize_weights()
        if self._clip_weights_norm is not None:
            self.clip_weights_norm()


class MLPWithBatchNorm(layers.Layer):
    """Multi-layer perceptron with batch-norm.

    Batch-normalization is applied at each hidden layer (not the
    last layer).

    Parameters
    ----------
    filters : [int]
        Number of filters for each of the layers.
    last_layer_act_fn : str
        The activation function of the last layer. All the other
        layers use (leaky) ReLU.
    use_bias : bool
        Whether to use bias for the last layer. All other layers
        have a dense layer without bias plus a batchnorm layer.
    use_leaky : bool
        Whether to use leaky ReLU for hidden layers.
    normalize_weights : bool
        Whether to normalize the filter weights
    n_input_channels : int | None
        Number of inputs. Only needed when `normalize_weights` is True
    trainable : bool

    """
    def __init__(
        self,
        filters,
        last_layer_act_fn='relu',
        use_bias=True,
        use_leaky=False,
        normalize_weights=False,
        n_input_channels=None,
        trainable=True,
    ):
        # TODO: test this again. I think the previous time I didn't
        # set training to False in validation
        super(MLPWithBatchNorm, self).__init__()
        self._filters = filters
        self._last_layer_act_fn = last_layer_act_fn
        self._use_bias = use_bias
        self._use_leaky = use_leaky

        # Not supported yet
        assert normalize_weights is False

        self._local_layers = []
        for n_filters in filters[:-1]:
            self._local_layers.append(layers.Dense(
                n_filters,
                activation='linear',
                use_bias=False,
                trainable=trainable,
            ))
            self._local_layers.append(
                layers.BatchNormalization(trainable=trainable)
            )
            if use_leaky:
                self._local_layers.append(layers.LeakyReLU(alpha=0.2))
            else:
                self._local_layers.append(
                    activations.Activation('relu')
                )

        self._local_layers.append(layers.Dense(
            filters[-1],
            activation=last_layer_act_fn,
            use_bias=use_bias,
            trainable=trainable,
        ))

    def call(self, inputs, training=None):
        x = inputs
        for l in self._local_layers:
            if isinstance(l, layers.BatchNormalization):
                x = l(x, training=training)
                # TODO: add update ops
            else:
                x = l(x, training=training)
        return x

    @property
    def local_layers(self):
        return self._local_layers

    def train_callback(self):
        pass


class MLPModel(Model):
    def __init__(self, filters, last_layer_act_fn='relu', **kwargs):
        super(MLPModel, self).__init__(**kwargs)
        self._mlp = MLP(filters=filters, last_layer_act_fn=last_layer_act_fn)

    def call(self, inputs):
        return self._mlp(inputs)

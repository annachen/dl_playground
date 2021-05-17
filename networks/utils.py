import tensorflow as tf
from functools import partial


def scaled_tanh(x, x_scale=1.0, y_scale=1.0):
    return tf.keras.activations.tanh(x * x_scale) * y_scale


ACTIVATION_FNS = {
    'linear': tf.keras.activations.linear,
    'relu': tf.keras.activations.relu,
    'leaky_relu': tf.nn.leaky_relu,  # default alpha = 0.2
    'tanh': tf.keras.activations.tanh,
    'scaled_tanh': scaled_tanh,
    'sigmoid': tf.nn.sigmoid,
    'swish': tf.keras.activations.swish,
}
ACT_EPS = 1e-6

INITIALIZERS = {
    'RandomNormal': tf.keras.initializers.RandomNormal,
    'RandomUniform': tf.keras.initializers.RandomUniform,
    'GlorotNormal': tf.keras.initializers.GlorotNormal,
    'GlorotUniform': tf.keras.initializers.GlorotUniform,
    'HeNormal': tf.keras.initializers.HeNormal,
    'HeUniform': tf.keras.initializers.HeUniform,
}


def activation_function(act_fn, act_fn_kwargs=None):
    act_fn_kwargs = {} if act_fn_kwargs is None else act_fn_kwargs
    return partial(
        ACTIVATION_FNS[act_fn],
        **act_fn_kwargs
    )


def random_init(init_fn, init_kwargs, fin, fout):
    """Returns randomly initialized weights.

    Parameters
    ----------
    init_fn : str
    init_kwargs : dict | None
    fin : int
    fout : int

    Returns
    -------
    init_w : tf.Tensor, shape (fin, fout)

    """
    init_kwargs = {} if init_kwargs is None else init_kwargs
    initer = INITIALIZERS[init_fn](**init_kwargs)
    return initer(shape=(fin, fout))


@tf.function
def calculate_padding(padding_type, size, kernel_size, stride):
    """
    Calculates the amount of padding given type and other params.

    Implement according to formulas in https://mmuratarat.github.io/20
    19-01-17/implementing-padding-schemes-of-tensorflow-in-python

    Parameters
    ----------
    padding_type : str
        'SAME' or 'VALID'
    size : int
        Width or height of the tensor considered
    kernel_size : int
        Width or height of the convolution kernel
    stride : int

    Returns
    -------
    padding : int

    """
    if padding_type == 'VALID':
        return 0
    assert padding_type == 'SAME'

    if size % stride == 0:
        padding = tf.maximum(kernel_size - size, 0)
    else:
        padding = tf.maximum(kernel_size - (size % stride), 0)
    return padding


@tf.function
def conv2d_transpose_output_size(input_size, kernel_size, strides, paddings):
    """Calculates the output shape of a conv2d transpose operation.

    Implemented according to https://www.tensorflow.org/api_docs/pytho
    n/tf/keras/layers/Conv2DTranspose

    Parameters
    ----------
    input_size : [int, int]
        [H, W] of the input tensor
    kernel_size : [int, int]
        [H, W] of the convolution kernel
    strides : [int, int]
        Strides on h and w dimensions
    paddings : [int, int]
        Paddings on h and w dimensions

    Returns
    -------
    output_size : [int, int]
        [H, W] of the output

    """
    raise RuntimeError(
        'Use tf.python.keras.utils.conv_utils.deconv_output_length '
        'instead'
    )

    new_size_h = (
        (input_size[0] - 1) * strides[0] + kernel_size[0] - 2 * paddings[0]
    )
    new_size_w = (
        (input_size[1] - 1) * strides[1] + kernel_size[1] - 2 * paddings[1]
    )

    return [new_size_h, new_size_w]


def config_to_regularizer(tup):
    if tup is None:
        return None

    # tup: (type, weight)
    if tup[0] == 'l1':
        return tf.keras.regularizers.L1(l1=tup[1])
    if tup[0] == 'l2':
        return tf.keras.regularizers.L2(l2=tup[1])
    raise NotImplementedError(tup)


class ActivityMonitor:
    def __init__(self):
        self._layer_lifetime_acts = None
        self._layer_lifetime_cnts = None
        self._layer_last_acts = None

    def update_counters(self, layer_names, layer_acts):
        if self._layer_lifetime_acts is None:
            self._layer_lifetime_acts = {}
            self._layer_lifetime_cnts = {}
            self._layer_last_acts = {}

            for name, act in zip(layer_names, layer_acts):
                batch_size = tf.shape(act)[0]
                is_active = tf.cast(act > ACT_EPS, tf.int32)
                is_active = tf.reshape(is_active, (batch_size, -1))
                # sum over the batch (and spatial) dimension
                is_active = tf.reduce_sum(is_active, axis=0)
                self._layer_lifetime_acts[name] = tf.Variable(
                    is_active, trainable=False
                )
                self._layer_lifetime_cnts[name] = tf.Variable(
                    tf.ones_like(is_active) * batch_size,
                    trainable=False
                )
                self._layer_last_acts[name] = tf.Variable(
                    is_active,
                    trainable=False
                )

        else:
            for name, act in zip(layer_names, layer_acts):
                batch_size = tf.shape(act)[0]
                is_active = tf.cast(act > ACT_EPS, tf.int32)
                is_active = tf.reshape(is_active, (batch_size, -1))
                is_active = tf.reduce_sum(is_active, axis=0)
                self._layer_lifetime_acts[name].assign_add(is_active)
                self._layer_lifetime_cnts[name].assign_add(
                    tf.ones_like(is_active) * batch_size
                )
                self._layer_last_acts[name].assign(is_active)

    def activity_ratio(self):
        assert self._layer_lifetime_acts is not None

        ret = {}
        single_batch_ret = {}
        for name in self._layer_lifetime_acts.keys():
            act_cnt = tf.cast(
                self._layer_lifetime_acts[name], tf.float32
            )
            total_cnt = tf.cast(
                self._layer_lifetime_cnts[name], tf.float32
            )
            ratio = act_cnt / total_cnt
            ret[name] = ratio

            single_batch_ret[name] = self._layer_last_acts[name]

        return ret, single_batch_ret

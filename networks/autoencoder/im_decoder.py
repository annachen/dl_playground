"""Implicit fields decoder.

https://arxiv.org/pdf/1812.02822.pdf

"""

import tensorflow as tf

from dl_playground.networks.utils import activation_function


class IMDecoder(tf.keras.layers.Layer):
    """Implicit fields decoder.

    Implements https://arxiv.org/pdf/1812.02822.pdf. Note that the
    output is single channel, as it predicts whether a coordinate
    location is in the shape or not.

    The output has the specified activation function applied already.

    Parameters
    ----------
    n_vars : int
        Number of channels of the input code
    filters : [int]
        Number of filters of the networks
    act_fn : str
    act_fn_kwargs : dict | None
    coord_dims : int
    concat_layers : int
        Number of layers to concatenate the code and the coordinates.
        Default to all but the last 2 layers (as in the paper).
    last_layer_act_fn : str
        Default to sigmoid
    use_map_not_tile : bool

    """
    def __init__(
        self,
        n_vars,
        filters,
        act_fn='leaky_relu',
        act_fn_kwargs=None,
        coord_dims=2,
        concat_layers=None,
        last_layer_act_fn='sigmoid',
        use_map_not_tile=False,
    ):
        super(IMDecoder, self).__init__()
        self._n_vars = n_vars
        self._filters = filters
        self._act_fn = activation_function(act_fn, act_fn_kwargs)
        self._coord_dims = coord_dims
        self._last_layer_act_fn = last_layer_act_fn
        self._use_map_not_tile = use_map_not_tile

        if concat_layers is None:
            concat_layers = len(filters) - 1
        self._concat_layers = concat_layers

        self._local_layers = []
        for filt in self._filters:
            self._local_layers.append(
                tf.keras.layers.Dense(
                    filt,
                    activation='linear',
                )
            )
        self._local_layers.append(
            tf.keras.layers.Dense(1, activation=last_layer_act_fn)
        )

    def call(self, batch, training=None):
        """Runs the network

        Parameters
        ----------
        batch : dict
          "code" : (B, F)
          "coord" : (B, N, coord_dims)

        Returns
        -------
        is_in_shape : tf.Tensor, (B, N)

        """
        code = batch['code']
        coord = batch['coord']

        B = tf.shape(coord)[0]
        N = tf.shape(coord)[1]
        F = code.shape[-1]  # static shape
        D = coord.shape[-1]

        if self._use_map_not_tile:
            # (N, B, coord_dims)
            coord = tf.transpose(coord, (1, 0, 2))

            def fn(cur_coord):
                # cur_coord: (B, coord_dims)
                # (B, F + coord_dims)
                cur_code = tf.concat([code, cur_coord], axis=-1)
                # (B, 1)
                return self._forward(cur_code, training=training)

            # (N, B, 1)
            x = tf.map_fn(
                fn=fn,
                elems=coord,
                fn_output_signature=tf.TensorSpec(
                    shape=[None, 1], dtype=tf.float32
                ),
            )
            # (B, N, 1)
            x = tf.transpose(x, (1, 0, 2))

        else:
            # this tiling is expensie as N is typically close to a
            # receptive field size of some neuron, likely in the order
            # of 50-hundreds
            tiled = tf.tile(code[:, tf.newaxis], [1, N, 1])
            code = tf.concat([tiled, coord], axis=-1)
            code = tf.reshape(code, (B * N, F + D))
            # (B * N, 1)
            x = self._forward(code, training=training)

        return tf.reshape(x, (B, N))

    def _forward(self, code, training=None):
        # code: (N, F)
        x = code
        for lidx, layer in enumerate(self._local_layers[:-1]):
            x = layer(x, training=training)
            x = self._act_fn(x)

            if lidx < self._concat_layers:
                x = tf.concat([x, code], axis=-1)

        # the last layer
        # (N, 1)
        x = self._local_layers[-1](x, training=training)
        return x

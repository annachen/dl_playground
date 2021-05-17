import tensorflow as tf
import numpy as np

from artistcritic.networks.layers.mlp import MLP, MLPWithBatchNorm
from artistcritic.networks.layers.weight_sharing import (
    MLPWithTransposedWeights,
    CNNWithTransposedWeights,
)
from artistcritic.networks.layers import cnn
from artistcritic.networks.layers.interface import (
    BatchLayer,
    MetricType,
)
from artistcritic.networks.autoencoder.losses import (
    reconstruction_loss,
)

EPS = 1e-5


class AELayer(tf.keras.layers.Layer, BatchLayer):
    """An autoencoder.

    Note: the encoder and the decoder uses linear activations by
    default.

    Parameters
    ----------
    n_vars : int
        Number of variables in the bottleneck layer
    n_input_channels : int
    encoder_filters : [int]
        The filters used for the encoder
    decoder_filters : [int] | None
        The filters used for the decoder. If None, the encoder and
        the decoder share weights.
    encoder_act_fn : str
    decoder_act_fn : str
    encoder_act_regularizers : [(str, float)] | None
        The types (l1 or l2) and the weight of the regularization
    encoder_act_sparse_k : int | None
        The number or nodes that are allowed to fire per sample in the
        code.
    encoder_act_lifetime_sparsity : float | None
        The ratio of the times a node is activated over the lifetime,
        which is approximated by a batch. Implementing WTA autoencoder
    sparsify_with_abs_value : bool
    use_bias : bool
        Whether to use biases in encoder and decoder
    use_leaky : bool
        Whether to use leaky ReLU in encoder and decoder
    use_batchnorm : bool
        Whether to use batchnorm layers in encoder and decoder
    normalize_weights : bool
        Whether to normalize the filter weights
    noise_stddev : float
    output_is_logit : bool
    trainable: bool
    summary_image_shape : [int, int, int] | None
        [H, W, C] of the shape of the image to log in summary
    name : str

    """
    def __init__(
        self,
        n_vars,
        n_input_channels,
        encoder_filters,
        decoder_filters=None,
        encoder_act_fn='linear',
        decoder_act_fn='linear',
        encoder_act_regularizers=None,
        encoder_act_sparse_k=None,
        encoder_act_lifetime_sparsity=None,
        sparsify_with_abs_value=False,
        use_bias=True,
        use_leaky=False,
        use_batchnorm=False,
        normalize_weights=False,
        noise_stddev=0.0,
        reconstruction_error_type='mse',
        output_is_logit=True,
        trainable=True,
        summary_image_shape=None,
        name='',
    ):
        super(AELayer, self).__init__(trainable=trainable)
        self._n_vars = n_vars
        self._n_input_channels = n_input_channels
        self._enc_filters = encoder_filters
        self._dec_filters = decoder_filters
        self._enc_act_fn = encoder_act_fn
        self._dec_act_fn = decoder_act_fn
        self._enc_act_regs = encoder_act_regularizers or []
        self._enc_act_sparse_k = encoder_act_sparse_k
        self._enc_act_lifetime_k = encoder_act_lifetime_sparsity
        self._use_bias = use_bias
        self._use_leaky = use_leaky
        self._use_batchnorm = use_batchnorm
        self._normalize_weights = normalize_weights
        self._recons_err_type = reconstruction_error_type
        self._output_is_logit = output_is_logit
        self._summary_image_shape = summary_image_shape
        self._name = name
        self._sp_abs_value = sparsify_with_abs_value
        self._noise_stddev = noise_stddev

        mlp_cls = MLPWithBatchNorm if self._use_batchnorm else MLP

        self._encoder = mlp_cls(
            filters=encoder_filters + [n_vars],
            last_layer_act_fn=self._enc_act_fn,
            use_bias=self._use_bias,
            use_leaky=self._use_leaky,
            normalize_weights=self._normalize_weights,
            n_input_channels=self._n_input_channels,
            trainable=trainable,
        )
        if decoder_filters is not None:
            self._decoder = mlp_cls(
                filters = decoder_filters + [n_input_channels],
                last_layer_act_fn=self._dec_act_fn,
                use_bias=self._use_bias,
                use_leaky=self._use_leaky,
                normalize_weights=self._normalize_weights,
                n_input_channels=self._n_vars,
                trainable=trainable,
            )
        else:
            assert self._use_leaky is False, (
                "leaky-ReLU not supported for weight sharing AE."
            )
            assert self._use_batchnorm is False, (
                "Batchnorm not supported for weight sharing AE."
            )

            # initialize the encoder weights (otherwise they're lazily
            # initialized and I can't get them here)
            self._encoder(np.zeros((1, n_input_channels), dtype=np.float32))

            weights = []
            biases = []
            for enc_layer in self._encoder.local_layers[::-1]:
                n_filters = tf.shape(enc_layer.weights[0])[0]
                weights.append(enc_layer.weights[0])
                if self._use_bias:
                    biases.append(tf.Variable(
                        np.zeros(n_filters), dtype=tf.float32
                    ))  # They don't share biases

            if not self._use_bias:
                biases = None

            self._decoder = MLPWithTransposedWeights(
                weights=weights,
                biases=biases,
                last_layer_act_fn=self._dec_act_fn,
            )

    def call(self, inputs, training=None):
        code = self.encode(inputs, training=training)
        decoded = self.decode(code, training=training)
        return {
            'code': code,
            'decoded': decoded,
        }

    def encode(self, inputs, training=None):
        # Add noise if applicable
        if training is True and self._noise_stddev > EPS:
            x = inputs + tf.random.normal(
                shape=tf.shape(inputs),
                stddev=self._noise_stddev,
            )
        else:
            x = inputs

        code = self._encoder(x, training=training)
        if self._enc_act_sparse_k is not None:
            # applies k-sparse constraint, which only takes the
            # largest k activation (we use the absolute of the act)
            # and set the others to 0
            if self._sp_abs_value:
                act = tf.stop_gradient(tf.math.abs(code))
            else:
                act = tf.identity(code)

            final_code = tf.zeros_like(code)
            batch_idx = tf.range(tf.shape(code)[0])
            for _ in range(self._enc_act_sparse_k):
                # (B,)
                max_idx = tf.cast(tf.argmax(act, axis=1), tf.int32)
                # (B, 2)
                idx = tf.stack([batch_idx, max_idx], axis=1)

                # (B,) each the max value for the sample
                # Note that we gather from `code` which can be signed
                values = tf.gather_nd(code, idx)

                # Get the cur max code to the final code
                cur_code = tf.scatter_nd(
                    indices=idx,
                    updates=values,
                    shape=tf.shape(final_code),
                )

                final_code += cur_code

                mask = tf.scatter_nd(
                    indices=idx,
                    updates=tf.ones_like(values),
                    shape=tf.shape(final_code)
                )
                act = tf.where(
                    condition=tf.cast(mask, tf.bool),
                    x=tf.zeros_like(act),
                    y=act,
                )
                """
                # set the values at the cur max to negative so they
                # don't get selected later
                act = tf.where(
                    condition=(tf.math.abs(cur_code) < EPS),
                    x=act,
                    y=-tf.ones_like(act),
                )
                """
                act = tf.stop_gradient(act)

            code = final_code

        if self._enc_act_lifetime_k is not None:
            # similar to above but doing it for the batch dimension
            batch_size = tf.shape(code)[0]
            k = tf.cast(
                tf.ceil(batch_size * self._enc_act_lifetime_k),
                tf.int32
            )  # Using `ceil` to prevent it from being 0

            abs_act = tf.math.abs(code)

            final_code = tf.zeros_like(code)
            var_idx = tf.range(self._n_vars)
            for _ in range(k):
                # (n_vars,)
                max_idx = tf.cast(
                    tf.argmax(abs_act, axis=0), tf.int32
                )
                # (n_vars, 2)
                idx = tf.stack([max_idx, var_idx], axis=1)

                # (n_vars,) each the max value for the node
                # Note that we gather from `code` which is signed
                values = tf.gather_nd(code, idx)

                # Get the cur max code to the final code
                cur_code = tf.scatter_nd(
                    indices=idx,
                    updates=values,
                    shape=tf.shape(final_code),
                )

                final_code += cur_code

                # set the values at the cur max to negative so they
                # don't get selected later
                act = tf.where(
                    condition=(tf.math.abs(cur_code) < EPS),
                    x=act,
                    y=-tf.ones_like(act),
                )

            code = final_code

        return code

    def decode(self, inputs, training=None):
        return self._decoder(inputs, training=training)

    def predict(self, inputs):
        code = self.encode(inputs, training=False)
        decoded = self.decode(code, training=False)
        if self._output_is_logit:
            decoded = tf.nn.sigmoid(decoded)
        return code, decoded

    def loss_fn(self, batch, prediction, step):
        """The loss function for the autoencoder.

        Parameters
        ----------
        batch : tf.Tensor
        prediction : dict
            The output of the call function
        step : int

        Returns
        -------
        loss : tf.Tensor, shape (B,)

        """
        losses = {}
        loss = reconstruction_loss(
            loss_type=self._recons_err_type,
            prediction=prediction['decoded'],
            labels=batch,
            is_logit=self._output_is_logit,
        )
        losses['recons'] = loss

        # Add regularization
        for reg in self._enc_act_regs:
            if reg[0] == 'l1':
                r = tf.math.abs(prediction['code'])
                r = tf.reduce_sum(r, axis=-1)
                losses['l1_act_reg'] = r
                loss += r * reg[1]
            elif reg[0] == 'l2':
                r = prediction['code'] * prediction['code']
                r = tf.reduce_sum(r, axis=-1)
                losses['l2_act_reg'] = r
                loss += r * reg[1]
            else:
                raise ValueError()

        losses['loss'] = loss
        return losses

    def train_callback(self):
        self._encoder.train_callback()
        if self._dec_filters is not None:
            self._decoder.train_callback()

    def metric_fn(self, batch, prediction):
        return MetricType.LOW, self.loss_fn(
            batch=batch,
            prediction=prediction,
            step=None,
        )

    def compete_fn(self, batch, prediction):
        return self._loss_fn(
            batch=batch,
            prediction=prediction,
            step=None,
        )['loss']

    def summary(self, writer, batch, step, training=None):
        if self._summary_image_shape is None:
            return

        pred = self.call(batch, training=training)
        recons = pred['decoded']
        if self._output_is_logit:
            recons = tf.nn.sigmoid(recons)

        input_imgs = tf.reshape(
            batch, [-1] + self._summary_image_shape
        )
        recon_imgs = tf.reshape(
            recons, [-1] + self._summary_image_shape
        )

        if self._summary_image_shape[-1] >= 5:
            # can't plot images with more than 4 channels
            # plot the first 3 channels
            input_imgs = input_imgs[..., :3]
            recon_imgs = recon_imgs[..., :3]

        with writer.as_default():
            tf.summary.image(
                "input{}".format(self._name), input_imgs, step=step
            )
            tf.summary.image(
                "recons{}".format(self._name), recon_imgs, step=step
            )

        # also log activity percentage
        code = pred['code']
        abs_code = tf.math.abs(code)
        # (N, 2)
        act_idxs = tf.where(abs_code > EPS)
        n_act = tf.cast(tf.shape(act_idxs)[0], tf.float32)
        n_elems = tf.shape(code)[0] * tf.shape(code)[1]
        ratio = n_act / tf.cast(n_elems, tf.float32)

        with writer.as_default():
            tf.summary.scalar(
                "diag/activity{}".format(self._name), ratio, step=step
            )


class ConvAELayer(tf.keras.layers.Layer, BatchLayer):
    """Convolutional Autoencoder layer.

    Parameters
    ----------
    input_shape : [int, int, int]
        [H, W, C] of the input. Only used when sharing weights (only
        really uses C, H and W can be modified out)
    n_vars : int
        Number of values in the bottleneck layer (length of code).
        Only used if `use_dense` is True
    encoder_configs : [cnn.LayerConfig]
    decoder_configs : [cnn.LayerConfig]
    encoder_cnn_output_shape : [H, W, C]
        The output shape of the encoder CNN. Only used if `use_dense`
        is True.
    use_dense : bool
        Whether to use a dense layer after the conv layers. Default
        to True.
        If False, the output of the cnn must have `n_vars` elements.
    share_weights : bool
        Whether the encoder and decoder should share weights
    encoder_act_fn : str
        The activation function of the last layer of the encoder.
        This is only used when `use_dense` is True.
    decoder_act_fn : str
        The activation function of the last layer of the decoder
    use_bias : bool
        Whether the encoder and decoder should use bias
    output_is_logit : bool
    reconstruction_error_type : str
    noise_stddev : float
        Whether to add noise to the input and make it a denoising AE.
        Default to 0.

    """
    def __init__(
        self,
        input_shape,
        n_vars,
        encoder_configs,
        decoder_configs,
        encoder_cnn_output_shape,
        use_dense=True,
        share_weights=False,
        encoder_act_fn='linear',
        decoder_act_fn='linear',
        use_bias=True,
        output_is_logit=True,
        reconstruction_error_type='mse',
        noise_stddev=0.0,
    ):
        super(ConvAELayer, self).__init__()
        self._input_shape = input_shape
        self._n_vars = n_vars
        self._enc_configs = encoder_configs
        self._dec_configs = decoder_configs
        self._enc_output_shape = encoder_cnn_output_shape
        self._share_weights = share_weights
        self._recons_err_type = reconstruction_error_type
        self._use_bias = use_bias
        self._use_dense = use_dense
        self._output_is_logit = output_is_logit
        self._noise_stddev = noise_stddev

        # NOTE: whether cnns use bias is decided by `encoder_configs`
        # and `decoder_configs
        self._enc_cnn = cnn.ConvNet(encoder_configs)
        if use_dense:
            self._enc_mlp = MLP(
                filters=[n_vars],
                last_layer_act_fn=encoder_act_fn,
                use_bias=self._use_bias,
            )

        n_flatten_values = np.product(self._enc_output_shape)

        if not self._share_weights:
            # NOTE: the last layer activation of the cnn is decided
            # by `decoder_configs`
            self._dec_cnn = cnn.ConvNet(decoder_configs)
            if use_dense:
                self._dec_mlp = MLP(
                    filters=[n_flatten_values],
                    last_layer_act_fn='relu',
                    use_bias=self._use_bias,
                )
        else:
            if use_dense:
                # initialize the encoder layers to get weights
                # variables
                self._enc_mlp(np.zeros(
                    [1, n_flatten_values], dtype=np.float32
                ))

                # create the MLP with shared weights
                if self._use_bias:
                    biases = [tf.Variable(
                        np.zeros(n_flatten_values), dtype=np.float32
                    )]
                else:
                    biases = None

                self._dec_mlp = MLPWithTransposedWeights(
                    weights=[self._enc_mlp.local_layers[0].weights[0]],
                    biases=biases,
                    last_layer_act_fn='linear',
                )

            # initialize the encoder layers to get weights variables
            self._enc_cnn(np.zeros(
                [1] + self._input_shape, dtype=np.float32
            ))
            # create the list of weights and biases for the decoder
            # cnn
            n_channels = (
                self._enc_cnn._layer_channels[::-1] +
                [self._input_shape[-1]]
            )[1:]
            weights = []
            biases = []
            for layer_idx, layer in enumerate(self._enc_cnn.local_layers[::-1]):
                if type(layer) == tf.keras.layers.Conv2D:
                    weights.append(layer.weights[0])
                    if self._use_bias:
                        b = tf.Variable(np.zeros(
                            n_channels[layer_idx], dtype=np.float32
                        ))
                    else:
                        b = None
                    biases.append(b)
                elif type(layer) == tf.keras.layers.MaxPool2D:
                    weights.append(None)
                    biases.append(None)

            if not self._use_bias:
                biases = None

            self._dec_cnn = CNNWithTransposedWeights(
                layer_configs=self._dec_configs,
                weights=weights,
                biases=biases,
            )

    def call(self, inputs, masks=None, training=None):
        code = self.encode(inputs, masks=masks, training=training)
        decoded = self.decode(code, masks=masks, training=training)
        return {
            'code': code,
            'decoded': decoded,
        }

    def encode(self, inputs, masks=None, training=None):
        """Encode.

        Parameters
        ----------
        inputs : tf.Tensor, shape (B, H, W, C)
        masks : [tf.Tensor] | None
        training : bool | None

        Returns
        -------
        code : tf.Tensor
            If `use_dense` is True, shape (B, n_vars)
            Otherwise, shape (B, H', W', n_vars)

        """
        if training is True and self._noise_stddev > EPS:
            x = inputs + tf.random.normal(
                shape=tf.shape(inputs),
                stddev=self._noise_stddev,
            )
        else:
            x = inputs

        if masks is None:
            x = self._enc_cnn(x, training=training)
        else:
            n_cnn_layers = len(self._enc_cnn._local_layers)
            x = self._enc_cnn(
                x, masks=masks[:n_cnn_layers], training=training
            )

        batch_size = tf.shape(inputs)[0]
        if self._use_dense:
            x = tf.reshape(x, [batch_size, -1])
            if masks is None:
                x = self._enc_mlp(x, training=training)
            else:
                x = self._enc_mlp(
                    x, masks=[masks[n_cnn_layers]], training=training
                )
        return x

    def decode(self, inputs, masks=None, training=None):
        """Decodes.

        Parameters
        ----------
        inputs : tf.Tensor
            If `use_dense` is True, shape (B, n_vars)
            Otherwise, shape (B, H', W', n_vars)
        masks : [tf.Tensor] | None
        training : bool | None

        Returns
        -------
        decoded : tf.Tensor, shape (B, H, W, C)

        """
        x = inputs
        if self._use_dense:
            if masks is None:
                x = self._dec_mlp(x, training=training)
            else:
                n_cnn_layers = len(
                    self._dec_cnn._local_layers
                )
                n_total_layers = len(masks)
                mask_idx = n_total_layers - n_cnn_layers - 1
                x = self._dec_mlp(
                    x, masks=[masks[mask_idx]], training=training
                )
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, [batch_size] + self._enc_output_shape)

        if masks is None:
            x = self._dec_cnn(x, training=training)
        else:
            x = self._dec_cnn(
                x, masks=masks[-n_cnn_layers:], training=training
            )
        return x

    def predict(self, inputs):
        code = self.encode(inputs, training=False)
        decoded = self.decode(code, training=False)
        if self._output_is_logit:
            decoded = tf.nn.sigmoid(decoded)
        return {
            'code': code,
            'decoded': decoded,
        }

    def loss_fn(self, batch, prediction, step):
        loss = reconstruction_loss(
            loss_type=self._recons_err_type,
            labels=batch,
            prediction=prediction['decoded'],
            is_logit=self._output_is_logit,
        )
        return {'loss': loss}

    def metric_fn(self, batch, prediction):
        value = self.loss_fn(batch, prediction, step=None)
        return MetricType.LOW, value

    def train_callback(self):
        pass

    def summary(self, writer, batch, step, training=None):
        if self._output_is_logit:
            recons = tf.nn.sigmoid(
                self.call(batch, training=training)['decoded']
            )
        else:
            recons = self.call(batch, training=training)['decoded']

        with writer.as_default():
            tf.summary.image("input", batch, step=step)
            tf.summary.image("recons", recons, step=step)

    @classmethod
    def from_config(cls, config):
        # Create encoder configs
        encoder_configs = config['encoder']
        encoder_configs = [
            cnn.LayerConfig(**layer_config)
            for layer_config in encoder_configs
        ]

        # Create decoder configs
        decoder_configs = config['decoder']
        decoder_configs = [
            cnn.LayerConfig(**layer_config)
            for layer_config in decoder_configs
        ]

        # Create conv AE layer
        conv_ae_config = config['conv_ae_layer']
        conv_layer = cls(
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            **conv_ae_config,
        )

        return conv_layer

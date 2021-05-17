import tensorflow as tf
import numpy as np

from artistcritic.networks.layers import mlp, cnn
from artistcritic.networks.autoencoder import losses


class VAELayer(tf.keras.layers.Layer):
    """Variational Autoencoder layer.

    Note: sigma here is the covariance matrix; it's equivalent to
    variance in univariate Gaussian, and not std.

    Parameters
    ----------
    n_input_channels : int
    n_vars : int
    encoder_filters : [int]
        Number of filters for the encoder. Does not include the last
        layer where the network predicts the mean and variance.
    decoder_filters : [int]
        Number of filters for the decoder. Does not include the last
        layer where the network reconstruct the inputs.
    reconstruction_type : str
        'ce' for cross entropy, or 'mse' for mean square error
    kl_type : str
        'close' for close-form calculation, 'mc' for monte-carlo
    beta : float
        The weight on the KL term
    kl_add_step : int
        The step to add in the KL term loss with weight beta
    output_is_logit : bool
    summary_image_shape : [int, int, int] | None
    name : str

    """
    def __init__(
        self,
        n_input_channels,
        n_vars,
        encoder_filters,
        decoder_filters,
        reconstruction_type='ce',
        kl_type='close',
        beta=1.0,
        kl_add_step=0,
        output_is_logit=True,
        summary_image_shape=None,
        name='',
    ):
        super(VAELayer, self).__init__()
        self._n_vars = n_vars
        self._enc_filters = encoder_filters
        self._dec_filters = decoder_filters

        enc_filters = encoder_filters + [n_vars * 2]
        self._encoder = mlp.MLP(
            filters=enc_filters,
            last_layer_act_fn='linear',
        )
        dec_filters = decoder_filters + [n_input_channels]
        self._decoder = mlp.MLP(
            filters=dec_filters,
            last_layer_act_fn='linear',
        )
        self._recons_type = reconstruction_type
        self._kl_type = kl_type
        self._beta = beta
        self._kl_add_step = kl_add_step
        self._output_is_logit = output_is_logit
        self._summary_image_shape = summary_image_shape
        self._name = name

    def encode(self, inputs, training=None):
        # inputs: (B, n_feats)
        x = self._encoder(inputs, training=training)
        # (B, n_vars)
        means = x[:, :self._n_vars]
        # (B, n_vars)
        log_sigmas = x[:, self._n_vars:]

        return means, log_sigmas

    def sample(self, means, log_sigmas):
        batch_size = tf.shape(means)[0]
        samples = tf.random.normal(
            shape=(batch_size, self._n_vars),
            mean=0.0,
            stddev=1.0,
        )
        samples = samples * tf.math.exp(0.5 * log_sigmas) + means
        return samples

    def decode(self, latent_vars, training=None):
        decoded = self._decoder(latent_vars, training=training)
        return decoded

    def call(self, inputs, training=None):
        # inputs: (B, n_feats)
        # (B, n_vars), (B, n_vars)
        means, log_sigmas = self.encode(inputs, training=training)
        # (B, n_vars)
        samples = self.sample(means, log_sigmas)
        # (B, n_feats)
        decoded = self.decode(samples, training=training)
        return {
            'mean': means,
            'log_sigma': log_sigmas,
            'samples': samples,
            'decoded': decoded,
        }

    def predict(self, inputs):
        pred = self(inputs, training=False)
        pred['sigma'] = tf.math.exp(pred['log_sigma'])
        pred['decoded'] = tf.nn.sigmoid(pred['decoded'])
        return pred

    def loss_fn(self, batch, prediction, step):
        # TODO: make beta increase over time
        if self._recons_type == 'ce':
            recons_loss = losses.reconstruction_cross_entropy(
                prediction=prediction['decoded'],
                labels=batch,
                is_logit=True,
            )
        elif self._recons_type == 'mse':
            recons_loss = losses.reconstruction_mean_square_error(
                prediction=prediction['decoded'],
                labels=batch,
                is_logit=True,
            )
        else:
            raise ValueError()

        if self._kl_type == 'close':
            kl_loss = losses.KL(
                mean=prediction['mean'],
                log_sigma=prediction['log_sigma'],
            )
        elif self._kl_type == 'mc':
            kl_loss = losses.KL_monte_carlo(
                prediction['sample'],
                mean=prediction['mean'],
                log_sigma=prediction['log_sigma'],
            )
        else:
            raise ValueError()

        if step >= self._kl_add_step:
            loss = recons_loss + self._beta * kl_loss
        else:
            loss = recons_loss

        return {
            'loss': loss,
            'recons_loss': recons_loss,
            'kl': kl_loss,
        }

    def train_callback(self):
        pass

    def compete_fn(self, batch, prediction, training=None):
        return self.loss_fn(
            batch=batch,
            prediction=prediction,
            step=self._kl_add_step,
        )['loss']

    def code_from_encode(self, encode_output):
        return encode_output[0]

    def code_from_call(self, call_output):
        return call_output['mean']

    def summary(self, writer, batch, step, training=None):
        if self._summary_image_shape is None:
            return

        recons = self.call(batch, training=training)['decoded']
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


class ConvVAELayer(tf.keras.layers.Layer):
    """Convolutional Variational Autoencoder layer.

    The inputs is first passed through a CNN, whose output shape is
    `encoder_cnn_output_shape`. The tensor is then flattened and passed
    into a Dense layer with `n_vars` units.

    Parameters
    ----------
    input_shape: [int, int, int]
        [H, W, C] of the input
    n_vars : int
        Number of latent variables.
        When `use_dense` is True, the top MLP has `n_vars` * 2 nodes.
        When `use_dense` is False, the last conv layer has
        `n_vars` * 2 channels.
    encoder_configs : [cnn.LayerConfig]
        The configs of the convolutional layers. Does not include the
        last layer where the network predicts the mean and variance.
    decoder_configs : [cnn.LayerConfig]
        The configs of the decoder layers. Does not include the last
        layer where the network reconstruct the inputs.
    encoder_cnn_output_shape : [int, int, int]
        [H, W, C] of the encoder output. Only used when `use_dense` is
        True.
    use_dense : bool
        Whether to add a dense layer after the conv layers. Default to
        True. Note that if this is set to False, the encoder cnn has
        to output `n_vars * 2` nodes in total.
    reconstruction_type : str
        'ce' for cross entropy, or 'mse' for mean square error
    kl_type : str
        'close' for close-form calculation, 'mc' for monte-carlo
    beta : float
        The weight on the KL term
    kl_add_step : int
        The step to add in KL loss

    """
    def __init__(
        self,
        input_shape,
        n_vars,
        encoder_configs,
        decoder_configs,
        encoder_cnn_output_shape,
        use_dense=True,
        reconstruction_type='ce',
        kl_type='close',
        beta=1.0,
        kl_add_step=0,
    ):
        super(ConvVAELayer, self).__init__()
        self._n_vars = n_vars
        self._enc_configs = encoder_configs
        self._dec_configs = decoder_configs
        self._enc_output_shape = encoder_cnn_output_shape
        self._use_dense = use_dense

        self._enc_cnn = cnn.ConvNet(encoder_configs)
        self._dec_cnn = cnn.ConvNet(decoder_configs)
        if self._use_dense:
            self._enc_mlp = mlp.MLP(
                filters=[n_vars * 2],
                last_layer_act_fn='linear'
            )
            self._dec_mlp = mlp.MLP(
                filters=[np.product(encoder_cnn_output_shape)]
            )

        self._recons_type = reconstruction_type
        self._kl_type = kl_type
        self._beta = beta
        self._kl_add_step = kl_add_step

    def encode(self, inputs, training=None):
        """Encodes.

        Parameters
        ----------
        inputs : tf.Tensor, shape (B, H, W, C)
        training : bool | None

        Returns
        -------
        means : tf.Tensor
            If `use_dense` is True, shape (B, n_vars).
            Otherwise, shape (B, H', W', n_vars)
        log_sigmas : tf.Tensor
            Same shape as `means`

        """
        x = self._enc_cnn(inputs, training=training)
        if self._use_dense:
            batch_size = tf.shape(x)[0]
            x = tf.reshape(x, (batch_size, -1))  # (B, H'*W'*C')
            x = self._enc_mlp(x, training=training)
        # (B, H', W', C) or (B, n_vars)
        means = x[..., :self._n_vars]
        # (B, H', W', C) or (B, n_vars)
        log_sigmas = x[..., self._n_vars:]
        return means, log_sigmas

    def decode(self, latent_vars, training=None):
        """Decodes.

        Parameters
        ----------
        latent_vars : tf.Tensor
            If `use_dense` is True, shape (B, n_vars)
            Otherwise, shape (B, H', W', n_vars)
        training : bool | None

        Returns
        -------
        decoded : tf.Tensor, shape (B, H, W, C)

        """
        x = latent_vars
        if self._use_dense:
            batch_size = tf.shape(x)[0]
            x = self._dec_mlp(x, training=training)
            x = tf.reshape(x, [batch_size] + self._enc_output_shape)
        decoded = self._dec_cnn(x, training=training)
        return decoded

    def sample(self, means, log_sigmas):
        samples = tf.random.normal(
            shape=tf.shape(means),
            mean=0.0,
            stddev=1.0,
        )
        samples = samples * tf.math.exp(0.5 * log_sigmas) + means
        return samples

    def call(self, inputs, training=None):
        """Runs the network

        Parameters
        ----------
        inputs : tf.Tensor, shape (B, H, W, C)

        Returns
        -------
        outputs : dict
            'mean' : tf.Tensor
                If `use_dense` is True, shape (B, n_vars)
                Otherwise, shape (B, H', W', n_vars)
            'log_sigma' : tf.Tensor, same shape as 'means'
            'sample' : tf.Tensor, same shape as 'means'
            'decoded' : tf.Tensor, shape (B, H, W, C)

        """
        means, log_sigmas = self.encode(inputs, training=training)
        samples = self.sample(means, log_sigmas)
        decoded = self.decode(samples, training=training)
        return {
            'mean': means,
            'log_sigma': log_sigmas,
            'sample': samples,
            'decoded': decoded,
        }

    def predict(self, inputs):
        pred = self(inputs, training=False)
        pred['sigma'] = tf.math.exp(pred['log_sigma'])
        pred['decoded'] = tf.nn.sigmoid(pred['decoded'])
        return pred

    def loss_fn(self, batch, prediction, step):
        # TODO: make beta increase over time
        recons_loss = losses.reconstruction_loss(
            loss_type=self._recons_type,
            prediction=prediction['decoded'],
            labels=batch,
            is_logit=True,
        )

        mean_reshaped = tf.reshape(
            prediction['mean'], (-1, self._n_vars)
        )
        log_sigma_reshaped = tf.reshape(
            prediction['log_sigma'], (-1, self._n_vars)
        )
        sample_reshaped = tf.reshape(
            prediction['sample'], (-1, self._n_vars)
        )

        if self._kl_type == 'close':
            kl_loss = losses.KL(
                mean=mean_reshaped,
                log_sigma=log_sigma_reshaped,
            )
        elif self._kl_type == 'mc':
            kl_loss = losses.KL_monte_carlo(
                sample_reshaped,
                mean=mean_reshaped,
                log_sigma=log_sigma_reshaped,
            )
        else:
            raise ValueError()

        Hp = tf.shape(prediction['mean'])[1]
        Wp = tf.shape(prediction['mean'])[2]
        kl_loss = tf.reshape(kl_loss, (-1, Hp, Wp))
        kl_loss = tf.reduce_mean(kl_loss, axis=(1, 2))

        if step >= self._kl_add_step:
            loss = recons_loss + self._beta * kl_loss
        else:
            loss = recons_loss
        return loss

    def train_callback(self):
        pass

    def summary(self, writer, batch, step, training=None):
        pred = self.call(batch, training=training)
        recons = tf.nn.sigmoid(pred['decoded'])
        with writer.as_default():
            tf.summary.image("input", batch, step=step)
            tf.summary.image("recons", recons, step=step)

    @classmethod
    def from_config(cls, config):
        encoder_configs = [
            cnn.LayerConfig(**layer_config)
            for layer_config in config['encoder']
        ]
        decoder_configs = [
            cnn.LayerConfig(**layer_config)
            for layer_config in config['decoder']
        ]
        layer = cls(
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            **config['conv_vae_layer'],
        )
        return layer

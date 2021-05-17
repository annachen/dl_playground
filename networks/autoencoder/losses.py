import tensorflow as tf
import numpy as np

EPS = 1e-5


def KL_monte_carlo(z, mean, sigma=None, log_sigma=None):
    """Computes the KL divergence at a point, given by z.

    Implemented based on https://www.tensorflow.org/tutorials/generative/cvae
    This is the part "log(p(z)) - log(q(z|x)) where z is sampled from
    q(z|x).

    Parameters
    ----------
    z : (B, N)
    mean : (B, N)
    sigma : (B, N) | None
    log_sigma : (B, N) | None

    Returns
    -------
    KL : (B,)

    """
    if log_sigma is None:
        log_sigma = tf.math.log(sigma)

    zeros = tf.zeros_like(z)
    log_p_z = log_multivar_gaussian(z, mean=zeros, log_sigma=zeros)
    log_q_z_x = log_multivar_gaussian(z, mean=mean, log_sigma=log_sigma)
    return log_q_z_x - log_p_z


def KL(mean, sigma=None, log_sigma=None):
    """KL divergence between a multivariate Gaussian and Multivariate
    N(0, I).

    Implemented based on
    https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/

    Parameters
    ----------
    mean : (B, N)
    sigma : (B, N) | None
        The diagonol of a covariance matrix of a factorized Gaussian
        distribution.
    log_sigma : (B, N) | None
        The log diagonol of a covariance matrix of a factorized
        Gaussian distribution.
        One of `sigma` and `log_sigma` has to be passed in.

    Returns
    -------
    KL : (B,)

    """
    if sigma is None:
        sigma = tf.math.exp(log_sigma)
    if log_sigma is None:
        log_sigma = tf.math.log(sigma)

    u = tf.reduce_sum(mean * mean, axis=1)  # (B,)
    tr = tf.reduce_sum(sigma, axis=1)  # (B,)
    k = tf.cast(tf.shape(mean)[1], tf.float32)  # scalar
    lg = tf.reduce_sum(log_sigma, axis=1)  # (B,)
    return 0.5 * (u + tr - k - lg)


def log_multivar_gaussian(x, mean, sigma=None, log_sigma=None):
    """Computes log pdf at x of a multi-variate Gaussian.

    Parameters
    ----------
    x : (B, N)
    mean : (B, N)
    sigma : (B, N) | None
    log_sigma: (B, N) | None

    Returns
    -------
    log_p : (B,)

    """
    if sigma is None:
        sigma = tf.math.exp(log_sigma)
    if log_sigma is None:
        log_sigma = tf.math.log(sigma)

    x = x - mean
    upper = -0.5 * tf.reduce_sum(x * x / (sigma + EPS), axis=-1)  # (B,)

    k = tf.cast(tf.shape(x)[1], tf.float32)
    log_pi = tf.math.log(np.pi * 2)
    log_prod_sig = tf.reduce_sum(log_sigma, axis=1)  # (B,)
    lower = -0.5 * (k * log_pi + log_prod_sig)

    return upper - lower


def multivar_gaussian(x, mean, sigma):
    """Computes pdf at x of a multi-variate Gaussian

    Parameters
    ----------
    x : (B, N)
    mean : (B, N)
    sigma : (B, N)
        Represents the diagonol of a covariance matrix of a factorized
        Gaussian distribution.

    Returns
    -------
    p_x : (B,)

    """
    x = x - mean
    upper = tf.reduce_sum(x * x / sigma, axis=-1)  # (B,)
    upper = tf.math.exp(-0.5 * upper)  # (B,)

    pi_vec = tf.ones_like(x) * np.pi * 2  # (B, N)
    lower = pi_vec * sigma
    lower = tf.reduce_prod(lower, axis=-1)  # (B,)
    lower = tf.math.sqrt(lower)

    return upper / lower


def reconstruction_cross_entropy(prediction, labels, is_logit=True):
    """Computes reconstruction error using cross entropy.

    Parameters
    ----------
    prediction : (B, ...)
    labels : (B, ...)
        Same dimensions as `prediction`
    is_logit : bool
        Whether the prediction is logit (pre-softmax / sigmoid)

    Returns
    -------
    recons_error : (B,)

    """
    assert is_logit, "Not Implemented"

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(labels, tf.float32),
        logits=prediction,
    )
    batch_size = tf.shape(prediction)[0]
    cross_ent = tf.reshape(cross_ent, (batch_size, -1))
    return tf.reduce_mean(cross_ent, -1)


def reconstruction_mean_square_error(prediction, labels, is_logit=True):
    """Computes reconstruction error using mean-square-error.

    Parameters
    ----------
    prediction : (B, ...)
    labels : (B, ...)
        Same dimensions as `prediction`
    is_logit : bool
        Whether the prediciton is logit.

    Returns
    -------
    recons_error : (B,)

    """
    if is_logit:
        prediction = tf.nn.sigmoid(prediction)

    error = prediction - tf.cast(labels, tf.float32)
    error = error * error

    batch_size = tf.shape(labels)[0]
    error = tf.reshape(error, (batch_size, -1))
    return tf.reduce_mean(error, axis=1)


def reconstruction_loss(loss_type, prediction, labels, is_logit):
    # `is_logit` : whether the input `recons` is logit
    if loss_type == 'mse':
        loss = reconstruction_mean_square_error(
            prediction=prediction,
            labels=labels,
            is_logit=is_logit,
        )
    elif loss_type == 'ce':
        loss = reconstruction_cross_entropy(
            prediction=prediction,
            labels=labels,
            is_logit=is_logit,
        )
    else:
        raise ValueError()

    return loss

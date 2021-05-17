import numpy as np
import tensorflow as tf
import os
import cv2

from artistcritic.path import PACKAGE_ROOT


CIRCLE_MASKS = np.load(
    os.path.join(PACKAGE_ROOT, 'utils', 'circle_masks.npy'),
    allow_pickle=True
).item()


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def squeezed_sigmoid_tf(x, center, factor):
    # original sigmoid has x=0 being the center of the curve where y
    # passes 0.5. This function shifts the center to `center` and
    # squeeze the function by `factor`
    return tf.nn.sigmoid((x - center) * factor)


def entropy(probs):
    """Calculates entropy for Bernoulli probabilities.

    Parameters
    ----------
    probs : np.array
        Each element `i` is a Bernoulli probabilitiy of P(x_i = 1)

    Returns
    -------
    entropy : np.array, same shape as input
        Each element is the entropy of probability distribution `i`

    """
    # have some checks for now. can remove when the code is more stable
    if np.min(probs) < 0.:
        raise ValueError("Probability cannot be < 0")
    if np.max(probs) > 1.:
        raise ValueError("Probability cannot be > 1")
    # assuming binary distribution, input probability is Pr(x = 1)
    eps = 1e-5  # to avoid np.log(0)
    probs = probs + eps
    return -probs * np.log(probs) - (1. - probs) * np.log(1. - probs)


def softmax(logits):
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def gaussian_kl_tf(m1, m2, sigma1, sigma2):
    """KL divergence between two multi-variate gaussians.

    https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

    m1, m2, sigma1, sigma2 need to have the same shape or broadcast-
    able. The last dimension is the multi-variate dimension.

    """
    t1 = tf.math.log(
        tf.reduce_prod(sigma2, axis=-1) /
        tf.reduce_prod(sigma1, axis=-1)
    )
    d = tf.cast(tf.shape(m1)[-1], tf.float32)
    t2 = tf.reduce_sum(sigma1 / sigma2, axis=-1)
    t3 = tf.reduce_sum((m2 - m1) * (m2 - m1) / sigma2, axis=-1)
    return (t1 - d + t2 + t3) * 0.5


def to_numpy(tensors):
    """Converts tf.Tensors to numpy array.

    Parameters
    ----------
    tensors : tf.Tensor | dict | list

    Returns
    -------
    arrays : np.array | dict | list

    """
    if type(tensors) == list:
        return [to_numpy(t) for t in tensors]
    if type(tensors) == dict:
        return {k: to_numpy(v) for k, v in tensors.items()}
    return tensors.numpy()


def fourier_descriptor(contour, n_comps=None):
    """Returns the Fourier descriptor of a closed contour.

    Parameters
    ----------
    contour : np.array, shape (N, 2)
    n_comps : int | None
        If None, it'll be set to N // 2

    Returns
    -------
    pos_comps : np.array, shape (n_comps + 1,)
        The components from 0 to n_comps
    neg_comps : np.array, shape (n_comps,)
        The components from -n_comps to -1

    """
    n = len(contour)

    if n_comps is None:
        n_comps = n // 2

    # we need to compute (0, n_comps + 1) and (-n_comps, 0)
    pos_grid = np.meshgrid(range(n), range(n_comps + 1), indexing='ij')
    # (2, N, n_comps + 1)
    theta = -2.0 * np.pi * pos_grid[0] * pos_grid[1] / n
    pos_e_r = np.cos(theta)
    pos_e_i = np.sin(theta)

    neg_grid = np.meshgrid(range(n), range(-n_comps, 0), indexing='ij')
    theta = -2.0 * np.pi * neg_grid[0] * neg_grid[1] / n
    neg_e_r = np.cos(theta)
    neg_e_i = np.sin(theta)

    z = contour[:, 0] + contour[:, 1] * 1j
    pos_e = pos_e_r + pos_e_i * 1j
    pos_comps = np.dot(z, pos_e)

    neg_e = neg_e_r + neg_e_i * 1j
    neg_comps = np.dot(z, neg_e)

    return pos_comps / n, neg_comps / n


def inverse_fourier_descriptor(pos_comps, neg_comps, n):
    """Inverse fourier descriptor

    Parameters
    ----------
    pos_comps : np.arary, shape (n_comps + 1,)
    neg_comps : np.array, shape (n_comps,)
    n : int
        Number of points to output

    Returns
    -------
    points : np.array, shape (n, 2)

    """
    n_comps = len(neg_comps)

    pos_grid = np.meshgrid(range(n), range(n_comps + 1), indexing='ij')
    # (2, N, n_comps + 1)
    theta = -2.0 * np.pi * pos_grid[0] * pos_grid[1] / n
    pos_e_r = np.cos(theta)
    pos_e_i = np.sin(theta)

    neg_grid = np.meshgrid(range(n), range(-n_comps, 0), indexing='ij')
    theta = -2.0 * np.pi * neg_grid[0] * neg_grid[1] / n
    neg_e_r = np.cos(theta)
    neg_e_i = np.sin(theta)

    pos_e = pos_e_r + pos_e_i * 1j
    pos_z = np.dot(pos_comps, pos_e.T)

    neg_e = neg_e_r + neg_e_i * 1j
    neg_z = np.dot(neg_comps, neg_e.T)

    z = pos_z + neg_z
    ys = np.real(z)
    xs = np.imag(z)
    pts = np.stack([ys, xs], axis=-1)
    return pts


def inverse_fourier_descriptor_tf(pos_comps, neg_comps, n):
    """Inverse fourier descriptor, tensorflow version

    Parameters
    ----------
    pos_comps : tf.tensor, shape (n_comps + 1,)
    neg_comps : tf.tensor, shape (n_comps,)
    n : int
        Number of points to output

    Returns
    -------
    points : tf.tensor, shape (n, 2)

    """
    n_comps = tf.shape(neg_comps)[0]
    float_n = tf.cast(n, tf.float32)

    # (2, N, n_comps + 1)
    pos_grid = tf.cast(tf.meshgrid(
        tf.range(n), tf.range(n_comps + 1), indexing='ij'
    ), tf.float32)
    # (N, n_comps + 1)
    theta = -2.0 * np.pi * pos_grid[0] * pos_grid[1] / float_n
    pos_e_r = tf.math.cos(theta)
    pos_e_i = tf.math.sin(theta)

    neg_grid = tf.cast(tf.meshgrid(
        tf.range(n), tf.range(-n_comps, 0), indexing='ij'
    ), tf.float32)
    theta = -2.0 * np.pi * neg_grid[0] * neg_grid[1] / float_n
    neg_e_r = tf.math.cos(theta)
    neg_e_i = tf.math.sin(theta)

    pos_e = tf.complex(pos_e_r, pos_e_i)
    pos_z = tf.matmul(
        tf.cast(pos_comps[tf.newaxis], tf.complex64),
        tf.transpose(pos_e)
    )[0]

    neg_e = tf.complex(neg_e_r, neg_e_i)
    neg_z = tf.matmul(
        tf.cast(neg_comps[tf.newaxis], tf.complex64),
        tf.transpose(neg_e)
    )[0]

    z = pos_z + neg_z
    ys = tf.math.real(z)
    xs = tf.math.imag(z)
    pts = tf.stack([ys, xs], axis=-1)
    return pts


def gaussian_pdf_tf(x, m=0.0, sigma=1.0):
    t = (x - m) / sigma
    return tf.exp(-t * t / 2.0) / (sigma * tf.sqrt(2.0 * np.pi))


def log_gaussian_pdf_tf(x, m=0.0, sigma=1.0):
    t = (x - m) / sigma
    return -t * t / 2.0 - tf.log(sigma) - 0.5 * tf.log(2.0 * np.pi)


def local_mean_tf(ims, patch_size):
    # ims: (B, H, W, C)
    # kernel should be (ps, ps, in, out) which is (ps, ps, C, C)
    # the channels don't mix so those 2 dimensions should just be
    # identity.
    # The ps x ps dimensions should be 1 / (ps**2) everywhere
    C = tf.shape(ims)[-1]
    # (C, C)
    identity = tf.eye(C)
    kernel = tf.tile(
        identity[tf.newaxis, tf.newaxis],
        [patch_size, patch_size, 1, 1]
    ) / (patch_size * patch_size)
    # want to do symmetric padding so need to do it manually
    padding = (patch_size - 1)
    left = padding // 2  # left pad less than right, if not equal
    right = padding - left
    padded = tf.pad(
        tensor=ims,
        paddings=[[0, 0], [left, right], [left, right], [0, 0]],
        mode='REFLECT',
    )
    mean = tf.nn.conv2d(
        input=padded,
        filters=kernel,
        strides=1,
        padding='VALID',
    )
    return mean


def high_pass_tf(ims, patch_size):
    # (ps, ps)
    mask = CIRCLE_MASKS[patch_size]
    n = tf.reduce_sum(mask)

    C = tf.shape(ims)[-1]
    identity = tf.eye(C)
    # (ps, ps, C, C)
    kernel = (
        tf.cast(mask[..., tf.newaxis, tf.newaxis], tf.float32) *
        identity[tf.newaxis, tf.newaxis]
    ) / tf.cast(n, tf.float32)

    # want to do symmetric padding so need to do it manually
    padding = (patch_size - 1)
    left = padding // 2  # left pad less than right, if not equal
    right = padding - left
    padded = tf.pad(
        tensor=ims,
        paddings=[[0, 0], [left, right], [left, right], [0, 0]],
        mode='REFLECT',
    )
    mean = tf.nn.conv2d(
        input=padded,
        filters=kernel,
        strides=1,
        padding='VALID',
    )

    diff = ims - mean

    return diff


def vertical_high_pass_tf(ims, filter_size):
    f = np.zeros((filter_size, filter_size))
    f[1, :] = 1.0 / filter_size
    C = tf.shape(ims)[-1]
    identity = np.eye(C)
    # (fs, fs, C, C)
    kernel = (
        f[..., tf.newaxis, tf.newaxis] *
        identity[tf.newaxis, tf.newaxis]
    )

    # want to do symmetric padding so need to do it manually
    padding = (filter_size - 1)
    left = padding // 2  # left pad less than right, if not equal
    right = padding - left
    padded = tf.pad(
        tensor=ims,
        paddings=[[0, 0], [left, right], [left, right], [0, 0]],
        mode='REFLECT',
    )
    mean = tf.nn.conv2d(
        input=padded,
        filters=kernel,
        strides=1,
        padding='VALID',
    )

    diff = ims - mean
    return diff


def horizontal_high_pass_tf(ims, filter_size):
    # extracts horizontal edges
    f = np.zeros((filter_size, filter_size))
    f[:, 1] = 1.0 / filter_size
    C = tf.shape(ims)[-1]
    identity = np.eye(C)
    # (fs, fs, C, C)
    kernel = (
        f[..., tf.newaxis, tf.newaxis] *
        identity[tf.newaxis, tf.newaxis]
    )

    # want to do symmetric padding so need to do it manually
    padding = (filter_size - 1)
    left = padding // 2  # left pad less than right, if not equal
    right = padding - left
    padded = tf.pad(
        tensor=ims,
        paddings=[[0, 0], [left, right], [left, right], [0, 0]],
        mode='REFLECT',
    )
    mean = tf.nn.conv2d(
        input=padded,
        filters=kernel,
        strides=1,
        padding='VALID',
    )

    diff = ims - mean
    return diff


def normal_from_depth_tf(ims, fx, fy):
    # ims has to have 4 dimensions
    # (B, H, W, 1, 2), last dim (dy, dx)
    out = tf.image.sobel_edges(ims)
    # (B, H, W, 1)
    dzdy_per_mm = out[..., 0] / (im / fy)
    dzdx_per_mm = out[..., 1] / (im / fx)
    # (B, H, W, 3)
    n = tf.concat(
        [dzdy_per_mm, dzdx_per_mm, tf.ones_like(dzdy_per_mm) * 1.0],
        axis=-1
    )
    n, _ = tf.linalg.normalize(n, axis=-1)

    # this returns the normal pointing away from the camera
    # (z positive)
    return n


def laplacian_pyramid(im, levels=3):
    im = im.copy()

    gps = [im]  # gaussian pyramid
    for l in range(levels):
        im = cv2.pyrDown(im)
        gps.append(im)

    lps = []
    for idx, l in enumerate(range(levels)):
        next_up = cv2.pyrUp(gps[idx + 1])
        diff = cv2.subtract(gps[idx], next_up)
        lps.append(diff)
    lps.append(gps[-1])

    return lps

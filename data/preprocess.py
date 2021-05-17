import tensorflow as tf

import numpy as np
import scipy.linalg


def extract_patches(image, patch_size, stride):
    """Split input sample into patches

    This function can be used as
        fn = partial(extract_patches, patch_size=3, stride=1)
        dataset.flat_map(fn)
    to convert an image dataset to a patch dataset.

    Parameters
    ----------
    image : tf.Tensor, shape (H, W, C)
    patch_size : int

    Returns
    -------
    tf.data.Dataset
        A new dataset with each patch as element.

    """
    # (1, H', W', ps*ps*C)
    patches = tf.image.extract_patches(
        images=image[tf.newaxis],
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, stride, stride, 1],
        padding='VALID',
        rates=[1, 1, 1, 1],
    )

    # Reshape to have number of patches as the first dimension
    C = tf.shape(image)[-1]
    patches = tf.reshape(patches, (-1, patch_size, patch_size, C))

    # Convert to dataset
    return tf.data.Dataset.from_tensor_slices(patches)


def random_crop_patches(image, crop_size, n_crops=1):
    crops = []
    for _ in range(n_crops):
        crop = tf.image.random_crop(image, size=crop_size)
        crops.append(crop)
    crops = tf.stack(crops)
    return tf.data.Dataset.from_tensor_slices(crops)


def zero_mean(image, cross_channels=False):
    """Make each image in the batch 0 mean.

    If `cross_channels` is False, do not do average across channels.

    Parameters
    ----------
    image : tf.Tensor, shape (..., H, W, C)
    cross_channels : bool

    Return
    ------
    zero_mean_image : tf.Tensor
        Same shape as the input.
    mean : tf.Tensor
        If `cross_channels` is False, shape (..., 1, 1, C)
        Otherwise, shape (..., 1, 1, 1)

    """
    axis = (-2, -3)
    if cross_channels:
        axis = (-1, -2, -3)
    mean = tf.reduce_mean(image, axis=axis, keepdims=True)
    return image - mean, mean


def fft(image):
    """Runs 2D FFT on the given image.

    Parameters
    ----------
    image : tf.Tensor, shape (..., H, W)

    Returns
    -------
    mags : tf.Tensor, shape (..., H, W)
        The frequency magnitude, between 0 and 1
    phases : tf.Tensor, shape (..., H, W)
        The phases, between -pi and pi

    """
    H = tf.shape(image)[-2]
    W = tf.shape(image)[-1]
    # NOTE: it seems like tf (and np) FFT implementation doesn't
    # normalize in the forward direction, and is divided by n (in 2D
    # case, m * n) in the inverse call.
    fft_res = tf.signal.fft2d(tf.cast(image, tf.complex64))
    # NOTE: divide by H*W to make the value range in 0 and 1
    mags = tf.math.abs(fft_res) / tf.cast(H * W, tf.float32)
    phases = tf.math.angle(fft_res)
    return mags, tf.cast(phases, tf.float32)


def ifft(mag, phase):
    """Runs inverse 2D FFT on given magnitude and phase.

    Parameters
    ----------
    mag : tf.Tensor, shape (..., H, W)
    phase : tf.Tensor, shape (..., H, W)

    Returns
    -------
    image : tf.Tensor, shape (..., H, W)

    """
    H = tf.shape(mag)[-2]
    W = tf.shape(mag)[-1]
    mag = mag * tf.cast(H * W, tf.float32)
    cos_phase = tf.math.cos(phase)
    sin_phase = tf.math.sin(phase)
    comp = tf.complex(mag * cos_phase, mag * sin_phase)
    ifft_res = tf.signal.ifft2d(comp)
    return tf.cast(tf.math.real(ifft_res), tf.float32)


def pca(images, epsilon=1e-6):
    B, H, W, C = images.shape
    # Make image into vectors
    flatten = np.reshape(images, (B, H * W * C))

    # Set the batch mean to 0
    mean = np.mean(flatten)
    flatten = flatten - mean

    # Get covariance matrix
    # (H * W * C, H * W * C)
    sigma = np.dot(flatten.T, flatten) / B
    # (H * W * C, H * W * C), (H * W * C,)
    # u: Unitary matrix having left singular vectors as columns
    u, s, _ = scipy.linalg.svd(sigma)
    s_inv = 1. / np.sqrt(s[np.newaxis] + epsilon)
    # columns are principal vectors
    principal_components = s_inv.dot(u.T)

    return principal_components


def zca_whiten(images, epsilon=1e-6):
    """Whitening the images (tensorflow implementation).

    Modified from `zca_whiten_np`

    Parameters
    ----------
    images : tf.Tensor, shape (B, H, W, C)

    Returns
    -------
    whitened : tf.Tensor, same shape as input

    """
    B = tf.shape(images)[0]
    # Make image into vectors
    flatten = tf.reshape(images, (B, -1))

    # Set the batch mean to 0
    mean = tf.reduce_mean(flatten)
    flatten = flatten - mean

    # Get covariance matrix
    # (H * W * C, H * W * C)
    sigma = tf.matmul(tf.transpose(flatten), flatten) / tf.cast(B, tf.float32)
    # (H * W * C, H * W * C), (H * W * C,)
    # u: Unitary matrix having left singular vectors as columns
    s, u, _ = tf.linalg.svd(sigma)
    s_inv = 1. / tf.sqrt(s[tf.newaxis] + epsilon)
    principal_components = tf.matmul(u * s_inv, tf.transpose(u))

    whitex = tf.matmul(flatten, principal_components)
    whitex = tf.reshape(whitex, tf.shape(images))

    return whitex


def zca_whiten_np(images, epsilon=1e-6):
    """Whitening the images using numpy/scipy.

    Stolen from https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py

    A good answer on ZCA vs. PCA: https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening

    Parameters
    ----------
    images : np.array, shape (B, H, W, C)

    Returns
    -------
    whitened : np.array, same shape as input

    """
    B, H, W, C = images.shape
    # Make image into vectors
    flatten = np.reshape(images, (B, H * W * C))

    # Set the batch mean to 0
    mean = np.mean(flatten)
    flatten = flatten - mean

    # Get covariance matrix
    # (H * W * C, H * W * C)
    sigma = np.dot(flatten.T, flatten) / B
    # (H * W * C, H * W * C), (H * W * C,)
    # u: Unitary matrix having left singular vectors as columns
    u, s, _ = scipy.linalg.svd(sigma)
    s_inv = 1. / np.sqrt(s[np.newaxis] + epsilon)
    principal_components = (u * s_inv).dot(u.T)

    whitex = np.dot(flatten, principal_components)
    whitex = np.reshape(whitex, (B, H, W, C))

    whitex = whitex.astype(np.float32)  # just in case

    return whitex

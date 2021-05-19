import numpy as np
import tensorflow as tf
import scipy.ndimage
import yaml
import os

from dl_playground.networks.layers.cnn import ConvNet
from dl_playground.utils.superpixel import rgb_to_normalized_lab
from dl_playground.utils.common import local_mean_tf


class Palette:
    """Convieniet functions for painting textures with spatial GAN.

    Note that the current GAN has the following modification: in
    training time, instead of using `z`s sampled from a normal distri-
    bution, we first sample the "mean" of a texture from a normal with
    stddev=1; then, this "mean" is repeated spatially and another
    noise is sampled with stddev=0.1. So, for each generated texture,
    the spatial stddev is 0.1. Different textures are distributed as
    a normal with stddev=1.

    So, similarly, at test time, we'd first sample the mean with
    stddev=1, then sample the noise for each texture with stddev=0.1.

    """
    def __init__(
        self,
        ckpt_path,
        palette_path,
        palette_is_lab=None,
        palette_bins=20,
        blur_type='gaussian',
        blur_sigma=1.0,
    ):
        # loads the model
        model_path = os.path.dirname(ckpt_path)
        config_path = os.path.join(model_path, 'exp_config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._gen = ConvNet.from_config(config['generator_configs'])
        self._noise_stddev = config['noise_stddev']

        ckpt = tf.train.Checkpoint(gen=self._gen)
        status = ckpt.restore(ckpt_path)
        status.expect_partial()

        # loads the color dict
        data = np.load(palette_path, allow_pickle=True)
        # maps from bin idxs to texture means
        self._texture_dict = data.item()

        # pop the settings
        self._use_lab = self._texture_dict.pop('use_lab')
        self._palette_bins = self._texture_dict.pop('n_bins')

        # (C, 3)
        self._avail_colors = np.array(list(self._texture_dict.keys()))
        if self._use_lab is None:
            self._use_lab = palette_is_lab
        if self._palette_bins is None:
            self._palette_bins = palette_bins

        self._blur_type = blur_type
        self._blur_sigma = blur_sigma

        self._last_zs = None

    def paint_random_walk(self, im, n_frames):
        # make sure we only use 3 channels
        im = im[..., :3]

        # convert to range (0, 1)
        if self._use_lab:
            im = rgb_to_normalized_lab(im[np.newaxis])[0]

        else:
            im = im / 255.0

        # im: (H, W, 3)
        H, W = im.shape[:2]

        step = 1.0 / self._palette_bins
        im_bin = np.floor((im - 0.001) / step).astype(np.int32)

        # find available colors
        # (H*W, 3)
        needed = im_bin.reshape((-1, 3))
        # (M, 3), (H * W)
        needed, idxs = np.unique(needed, axis=0, return_inverse=True)

        # (M, 3)
        avail = self._available_colors(needed)

        means = []
        for t in range(n_frames):
            if t == 0:
                # (M, n_vars)
                textures = self._random_texture_mean(avail)
                prev_txs = textures
            else:
                n_vars = prev_txs.shape[-1]
                dtx = np.random.normal(size=n_vars, scale=0.1)
                textures = prev_txs + dtx
                prev_txs = textures

            # (H, W, n_vars)
            texture_mean_map = textures[idxs].reshape((H, W, -1))

            # blur it a little so the boundaries look better
            tx_mean_map = self._blur_boundary(
                texture_mean_map, idxs.reshape((H, W))
            )

            means.append(tx_mean_map)

        means = np.array(means, dtype=np.float32)
        zs = np.random.normal(loc=means, scale=self._noise_stddev)

        self._last_zs = zs

        # runs the GAN
        painted = self._gen(zs.astype(np.float32))
        painted = painted / 2.0 + 0.5  # to (0, 1)
        painted = np.round(painted * 255.0).astype(np.uint8)

        return painted



    def paint_to_seq(self, im, n_frames):
        """Paint one image into sequence with morphing texture.

        """
        # make sure we only use 3 channels
        im = im[..., :3]

        # convert to range (0, 1)
        if self._use_lab:
            im = rgb_to_normalized_lab(im[np.newaxis])[0]

        else:
            im = im / 255.0

        # im: (H, W, 3)
        H, W = im.shape[:2]

        step = 1.0 / self._palette_bins
        im_bin = np.floor((im - 0.001) / step).astype(np.int32)

        # find available colors
        # (H*W, 3)
        needed = im_bin.reshape((-1, 3))
        # (M, 3), (H * W)
        needed, idxs = np.unique(needed, axis=0, return_inverse=True)

        # (M, 3)
        avail = self._available_colors(needed)

        means = []
        for t in range(n_frames):
            if t == 0:
                # (M, n_vars)
                textures = self._random_texture_mean(avail)
                prev_txs = textures
            else:
                textures = self._closest_texture_mean(
                    avail_colors=avail,
                    prev_tx_means = prev_txs,
                )
                prev_txs = textures

            # (H, W, n_vars)
            texture_mean_map = textures[idxs].reshape((H, W, -1))

            # blur it a little so the boundaries look better
            tx_mean_map = self._blur_boundary(
                texture_mean_map, idxs.reshape((H, W))
            )

            means.append(tx_mean_map)

        means = np.array(means, dtype=np.float32)
        zs = np.random.normal(loc=means, scale=self._noise_stddev)

        self._last_zs = zs

        # runs the GAN
        painted = self._gen(zs.astype(np.float32))
        painted = painted / 2.0 + 0.5  # to (0, 1)
        painted = np.round(painted * 255.0).astype(np.uint8)

        return painted

    def paint(self, ims):
        """Paint with texture.

        Parameters
        ----------
        ims : np.array, (B, H, W, 3), np.uint8

        Returns
        -------
        painted : np.array, (B, H*r, W*r, 3), np.uint8
            `r` is the upsampling ratio of the GAN.

        """
        # make sure we only use 3 channels
        ims = ims[..., :3]

        # convert to range (0, 1)
        if self._use_lab:
            ims = rgb_to_normalized_lab(ims)

        else:
            ims = ims / 255.0

        # (B, H, W, n_vars)
        means = np.array([
            self.get_mean_map(im) for im in ims
        ], dtype=np.float32)
        zs = np.random.normal(loc=means, scale=self._noise_stddev)

        self._last_zs = zs

        # runs the GAN
        painted = self._gen(zs.astype(np.float32))
        painted = painted / 2.0 + 0.5  # to (0, 1)
        painted = np.round(painted * 255.0).astype(np.uint8)

        return painted

    def get_mean_map(self, im):
        """Get the mean latent of an image.

        Parameters
        ----------
        im : np.array, (H, W, 3), np.uint8

        Returns
        -------
        tx_mean_map : np.array, (H, W, n_vars), np.float32

        """
        # im: (H, W, 3)
        H, W = im.shape[:2]

        step = 1.0 / self._palette_bins
        im_bin = np.floor((im - 0.001) / step).astype(np.int32)

        # find available colors
        # (H*W, 3)
        needed = im_bin.reshape((-1, 3))
        # (M, 3), (H * W)
        needed, idxs = np.unique(needed, axis=0, return_inverse=True)

        # (M, 3)
        avail = self._available_colors(needed)

        # (M, n_vars)
        textures = self._random_texture_mean(avail)

        # (H, W, n_vars)
        texture_mean_map = textures[idxs].reshape((H, W, -1))

        # blur it a little so the boundaries look better
        tx_mean_map = self._blur_boundary(
            texture_mean_map, idxs.reshape((H, W))
        )

        return tx_mean_map

    def _boundary_locations(self, idxs):
        # idxs: (H, W), a different index for each color
        # boundaries are where idxs - local_mean(idxs) != 0
        ps = 3
        local_mean = local_mean_tf(
            idxs[np.newaxis, :, :, np.newaxis].astype(np.float32),
            patch_size=ps
        )[0, ..., 0]
        diff = idxs - local_mean
        # (B, 2)
        bds = np.stack(np.where(np.abs(diff) > 1e-5), axis=-1)

        # remove image boundaries
        padding_left = (ps - 1) // 2
        padding_right = ps - 1 - padding_left
        shape = np.array(idxs.shape[:2])
        valid_idx = np.where(np.logical_and(
            np.all(bds >= padding_left, axis=1),
            np.all(bds < shape - padding_right, axis=1),
        ))[0]
        bds = bds[valid_idx]

        return bds

    def _blur_boundary(self, tx_mean_map, idxs):
        if self._blur_type == 'gaussian':
            tx_mean_map = scipy.ndimage.gaussian_filter(
                tx_mean_map,
                [self._blur_sigma, self._blur_sigma, 0.0]
            )

        elif self._blur_type == 'single_color':
            bds = self._boundary_locations(idxs)
            n_vars = tf.shape(tx_mean_map)[-1]
            tx_mean_map[bds[:, 0], bds[:, 1]] = np.zeros(n_vars)

        elif self._blur_type == 'dot_product':
            # (B, 2)
            bds = self._boundary_locations(idxs)
            # (H, W, 3*3*n_vars)
            patches = tf.image.extract_patches(
                images=tx_mean_map[np.newaxis],
                sizes=[1, 3, 3, 1],
                strides=[1, 1, 1, 1],
                padding='SAME',
                rates=[1, 1, 1, 1],
            )[0].numpy()
            # (B, 3*3*n_vars)
            bd_patches = patches[bds[:, 0], bds[:, 1]]
            B = len(bd_patches)
            # (B, n_vars)
            avg = _average_dir_mag(bd_patches.reshape(B, 9, -1))
            tx_mean_map[bds[:, 0], bds[:, 1]] = avg

        return tx_mean_map

    def _closest_texture_mean(
        self,
        avail_colors,
        prev_tx_means,
        explore_prob=0.3
    ):
        """Pick a texture that's closest to the previous texture."""
        textures = []
        for cid, color in enumerate(avail_colors):
            means = self._texture_dict[tuple(color)]
            explore = np.random.random() < explore_prob
            if explore:
                mean_idx = np.random.choice(range(len(means)))
                mean = means[mean_idx]
            else:
                means = np.array(means).reshape((-1, 4))
                diff = prev_tx_means[cid][np.newaxis] - means
                dist = np.sum(diff * diff, axis=-1)
                mean_idx = np.argmin(dist)
                mean = means[mean_idx]
            textures.append(mean.flatten())  # raw is (1, 1, 4)
        return np.array(textures)

    def _random_texture_mean(self, avail_colors):
        """Randomly pick a texture mean for each of the colors.

        Parameters
        ----------
        avail_colors : np.array, (M, 3), np.int32
            A list of colors to get textures for.

        Returns
        -------
        texture_means : np.array, (M, n_vars), np.float32

        """
        # TODO: I wonder what happens if I use a different texture
        # mean for each pixel (with the same color)
        textures = []
        for color in avail_colors:
            means = self._texture_dict[tuple(color)]
            mean_idx = np.random.choice(range(len(means)))
            mean = means[mean_idx]
            textures.append(mean.flatten())  # raw is (1, 1, 4)
        return np.array(textures)

    def _available_colors(self, colors):
        # colors: (N, 3)
        # return: (N, 3) availablecolor bins
        # (N, C, 3)
        diff = self._avail_colors[np.newaxis] - colors[:, np.newaxis]
        # (N, C)
        dist = np.sum(np.abs(diff), axis=-1)
        # (N,)
        to_use = np.argmin(dist, axis=1)
        # (N, 3)
        avail_colors = self._avail_colors[to_use]
        return avail_colors


def _average_dir_mag(patches):
    # average direction and magnitude of vectors over N.
    # patches: (B, N, n_vars)
    # (B, N)
    norm = np.linalg.norm(patches, axis=2)
    # (B, N, n_vars)
    normalized_dir = patches / norm[..., np.newaxis]
    # (B,)
    avg_mag = np.mean(norm, axis=1)
    # (B, n_vars)
    avg_dir = np.mean(normalized_dir, axis=1)

    avg = avg_dir * avg_mag[:, np.newaxis]
    return avg

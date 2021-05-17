import os
import yaml
import fire
import numpy as np
import tensorflow as tf
from collections import defaultdict
import skimage.color

from artistcritic.utils.superpixel import rgb_to_normalized_lab
from artistcritic.networks.layers.cnn import ConvNet


exp_path = '/home/data/anna/models/texture_gan/mp_medium_upsample_dual'
config_path = os.path.join(exp_path, 'exp_config.yaml')
ckpt_path = os.path.join(exp_path, 'ckpt-349')
palette_path = os.path.join(exp_path, 'palette-349.npy')
use_lab = False #True
n_bins = 20


def run():
    with open(config_path) as f:
        config = yaml.safe_load(f)
    gen = ConvNet.from_config(config['generator_configs'])

    ckpt = tf.train.Checkpoint(gen=gen)
    status = ckpt.restore(ckpt_path)
    status.expect_partial()

    color_dict = defaultdict(list)

    last_color_dict_len = -1
    while (len(color_dict) > last_color_dict_len):

        last_color_dict_len = len(color_dict)

        for _ in range(400):
            # create x
            m = tf.random.normal([10, 1, 1, 4])
            noise = tf.random.normal(
                shape=[10, 8, 8, 4], mean=m, stddev=0.1
            )

            # get images and convert back to (0, 1)
            # TODO: I haven't tried with training=True/False
            im = gen(noise)
            im = im / 2 + 0.5
            im = im.numpy()

            if use_lab:
                im = rgb_to_normalized_lab(im)

            # spatial mean color
            avg_im = np.mean(im, axis=(1, 2))
            step = 1.0 / n_bins
            bin_idxs = np.floor((avg_im - 0.001) / step).astype(
                np.int32
            )

            # over the batch
            for batch_id, bin_id in enumerate(bin_idxs):
                color_dict[tuple(bin_id.tolist())].append(
                    m[batch_id].numpy()
                )

    print('Found {} color bins'.format(last_color_dict_len))

    color_dict['use_lab'] = use_lab
    color_dict['n_bins'] = n_bins

    np.save(palette_path, color_dict)
    print('{} saved.'.format(palette_path))


if __name__ == '__main__':
    fire.Fire(run)

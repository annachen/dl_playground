import tensorflow as tf
import os
import imageio
from functools import partial
import numpy as np
import glob

ROOT = '/home/data/anna/datasets/my_painting_patches'
ROOT_109 = '/home/data/anna/datasets/my_painting_patches_109'


def get_dataset(patch_size, shuffle=False):
    if patch_size == 64:
        path = os.path.join(ROOT, '**')
    elif patch_size == 109:
        path = os.path.join(ROOT_109, '**')
    else:
        raise ValueError('Unpreprocessed patch size.')

    files = glob.glob(path, recursive=True)

    if shuffle:
        np.random.shuffle(files)

    def generator():
        for f in files:
            if not f.endswith('.png'):
                continue
            im = np.array(imageio.imread(f))
            yield {'image': im}

    return tf.data.Dataset.from_generator(
        generator,
        output_types={'image': tf.uint8},
        output_shapes={'image': tf.TensorShape(
            [patch_size, patch_size, 3]
        )}
    )

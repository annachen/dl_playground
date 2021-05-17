import tensorflow as tf
import os
import imageio
from functools import partial
import numpy as np

ROOT = '/home/data/anna/datasets/my_paintings'


def get_dataset(data_type, **kwargs):
    if data_type == 'raw':
        return raw_dataset(**kwargs)
    return patch_dataset(**kwargs)


def raw_dataset():
    files = os.listdir(ROOT)
    def generator():
        for f in files:
            im = np.array(
                imageio.imread(os.path.join(ROOT, f)),
            ) / 255.
            yield {'image': im}

    return tf.data.Dataset.from_generator(
        generator,
        output_types={'image': tf.float32},
        output_shapes={'image': tf.TensorShape([None, None, 3])}
    )


def patch_dataset(patch_size, scales):
    n_scales = len(scales)
    scale_to_use = tf.random.uniform(
        shape=(),
        minval=0,
        maxval=n_scales,
        dtype=tf.int32,
    )
    scale_to_use = scales[scale_to_use]

    crop_loc = tf.random.uniform(
        shape=(2,), maxval=1.0, dtype=tf.float32
    )

    proc_fn = partial(
        scale_and_crop,
        patch_size=patch_size,
        scale=scale_to_use,
        crop_loc=crop_loc,
    )

    dset = raw_dataset()
    dset = dset.map(proc_fn)
    return dset


def scale_and_crop(sample, scale, patch_size, crop_loc):
    im = sample['image']
    im = resize(im, scale)
    im_crop = crop(im, patch_size, crop_loc)
    return {'image': im_crop}


def crop(im, patch_size, loc):
    # loc: (y, x) location range (0, 1)

    shape = tf.shape(im)[:2]
    crop_start = tf.cast(shape - patch_size, tf.float32) * loc
    crop_start = tf.cast(tf.math.round(crop_start), tf.int32)

    crop = tf.image.crop_to_bounding_box(
        image=im,
        offset_height=crop_start[0],
        offset_width=crop_start[1],
        target_height=patch_size,
        target_width=patch_size,
    )
    return crop


def resize(im, scale):
    target_size = tf.cast(tf.shape(im)[:2], tf.float32) * scale
    target_size = tf.cast(tf.round(target_size), tf.int32)
    out = tf.image.resize(
        images=im,
        size=target_size,
    )
    return out

import os
import glob
import tensorflow as tf
from functools import partial
import imageio
import numpy as np
import json

from dl_playground.utils.tfrecord import (
    FeatureType,
    TFRecordConverter,
)


MANUAL_ROOT = '/home/data/anna/datasets/hand_keypoints_manual/tfrecords'
SYN_ROOT = '/home/data/anna/datasets/hand_keypoints_synthetic/tfrecords'
MV_ROOT = '/home/data/anna/datasets/hand_keypoints_multiview/tfrecords/train'
SKETCH_ROOT = '/home/data/anna/datasets/hand_gesture_sketches'

FEAT_DICT = {
    'image': FeatureType.IMAGE,
    'keypoints': FeatureType.FLOAT_ARRAY,
    'is_left': FeatureType.INT,
    'data_source': FeatureType.BYTES,
    'data_idx': FeatureType.INT,
}
FEAT_SHAPE = {
    'keypoints': [21, 3],
}


def get_sketch_dataset(
    target_size=None,
    annotated_only=False,
    to_float=True,
    to_right_hand=True,
):
    im_folder = os.path.join(SKETCH_ROOT, 'crops')
    ims = glob.glob(os.path.join(im_folder, '*.png'))
    ims.sort()

    anno_folder = os.path.join(SKETCH_ROOT, 'annotations')
    anno_files = os.listdir(anno_folder)
    anno_set = set([os.path.splitext(f)[0] for f in anno_files])

    meta_folder = os.path.join(SKETCH_ROOT, 'metadata')

    def gen():
        for im_idx, im_path in enumerate(ims):
            im_name = os.path.splitext(os.path.basename(im_path))[0]
            if annotated_only and im_name not in anno_set:
                continue

            anno_path = os.path.join(
                anno_folder, '{}.npy'.format(im_name)
            )
            meta_path = os.path.join(
                meta_folder, '{}.json'.format(im_name)
            )

            im = imageio.imread(im_path)

            if os.path.isfile(anno_path):
                kps = np.load(anno_path)
                # stored annotations is in (x, y, is_visible)
                coords = kps[:, :2][:, ::-1]
                kps = np.concatenate([coords, kps[:, 2:]], axis=-1)
            else:
                kps = np.zeros((21, 3))

            with open(meta_path) as f:
                meta = json.load(f)

            data = {
                'image': im,
                'keypoints': kps,
                'is_left': meta['is_left'],
                'data_source': 'sketch',
                'data_idx': im_idx,
                'style': meta['style'],
            }
            yield data

    dset = tf.data.Dataset.from_generator(
        generator=gen,
        output_signature={
            'image': tf.TensorSpec(
                dtype=tf.uint8, shape=(None, None, 3)
            ),
            'keypoints': tf.TensorSpec(
                dtype=tf.float32, shape=(21, 3)
            ),
            'is_left': tf.TensorSpec(dtype=tf.uint8, shape=()),
            'data_source': tf.TensorSpec(dtype=tf.string, shape=()),
            'data_idx': tf.TensorSpec(dtype=tf.int32, shape=()),
            'style': tf.TensorSpec(dtype=tf.uint8, shape=()),
        }
    )

    if target_size is not None:
        target_size = tf.constant([target_size, target_size])
        resize_fn = partial(_resize, target_size=target_size)
        dset = dset.map(resize_fn)

    if to_float:
        dset = dset.map(_to_float)

    if to_right_hand:
        dset = dset.map(_to_right_hand)

    return dset


def get_manual_dataset(split):
    assert split in ['train', 'test']
    path = os.path.join(MANUAL_ROOT, split)
    files = glob.glob(os.path.join(path, '*.tfr'))
    files.sort()
    dataset = tf.data.TFRecordDataset(files)
    cvtr = TFRecordConverter(FEAT_DICT, feature_shape=FEAT_SHAPE)
    dataset = dataset.map(cvtr.from_example)
    return dataset


def get_synthetic_dataset(split):
    assert split in ['synth1', 'synth2', 'synth3', 'synth4']
    path = os.path.join(SYN_ROOT, split)
    files = glob.glob(os.path.join(path, '*.tfr'))
    files.sort()
    dataset = tf.data.TFRecordDataset(files)
    cvtr = TFRecordConverter(FEAT_DICT, feature_shape=FEAT_SHAPE)
    dataset = dataset.map(cvtr.from_example)
    return dataset


def get_multiview_dataset():
    files = glob.glob(os.path.join(MV_ROOT, '*.tfr'))
    files.sort()
    dataset = tf.data.TFRecordDataset(files)
    cvtr = TFRecordConverter(FEAT_DICT, feature_shape=FEAT_SHAPE)
    dataset = dataset.map(cvtr.from_example)
    return dataset


def get_sketch_image_dataset():
    # without annotation
    files = glob.glob(os.path.join(SKETCH_ROOT, '*.png'))
    files.sort()

    def gen():
        for f in files:
            im = imageio.imread(f)
            yield im

    dset = tf.data.Dataset.from_generator(
        generator=gen,
        output_signature=tf.TensorSpec(
            dtype=tf.uint8,
            shape=(None, None, 3),
        )
    )
    return dset


def _resize(sample, target_size):
    H = tf.shape(sample['image'])[-3]
    W = tf.shape(sample['image'])[-2]
    h_ratio = tf.cast(target_size[0], tf.float32) / tf.cast(
        H, tf.float32
    )
    w_ratio = tf.cast(target_size[1], tf.float32) / tf.cast(
        W, tf.float32
    )

    # resize the image
    im = tf.image.resize(
        images=sample['image'],
        size=target_size,
    )
    sample['image'] = im

    # change the annotation
    keypoints = sample['keypoints']
    keypoints = keypoints * tf.stack(
        [h_ratio, w_ratio, 1.0], axis=0
    )
    sample['keypoints'] = keypoints

    return sample


def _to_float(sample):
    sample['image'] = tf.cast(sample['image'], tf.float32) / 255.0
    return sample


def _horizontal_flip_keypoints(keypoints, im_width):
    keypoints_w = keypoints[:, 1]
    new_w = tf.cast(im_width, tf.float32) - keypoints_w - 1.0
    new_keypoints = tf.stack([
        keypoints[:, 0],
        new_w,
        keypoints[:, 2],
    ], axis=-1)

    # set invalid points back to 0
    # (21, 3)
    to_zeros = tf.tile(new_keypoints[:, 2:] < 1e-4, [1, 3])
    new_keypoints = tf.where(
        to_zeros,
        tf.zeros((21, 3)),
        new_keypoints
    )
    return new_keypoints


def _to_right_hand(sample):
    # flip image
    im = tf.cond(
        sample['is_left'] > 0,
        true_fn=lambda: tf.image.flip_left_right(sample['image']),
        false_fn=lambda: sample['image'],
    )
    # flip points
    W = tf.shape(sample['image'])[-2]
    kps = tf.cond(
        sample['is_left'] > 0,
        true_fn=lambda: _horizontal_flip_keypoints(
            keypoints=sample['keypoints'],
            im_width=W,
        ),
        false_fn=lambda: sample['keypoints'],
    )

    sample['image'] = im
    sample['keypoints'] = kps
    return sample


def get_dataset(
    split,
    target_size=None,
    use_synth=False,
    use_sketch=False,
    to_float=True,
    to_right_hand=True,
):
    dset = get_manual_dataset(split)

    if split == 'train':
        mv_set = get_multiview_dataset()
        dset = dset.concatenate(mv_set)

        if use_synth:
            for synth_id in range(1, 5):
                synth_split = 'synth{}'.format(synth_id)
                synth_set = get_synthetic_dataset(synth_split)
                dset = dset.concatenate(synth_set)

        if use_sketch:
            raise NotImplementedError()

    elif split == 'test':
        if use_sketch:
            raise NotImplementedError()

    if target_size is not None:
        target_size = tf.constant([target_size, target_size])
        resize_fn = partial(_resize, target_size=target_size)
        dset = dset.map(resize_fn)

    if to_float:
        dset = dset.map(_to_float)

    if to_right_hand:
        dset = dset.map(_to_right_hand)

    return dset

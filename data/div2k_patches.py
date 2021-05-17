import tensorflow as tf
import glob
import os

from artistcritic.utils.tfrecord import (
    FeatureType,
    TFRecordConverter,
)


FEATURE_DEF = {
    'image': FeatureType.IMAGE,
    'scale': FeatureType.FLOAT,
}

ROOT = '/home/data/anna/datasets/div2k_hr_patches'


def get_dataset(path=ROOT):
    files = glob.glob(os.path.join(path, '*.tfr'))
    dset = tf.data.TFRecordDataset(files)

    cvtr = TFRecordConverter(FEATURE_DEF)

    dset = dset.map(cvtr.from_example)

    return dset

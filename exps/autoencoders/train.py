# Experiments for training any autoencoder with images

from collections import namedtuple
import fire
import os
import tensorflow_datasets as tfds
import tensorflow as tf
from functools import partial
import yaml
import shutil

from dl_playground.train.batch_train import TrainConfig, BatchTrainer
from dl_playground.networks.autoencoder.autoencoder import (
    ConvAELayer,
)
from dl_playground.networks.autoencoder.mean_shift_ae import (
    MeanShiftAELayer,
)
from dl_playground.networks.autoencoder.vae import (
    ConvVAELayer,
)
from dl_playground.networks.autoencoder.kae_v2 import KAE as KAEv2
from dl_playground.networks.autoencoder.patch_ae import (
    PatchConvAELayer,
)
from dl_playground.networks.autoencoder.stacked_coord_ae import (
    StackedCoordAELayer,
)
from dl_playground.networks.autoencoder.vqvae import VQVAE
from dl_playground.networks.autoencoder.feat_agg import (
    FeatureAggregation,
    ShapeAE,
)
from dl_playground.networks.autoencoder.texture_syn_v2 import (
    TextureSynthV2,
)
from dl_playground.networks.autoencoder.cnn_im import CNNIM
from dl_playground.networks.autoencoder.im_ae import IMAE
from dl_playground.networks.layers.model import LayerModel
from dl_playground.networks.layers.cnn import LayerConfig
from dl_playground.path import MODEL_ROOT
from dl_playground.data.datasets import DATASET_CHANNELS, load_dataset
from dl_playground.data.preprocess import random_crop_patches


OPTIMIZER_CLS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
}

LAYER_CLS = {
    'conv_ae': ConvAELayer,
    'conv_vae': ConvVAELayer,
    'mean_shift_ae': MeanShiftAELayer,
    'kaev2': KAEv2,
    'patch_ae': PatchConvAELayer,
    'stacked_coord_ae': StackedCoordAELayer,
    'vqvae': VQVAE,
    'feature_aggregation': FeatureAggregation,
    'shape_ae': ShapeAE,
    'texture_synth_v2': TextureSynthV2,
    'cnn_im': CNNIM,
    'im_ae': IMAE,
}


def preprocess(sample, to_grayscale=False):
    image = sample['image']
    if to_grayscale:
        image = tf.image.rgb_to_grayscale(image)

    image = tf.cast(image, tf.float32) / 255.0
    return image


def run(config_path, model_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = os.path.expanduser(model_path)

    # If `model_path` is absolute, os.path.join would return
    # `model_path` (!!)
    model_path = os.path.join(MODEL_ROOT, model_path)

    # Save the config
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    shutil.copyfile(
        src=config_path,
        dst=os.path.join(model_path, 'exp_config.yaml')
    )

    # Get the datasets
    dataset_kwargs = config.get('dataset_kwargs', {})
    train_dataset = load_dataset(
        config['dataset'], split='train', **dataset_kwargs
    )
    val_dataset = load_dataset(
        config['dataset'], split='test', **dataset_kwargs
    )

    #if DATASET_CHANNELS[config['dataset']] == 1:
    #    to_grayscale = False
    #else:
    #    to_grayscale = True
    #preproc_fn = partial(preprocess, to_grayscale=to_grayscale)
    #train_dataset = train_dataset.map(preproc_fn)
    #val_dataset = val_dataset.map(preproc_fn)

    # For div2k, random crop so the image is smaller
    if config['dataset'].startswith('div2k'):
        crop_fn = partial(
            random_crop_patches, crop_size=[320, 320, 1], n_crops=10
        )
        train_dataset = train_dataset.flat_map(crop_fn)
        val_dataset = val_dataset.flat_map(crop_fn)

    # Create training config
    train_config = TrainConfig(**config['train_config'])

    # Create optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        **config['learning_rate']
    )
    optimizer_name = config['optimizer']
    optimizer = OPTIMIZER_CLS[optimizer_name](
        learning_rate=lr_schedule
    )

    # Create model
    layer_cls = LAYER_CLS[config['ae_type']]
    layer = layer_cls.from_config(config)
    model = LayerModel(layer)

    # Load weights if needed
    if config['load_checkpoint'] is not None:
        model.load_weights(config['load_checkpoint'])

    trainer = BatchTrainer(
        model_path=model_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer,
        train_config=train_config,
    )

    trainer.train()


if __name__ == '__main__':
    fire.Fire(run)

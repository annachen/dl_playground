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
from dl_playground.networks.layers.model import LayerModel
from dl_playground.path import MODEL_ROOT
from dl_playground.data.naive_edge_graph import get_dataset
from dl_playground.networks.classification.gnn import (
    GNNGlobalClassifier
)


OPTIMIZER_CLS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
}


def preproc(sample):
    # Ragged (B, 1, N_i, F)
    edge_feats = sample['edge_feats']
    # Ragged (B, 1, N_i, 2)
    edge_coord = sample['edge_coord']
    node_feats = tf.concat([edge_feats, edge_coord], axis=-1)
    new_sample = {
        'node_feats': node_feats,
        'adj_mat': sample['adj_mat'],
        'label': sample['label'],
    }
    return new_sample


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
    train_dataset = get_dataset(config['dataset'], split='train')
    val_dataset = get_dataset(config['dataset'], split='test')
    train_dataset = train_dataset.map(preproc)
    val_dataset = val_dataset.map(preproc)

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
    layer = GNNGlobalClassifier.from_config(config)
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

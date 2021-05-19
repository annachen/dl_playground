import fire
import os
import tensorflow as tf
from functools import partial
import cv2
import numpy as np

from dl_playground.data.datasets import load_dataset
from dl_playground.data.preprocess import extract_patches
from dl_playground.networks.layers.cnn import LayerConfig, ConvNet
from dl_playground.exps.utils import load_and_save_config


n_steps = 500000
save_every_n_steps = 1000


def run(config_path, model_path):

    config = load_and_save_config(config_path, model_path)

    writer = tf.summary.create_file_writer(
        os.path.join(model_path, 'log')
    )

    # get the dataset
    # (H=W=64; l=m=2)
    dset = my_painting_patches_dset(config['patch_size'])

    # create the generator architecture
    gen = ConvNet.from_config(config['generator_configs'])

    # create the discriminator
    disc = ConvNet.from_config(config['discriminator_configs'])

    gen_opt = tf.keras.optimizers.Adam(1e-4)
    disc_opt = tf.keras.optimizers.Adam(1e-4)

    dset = dset.shuffle(1000, reshuffle_each_iteration=True)
    dset = dset.repeat(n_steps)
    dset = dset.batch(config['batch_size'])

    ckpt = tf.train.Checkpoint(
        gen_op=gen_opt,
        disc_op=disc_opt,
        gen=gen,
        disc=disc,
    )
    ckpt_prefix = os.path.join(model_path, 'ckpt')

    for step, batch in enumerate(dset):
        if config['dual_texture'] is True:
            pattern = _get_random_pattern(
                config['latent_size'], config['batch_size']
            )
        else:
            pattern = None
        train_out = train_step(
            batch=batch,
            gen=gen,
            disc=disc,
            gen_opt=gen_opt,
            disc_opt=disc_opt,
            config=config,
            pattern=pattern,
        )
        with writer.as_default():
            tf.summary.scalar(
                'loss/gen_loss',
                tf.reduce_mean(train_out['gen_loss']),
                step,
            )
            tf.summary.scalar(
                'loss/disc_loss',
                tf.reduce_mean(train_out['disc_loss']),
                step,
            )
            tf.summary.image('input', batch, step)
            tf.summary.image(
                'generated', train_out['generated'], step,
            )

        if step % save_every_n_steps == 0:
            ckpt.save(file_prefix=ckpt_prefix)


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE,
    )
    # (B, L, L)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    # (B,)
    total_loss = tf.reduce_mean(total_loss, axis=(1, 2))
    return total_loss


def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE,
    )
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return tf.reduce_mean(loss, axis=(1, 2))


@tf.function
def train_step(batch, gen, disc, gen_opt, disc_opt, config, pattern=None):
    L = config['latent_size']
    # first sample a mean
    m = tf.random.normal(
        [config['batch_size'], 1, 1, config['n_vars']]
    )
    if pattern is not None:
        # pattern is (B, L, L)
        m2 = tf.random.normal(
            [config['batch_size'], 1, 1, config['n_vars']]
        )
        m = tf.where(
            pattern[..., np.newaxis] < 0.5,
            m,
            m2
        )

    # then sample from that mean
    z = tf.random.normal(
        mean=m,
        stddev=config['noise_stddev'],
        shape=[config['batch_size'], L, L, config['n_vars']]
    )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(
            z, training=True,
        )

        # (B, L, L, 1)
        real_output = disc(batch, training=True)
        fake_output = disc(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, gen.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, disc.trainable_variables
    )

    gen_opt.apply_gradients(
        zip(gradients_of_generator, gen.trainable_variables)
    )
    disc_opt.apply_gradients(
        zip(gradients_of_discriminator, disc.trainable_variables)
    )
    return {
        'gen_loss': gen_loss,
        'disc_loss': disc_loss,
        'generated': generated_images,
    }


def random_crop(im, patch_size):
    H = tf.shape(im)[0]
    W = tf.shape(im)[1]
    crop_h_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=H - patch_size,
        dtype=tf.int32
    )
    crop_w_start = tf.random.uniform(
        shape=[],
        minval=0,
        maxval=W - patch_size,
        dtype=tf.int32
    )

    crop = tf.image.crop_to_bounding_box(
        image=im,
        offset_height=crop_h_start,
        offset_width=crop_w_start,
        target_height=patch_size,
        target_width=patch_size,
    )

    return crop


def normalize(float_im):
    # normalize to (-1, 1)
    return (float_im - 0.5) * 2


def single_im_dset(patch_size):
    dset = load_dataset(
        'dtd', 'train', image_only=True, to_float=True
    )
    #target_idx = 32  # manually looked at the image
    target_idx = 59
    for idx, im in enumerate(dset):
        if idx == target_idx:
            break

    dset = tf.data.Dataset.from_tensor_slices([im])
    crop_fn = partial(random_crop, patch_size=patch_size)
    dset = dset.map(crop_fn)
    dset = dset.map(normalize)
    return dset


def my_paintings_dset(patch_size, scales):
    dset = load_dataset(
        'my_paintings',
        'train',
        to_float=True,
        image_only=True,
        data_type='patch',
        patch_size=patch_size,
        scales=scales,
    )

    dset = dset.map(normalize)
    return dset


def my_painting_patches_dset(patch_size):
    dset = load_dataset(
        'my_painting_patches',
        'train',
        to_float=True,
        image_only=True,
        shuffle=True,
        patch_size=patch_size,
    )
    dset = dset.map(normalize)
    return dset


def _get_random_pattern(size, batch_size):
    outs = []
    for _ in range(batch_size):
        im = np.zeros((size, size))
        center = np.round(
            np.random.uniform(-size, size, size=2)
        ).astype(np.int32)
        radius = np.round(np.random.uniform(0, size)).astype(np.int32)
        o = cv2.circle(
            im, tuple(center), radius, color=1, thickness=-1
        )
        outs.append(o)
    return np.array(outs)


if __name__ == '__main__':
    fire.Fire(run)

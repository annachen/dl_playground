from artistcritic.data.augmentation import augment

from collections import namedtuple, defaultdict
import os
import tensorflow as tf
import yaml
from functools import partial


TrainConfig = namedtuple('TrainConfig', [
    'batch_size',
    'epochs',
    'eval_every_n_steps',
    'summary_every_n_steps',
    'save_every_n_steps',
    'max_ckpts_to_keep',
    'shuffle',
    'shuffle_buffer',
    'start_step',
    'run_in_python',
], defaults=[
    None,  # batch_size
    None,  # epochs
    None,  # eval_every_n_steps
    None,  # summary_every_n_steps
    None,  # save_every_n_steps
    None,  # max_ckpts_to_keep
    True,  # shuffle
    1000,  # shuffle_buffer
    0,     # start_step
    False,  # run_in_python
])

class BatchTrainer:
    """Run batched training, like normal deep learning.

    Parameters
    ----------
    model_path : str
    train_dataset : tf.Dataset
    val_dataset : tf.Dataset
    model : tf.keras.model.Model
    loss_fn : tf function
        A tensorflow function that takes in (batch_sample, batch_pred)
        and outputs a dictionary of losses with shape (batch_size,).
        The dictionary must contain key "loss" which is the total loss
    optimizer : tf.Optimizer
    train_config : TrainConfig
    data_summary_fn : python function
        A function that takes in (writer, batch, step)

    """
    def __init__(
        self,
        model_path,
        train_dataset,
        val_dataset,
        model,
        loss_fn,
        optimizer,
        train_config,
        aug_config=None,
        data_summary_fn=None,
    ):
        self._model_path = model_path
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._model = model
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._train_config = train_config
        self._ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            model=model,
        )
        # NOTE: this is not used! optimizer params are not saved
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt,
            directory=model_path,
            max_to_keep=train_config.max_ckpts_to_keep,
        )
        # TODO: also log the graph (now only in tf nightly :( )
        self._train_writer = tf.summary.create_file_writer(
            os.path.join(self._model_path, 'log', 'train')
        )
        self._val_writer = tf.summary.create_file_writer(
            os.path.join(self._model_path, 'log', 'val')
        )
        self._data_summary_fn = data_summary_fn
        self._aug_config = aug_config

    def train(self):
        train_config = self._train_config
        if train_config.shuffle:
            train_dataset = self._train_dataset.shuffle(
                train_config.shuffle_buffer,
                reshuffle_each_iteration=True
            )
        else:
            train_dataset = self._train_dataset
        train_dataset = train_dataset.repeat(train_config.epochs)
        train_dataset = train_dataset.batch(
            train_config.batch_size,
            drop_remainder=True
        )
        val_dataset = self._val_dataset.batch(
            train_config.batch_size,
        )

        if self._aug_config is not None:
            augment_fn = partial(
                augment, aug_config=self._aug_config
            )
            train_dataset = train_dataset.map(augment_fn)

        # Another way would be for loss_fn to return a batch of values
        # TODO: it now returns a batch of values, but code here is not
        # changed yet
        accm_train_loss = 0.0
        accm_val_loss = 0.0
        for step, train_batch in enumerate(train_dataset):
            global_step = train_config.start_step + step

            train_step_fn = (
                py_train_step if train_config.run_in_python
                else train_step
            )
            loss_values = train_step_fn(
                model=self._model,
                loss_fn=self._loss_fn,
                optimizer=self._optimizer,
                train_batch=train_batch,
                step=tf.convert_to_tensor(global_step),
            )
            total_loss = tf.reduce_mean(loss_values['loss'])
            accm_train_loss += total_loss

            if global_step % train_config.summary_every_n_steps == 0:
                print("Training: Step {}, Loss {}".format(
                    global_step, total_loss)
                )
                with self._train_writer.as_default():
                    for k, v in loss_values.items():
                        tf.summary.scalar(
                            'loss/{}'.format(k),
                            tf.reduce_mean(v),
                            step=global_step
                        )
                self._model.summary(
                    writer=self._train_writer,
                    batch=train_batch,
                    step=global_step,
                    training=True,
                )
                if self._data_summary_fn is not None:
                    self._data_summary_fn(
                        self._train_writer, train_batch, global_step
                    )

            if (
                train_config.eval_every_n_steps > 0 and
                global_step % train_config.eval_every_n_steps == 0
            ):
                # Run on validation dataset
                total_losses = defaultdict(float)
                total_cnt = 0
                for val_batch_idx, val_batch in enumerate(val_dataset):
                    val_step_fn = (
                        py_val_step if train_config.run_in_python
                        else val_step
                    )
                    loss_values = val_step_fn(
                        model=self._model,
                        loss_fn=self._loss_fn,
                        val_batch=val_batch,
                        step=tf.convert_to_tensor(global_step),
                    )
                    #mean_loss = tf.reduce_mean(loss_values['loss'])
                    #print("Val: Step {}, Batch {}, Loss {}".format(
                    #    global_step, val_batch_idx, mean_loss)
                    #)

                    for k, v in loss_values.items():
                        total_losses[k] += tf.reduce_sum(v)
                    total_cnt += tf.shape(loss_values['loss'])[0]

                with self._val_writer.as_default():
                    for k, v in total_losses.items():
                        tf.summary.scalar(
                            'loss/{}'.format(k),
                            v / tf.cast(total_cnt, tf.float32),
                            step=global_step
                        )
                self._model.summary(
                    writer=self._val_writer,
                    batch=val_batch,
                    step=global_step,
                    training=False,
                )

                mean_loss = total_losses['loss'] / tf.cast(
                    total_cnt, tf.float32
                )
                accm_val_loss += mean_loss
                print("Val: Step {}, Loss {}".format(
                    global_step, mean_loss
                ))

            if (
                train_config.save_every_n_steps > 0 and
                global_step % train_config.save_every_n_steps == 0
            ):
                cur_model_path = os.path.join(
                    self._model_path, str(global_step)
                )
                self._model.save_weights(cur_model_path)
                print("Model saved at {}".format(cur_model_path))

            self._train_writer.flush()
            self._val_writer.flush()

        # Save at the end of training
        cur_model_path = os.path.join(
            self._model_path, str(global_step)
        )
        self._model.save_weights(cur_model_path)
        print("Model saved at {}".format(cur_model_path))

        info = {
            'accumulated_train_loss': float(accm_train_loss),
            'accumulated_val_loss': float(accm_val_loss),
        }

        with open(os.path.join(
            self._model_path, 'train_info.yaml'
        ), 'w') as f:
            yaml.dump(info, f)

        return info


@tf.function(experimental_relax_shapes=True)
def train_step(model, loss_fn, optimizer, train_batch, step):
    with tf.GradientTape() as tape:
        outputs = model(train_batch, training=True)
        batch_loss = loss_fn(train_batch, outputs, step)
        loss_value = tf.reduce_mean(batch_loss['loss'])
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.train_callback()
    return batch_loss


@tf.function(experimental_relax_shapes=True)
def val_step(model, loss_fn, val_batch, step):
    outputs = model(val_batch, training=False)
    batch_loss = loss_fn(val_batch, outputs, step)
    return batch_loss


def py_train_step(model, loss_fn, optimizer, train_batch, step):
    with tf.GradientTape() as tape:
        outputs = model(train_batch, training=True)
        batch_loss = loss_fn(train_batch, outputs, step)
        loss_value = tf.reduce_mean(batch_loss['loss'])
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    model.train_callback()
    return batch_loss


def py_val_step(model, loss_fn, val_batch, step):
    outputs = model(val_batch, training=False)
    batch_loss = loss_fn(val_batch, outputs, step)
    return batch_loss

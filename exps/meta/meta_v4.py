import fire
import tensorflow as tf
from collections import deque, namedtuple
from functools import partial
import os
import numpy as np

from dl_playground.data.synthetic import (
    gaussian_in_nd,
    vertical_or_horizontal,
)
from dl_playground.data.osaka import (
    pretrain_dataset,
    stream_dataset,
    meta_train_config,
    meta_test_config,
)
from dl_playground.data.datasets import load_dataset
from dl_playground.networks.meta.stacked_masked import StackedMasked
from dl_playground.networks.meta.past_grads import (
    CoordWisePastGradsRNN,
    LayerCompetition,
)
from dl_playground.networks.meta.past_grads_v2 import DualRNN
from dl_playground.networks.layers.mlp import MLP
from dl_playground.exps.utils import load_and_save_config
from dl_playground.data.continual_utils import to_stream_dataset


TrainConfig = namedtuple('TrainConfig', [
    'window_size',
    'window_step',
    'graph_update_step',
    'log_inner_every_n_steps',
    'stop_gradient_from_meta',
    'stop_gradient_at_update',
    'use_val_loss',
    'meta_epochs',
    'n_steps_per_epoch',
    'shuffle',
    'shuffle_seed',
    'same_inner_init',
    'inner_init_path',
    'meta_init_path',
    'meta_lr',
    'meta_lr_decay',
    'inner_lr',
    'inner_pretrain_lr',
    'use_fwd_mask',
    'use_bwd_mask',
    'meta_gradient_clip_norm',
    'pass_bwd_mask_thru_layers',
    'meta_load_path',
    'meta_start_step',
    'meta_epoch_start',
    'inner_pretrain_steps',
    'run_inner_eval',
], defaults=[
    None,  # window_size
    None,  # window_step
    None,  # graph_update_step
    None,  # log_inner_every_n_steps
    True,  # stop_gradient_from_meta
    False,  # stop_gradient_at_update
    False,  # use_val_loss
    None,  # meta_epochs
    None,  # n_steps_per_epoch
    True,  # shuffle
    None,  # shuffle_seed
    False,  # same_inner_init
    None,  # inner_init_path
    None,  # meta_init_path
    None,  # meta_lr
    None,  # meta_lr_decay
    None,  # inner_lr
    None,  # inner_pretrain_lr
    True,  # use_fwd_mask
    False,  # use_bwd_mask
    None,  # meta_gradient_clip_norm
    False,  # pass_bwd_mask_thru_layers
    None,  # meta_load_path
    0,  # meta_start_step
    0,  # meta_epoch_start
    0,  # inner_pretrain_steps
    True,  # run_inner_eval
])

BUFFER_SIZE = 1000

cluster_to_class = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def preproc(sample):
    updated = {}
    updated['data'] = sample['data']
    label = sample['label']
    for old_label, new_label in enumerate(cluster_to_class):
        label = tf.where(
            tf.math.equal(label, old_label),
            new_label,
            tf.cast(label, tf.int32)
        )
    updated['label'] = label
    return updated


def run(config_path, model_path):
    config = load_and_save_config(config_path, model_path)

    train_config = TrainConfig(**config['train_config'])
    assert train_config.window_size <= train_config.graph_update_step

    meta_model_path = os.path.join(model_path, 'meta')
    inner_model_path = os.path.join(model_path, 'inner')
    meta_train_inner_model_path = os.path.join(
        inner_model_path, 'meta_train'
    )
    meta_val_inner_model_path = os.path.join(
        inner_model_path, 'meta_val'
    )

    pretrain_dset = None
    meta_test_train_dset = None
    meta_test_val_dset = None

    if config['dataset'] == 'guassian_in_nd':
        train_dset, val_dset = gaussian_in_nd(
            **config['dataset_configs']
        )
        train_dset = train_dset.map(preproc)
        val_dset = val_dset.map(preproc)
    elif config['dataset'] == 'mnist':
        train_dset = load_dataset(
            'mnist', split='train', to_float=True
        )
        val_dset = load_dataset('mnist', split='test', to_float=True)
    elif config['dataset'] == 'vertical_or_horizontal':
        train_dset, val_dset = vertical_or_horizontal(
            **config['dataset_configs']
        )
    elif config['dataset'] == 'osaka':
        if train_config.inner_pretrain_steps > 0:
            pretrain_dset = pretrain_dataset(meta_train_config)
        train_dset = stream_dataset(meta_train_config)
        meta_test_train_dset = stream_dataset(meta_test_config)
        val_dset = None
    else:
        raise ValueError()

    rnn_input_mlp = MLP(**config['rnn_input_mlp'])
    rnn_fwd_output_mlp = MLP(**config['rnn_fwd_output_mlp'])
    if train_config.use_bwd_mask:
        rnn_bwd_output_mlp = MLP(**config['rnn_bwd_output_mlp'])
    else:
        rnn_bwd_output_mlp = None

    if config['meta_cls'] == 'LayerCompetition':
        meta_cls = LayerCompetition
    elif config['meta_cls'] == 'DualRNN':
        meta_cls = DualRNN

    meta_net = meta_cls(
        input_mlp=rnn_input_mlp,
        fwd_output_mlp=rnn_fwd_output_mlp,
        bwd_output_mlp=rnn_bwd_output_mlp,
        use_bwd_mask=train_config.use_bwd_mask,
        **config['meta_net']
    )
    if train_config.meta_lr_decay is not None:
        meta_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=train_config.meta_lr,
            decay_steps=train_config.meta_lr_decay['decay_steps'],
            decay_rate=train_config.meta_lr_decay['decay_rate'],
        )
    else:
        meta_lr = train_config.meta_lr
    meta_opt = tf.keras.optimizers.Adam(meta_lr)
    meta_train_writer = tf.summary.create_file_writer(
        os.path.join(meta_model_path, 'log', 'train')
    )
    meta_val_writer = tf.summary.create_file_writer(
        os.path.join(meta_model_path, 'log', 'val')
    )
    meta_ckpt = tf.train.Checkpoint(
        meta_net=meta_net, meta_opt=meta_opt
    )

    meta_net.warm_start()

    # (load and) save the meta initial weights
    if train_config.meta_init_path is not None:
        status = meta_ckpt.restore(train_config.meta_init_path)
        # might not have meta_opt
        status.assert_existing_objects_matched()
        print('Loaded meta-net weights from {}'.format(
            train_config.meta_init_path)
        )
    meta_ckpt.save(os.path.join(meta_model_path, 'init'))

    # load the weights if specified
    if train_config.meta_load_path is not None:
        status = meta_ckpt.restore(train_config.meta_load_path)
        # meta_opt is lazily loaded
        status.assert_existing_objects_matched()
        print('Loaded meta-net weights from {}'.format(
            train_config.meta_load_path)
        )
        meta_ckpt.save(os.path.join(meta_model_path, 'loaded'))

    inner_net_create_fn = partial(
        StackedMasked,
        learning_rate=train_config.inner_lr,
        window_size=train_config.window_size,
        stop_gradient_from_meta=train_config.stop_gradient_from_meta,
        stop_gradient_at_update=train_config.stop_gradient_at_update,
        use_fwd_mask=train_config.use_fwd_mask,
        use_bwd_mask=train_config.use_bwd_mask,
        pass_bwd_mask_thru_layers=train_config.pass_bwd_mask_thru_layers,
        **config['inner_net']
    )

    # do shuffling before streaming
    if train_config.shuffle:
        train_dset = train_dset.shuffle(
            buffer_size=BUFFER_SIZE,
            seed=train_config.shuffle_seed,
            reshuffle_each_iteration=True,
        )
        val_dset = val_dset.shuffle(
            buffer_size=BUFFER_SIZE,
            seed=train_config.shuffle_seed,
            reshuffle_each_iteration=True,
        )

    if config['use_stream_dataset'] is True:
        assert config['dataset'] != 'osaka'

        n_classes = config['inner_net']['layer_configs'][-1]['fout']
        train_dset = to_stream_dataset(
            train_dset,
            n_classes=n_classes,
        )
        # TODO: fix validation set
        val_dset = to_stream_dataset(val_dset, n_classes=n_classes)

    train_dset = train_dset.batch(
        config['batch_size'], drop_remainder=True
    )
    if val_dset is not None:
        val_dset = val_dset.batch(
            config['batch_size'], drop_remainder=True
        ).repeat()

    if pretrain_dset is not None:
        pretrain_dset = pretrain_dset.batch(
            config['batch_size'], drop_remainder=True
        )

    if meta_test_train_dset is not None:
        meta_test_train_dset = meta_test_train_dset.batch(
            config['batch_size'], drop_remainder=True
        )
    if meta_test_val_dset is not None:
        meta_test_train_dset = meta_test_train_dset.batch(
            config['batch_size'], drop_remainder=True
        ).repeat()

    # train a baseline inner model
    # Note that this isn't necessarily a fair baseline as the optimal
    # learning rate is usually lower if not applying the masks
    print('=====================')
    print('Training baseline inner network..')
    inner_net = inner_net_create_fn(meta_net=None)
    inner_net.disable_fwd_masks()
    inner_net.disable_bwd_masks()

    # Load init weights if specified
    # don't save the meta weights (meta net is part of inner net)
    inner_ckpt = tf.train.Checkpoint(
        inner_net_mlp=inner_net.variables
    )
    if train_config.inner_init_path is not None:
        status = inner_ckpt.restore(train_config.inner_init_path)
        status.assert_consumed()
        inner_net.variables_updated()  # propogate loaded vars

    # Save init weights if specified
    if train_config.same_inner_init is True:
        inner_init_prefix = os.path.join(inner_model_path, 'init')
        inner_ckpt.save(inner_init_prefix)
        # get the ckpt path from prefix
        inner_init_path = tf.train.latest_checkpoint(
            os.path.dirname(inner_init_prefix)
        )
    else:
        inner_init_path = None

    #_ = meta_train_epoch(
    #    train_dset=train_dset,
    #    val_dset=val_dset,
    #    meta_net=None,
    #    inner_net=inner_net,
    #    meta_opt=meta_opt,
    #    config=train_config,
    #    inner_model_path=inner_model_path,
    #    epoch_idx='no_meta',
    #)

    print("inner init path:", inner_init_path)
    outer_step = train_config.meta_start_step

    for epoch_idx in range(
        train_config.meta_epoch_start,
        train_config.meta_epoch_start + train_config.meta_epochs
    ):
        print('=====================')
        print('Starting epoch {}, outer step {}..'.format(
            epoch_idx, outer_step
        ))
        inner_net = inner_net_create_fn(meta_net=meta_net)
        if train_config.same_inner_init is True:
            inner_ckpt = tf.train.Checkpoint(
                inner_net_mlp=inner_net.variables
            )
            status = inner_ckpt.restore(inner_init_path)
            status.assert_consumed()
            print("Loaded inner weights from {}".format(inner_init_path))
            inner_net.variables_updated()

        if train_config.inner_pretrain_steps > 0:
            print("Running inner net pretraining")
            assert pretrain_dset is not None

            inner_pretrain_writer = _create_inner_pretrain_writer(
                meta_train_inner_model_path,
                epoch_idx,
            )
            pretrain_losses = inner_pretrain(
                inner_net=inner_net,
                dataset=pretrain_dset,
                steps=train_config.inner_pretrain_steps,
                config=train_config,
                writer=inner_pretrain_writer,
            )

        train_losses, grad_ranges = meta_train_epoch(
            train_dset=train_dset,
            val_dset=val_dset,
            meta_net=meta_net,
            inner_net=inner_net,
            meta_opt=meta_opt,
            config=train_config,
            inner_model_path=meta_train_inner_model_path,
            epoch_idx=epoch_idx,
        )

        with meta_train_writer.as_default():
            for i, l in enumerate(train_losses):
                tf.summary.scalar('loss', l, step=outer_step + i)
                tf.summary.scalar(
                    'diag/gmin', grad_ranges[i][0], step=outer_step + i
                )
                tf.summary.scalar(
                    'diag/gmax', grad_ranges[i][1], step=outer_step + i
                )
            tf.summary.scalar(
                'loss/last_window', train_losses[-1], step=outer_step
            )
            tf.summary.scalar(
                'loss/all_windows',
                tf.reduce_sum(train_losses),
                step=outer_step,
            )

        inner_net = inner_net_create_fn(meta_net=meta_net)
        if train_config.same_inner_init is True:
            inner_ckpt = tf.train.Checkpoint(
                inner_net_mlp=inner_net.variables
            )
            status = inner_ckpt.restore(inner_init_path)
            status.assert_consumed()
            print("Loaded inner weights from {}".format(
                inner_init_path)
            )
            inner_net.variables_updated()

        if train_config.inner_pretrain_steps > 0:
            print("Running inner net pretraining")
            assert pretrain_dset is not None

            inner_pretrain_writer = _create_inner_pretrain_writer(
                meta_val_inner_model_path,
                epoch_idx,
            )
            pretrain_losses = inner_pretrain(
                inner_net=inner_net,
                dataset=pretrain_dset,
                steps=train_config.inner_pretrain_steps,
                config=train_config,
                writer=inner_pretrain_writer,
            )

        # lazy..
        if meta_test_train_dset is None:
            meta_test_train_dset = train_dset
        if meta_test_val_dset is None:
            meta_test_val_dset = val_dset

        # Each element a tuple (test_train_loss, test_val_loss)
        val_losses = meta_eval(
            meta_test_train_dset=meta_test_train_dset,
            meta_test_val_dset=meta_test_val_dset,
            meta_net=meta_net,
            inner_net=inner_net,
            config=train_config,
            inner_model_path=meta_val_inner_model_path,
            epoch_idx=epoch_idx,
        )

        with meta_val_writer.as_default():
            for i, (tl, vl) in enumerate(val_losses):
                tf.summary.scalar('tloss', tl, step=outer_step + i)
                tf.summary.scalar('vloss', vl, step=outer_step + i)
            tls, vls = zip(*val_losses)
            tf.summary.scalar(
                'tloss/last_window', tls[-1], step=outer_step
            )
            tf.summary.scalar(
                'vloss/last_window', vls[-1], step=outer_step
            )
            tf.summary.scalar(
                'tloss/all_windows',
                tf.reduce_sum(tls),
                step=outer_step,
            )
            tf.summary.scalar(
                'vloss/all_windows',
                tf.reduce_sum(vls),
                step=outer_step,
            )
        # assuming we run meta_val at every meta_train step
        outer_step += len(train_losses)

        meta_ckpt.save(os.path.join(meta_model_path, 'ckpt'))
        print('Meta checktpoing saved at {}'.format(
            tf.train.latest_checkpoint(meta_model_path)
        ))


def meta_train_epoch(
    train_dset,
    val_dset,
    meta_net,
    inner_net,
    meta_opt,
    config,
    inner_model_path,
    epoch_idx,
):
    window_losses = []
    grad_ranges = []

    inner_train_writer, inner_val_writer = _create_inner_writers(
        inner_model_path=inner_model_path, epoch_idx=epoch_idx,
    )

    if config.run_inner_eval:
        val_dset_iter = iter(val_dset)

    with tf.GradientTape(persistent=True) as tape:
        traj_losses = deque(maxlen=config.window_size)
        whole_traj_tloss = 0.0
        whole_traj_vloss = 0.0

        for inner_step, train_batch in enumerate(train_dset):
            train_loss, train_metr, _, _ = inner_train_step(
                batch=train_batch,
                net=inner_net,
                step=inner_step,
                meta_training=True,
                writer=inner_train_writer,
            )
            whole_traj_tloss += train_loss

            if config.run_inner_eval:
                val_batch = val_dset_iter.get_next()
                val_loss, val_metr = inner_eval(
                    batch=val_batch,
                    net=inner_net,
                    step=inner_step,
                    meta_training=True,
                )
                whole_traj_vloss += val_loss

            if config.use_val_loss is True:
                traj_losses.append(val_loss)
            else:
                traj_losses.append(train_loss)

            if inner_step % config.log_inner_every_n_steps == 0:
                with inner_train_writer.as_default():
                    tf.summary.scalar(
                        'loss', train_loss, inner_step
                    )
                    tf.summary.scalar(
                        'loss/cum', whole_traj_tloss, inner_step
                    )
                    tf.summary.scalar(
                        'metric', train_metr, inner_step
                    )
                if config.run_inner_eval:
                    with inner_val_writer.as_default():
                        tf.summary.scalar(
                            'loss', val_loss, inner_step
                        )
                        tf.summary.scalar(
                            'loss/cum', whole_traj_vloss, inner_step
                        )
                        tf.summary.scalar(
                            'metric', val_metr, inner_step
                        )

            if (
                meta_net is not None and
                (inner_step + 1) % config.window_step == 0
            ):
                # update the meta network
                window_loss = tf.reduce_mean(list(traj_losses))
                print("Meta train loss:", window_loss.numpy())

                with tape.stop_recording():
                    grads = tape.gradient(
                        window_loss, meta_net.trainable_variables
                    )
                    for v, g in zip(meta_net.trainable_variables, grads):
                        if tf.reduce_any(tf.math.is_nan(g)):
                            print("Gradient is NaN in {}".format(v.name))
                            raise RuntimeError()

                        if tf.reduce_max(tf.math.abs(g)) > 1e10:
                            print("Warning. Large gradient in {}!".format(v.name))

                    if config.meta_gradient_clip_norm is not None:
                        grads = [
                            tf.clip_by_norm(
                                g, config.meta_gradient_clip_norm
                            ) for g in grads
                        ]

                    updated = []
                    for g in grads:
                        ug = tf.where(
                            tf.math.abs(g) < 1e-20,
                            tf.zeros_like(g),
                            g
                        )
                        updated.append(ug)
                    grads = updated

                    meta_opt.apply_gradients(
                        zip(grads, meta_net.trainable_variables)
                    )

                    meta_net.train_callback()

                    gmaxs = [
                        tf.reduce_max(tf.math.abs(g)).numpy()
                        for g in grads
                    ]
                    gmins = [
                        tf.reduce_min(tf.math.abs(g)).numpy()
                        for g in grads
                    ]

                    gmax = np.max(gmaxs)
                    gmin = np.min(gmins)
                    grad_ranges.append((gmin, gmax))

                window_losses.append(window_loss)

            if (inner_step + 1) % config.graph_update_step == 0:
                inner_net.update_graph()
                tape.reset()

            if (
                config.n_steps_per_epoch is not None and
                inner_step + 1 == config.n_steps_per_epoch
            ):
                break

    del tape

    return window_losses, grad_ranges


def meta_eval(
    meta_test_train_dset,
    meta_test_val_dset,
    meta_net,
    inner_net,
    config,
    inner_model_path,
    epoch_idx,
):
    traj_train_losses = deque(maxlen=config.window_size)
    traj_val_losses = deque(maxlen=config.window_size)
    window_losses = []

    inner_train_writer, inner_val_writer = _create_inner_writers(
        inner_model_path=inner_model_path, epoch_idx=epoch_idx,
    )

    if config.run_inner_eval:
        val_dset_iter = iter(meta_test_val_dset)

    print("Running meta eval..")

    for inner_step, train_batch in enumerate(meta_test_train_dset):
        train_loss, train_metr, _, _ = inner_train_step(
            batch=train_batch,
            net=inner_net,
            step=inner_step,
            meta_training=False,
            writer=inner_train_writer,
        )
        traj_train_losses.append(train_loss)

        if config.run_inner_eval:
            val_batch = val_dset_iter.get_next()
            val_loss, val_metr = inner_eval(
                batch=val_batch,
                net=inner_net,
                step=inner_step,
                meta_training=False,
            )
            traj_val_losses.append(val_loss)

        if inner_step % config.log_inner_every_n_steps == 0:
            with inner_train_writer.as_default():
                tf.summary.scalar(
                    'loss', train_loss, inner_step
                )
                tf.summary.scalar(
                    'metric', train_metr, inner_step
                )
            if config.run_inner_eval:
                with inner_val_writer.as_default():
                    tf.summary.scalar(
                        'loss', val_loss, inner_step
                    )
                    tf.summary.scalar(
                        'metric', val_metr, inner_step
                    )

        if (inner_step + 1) % config.window_step == 0:
            train_window_loss = tf.reduce_mean(
                list(traj_train_losses)
            )
            print("Meta train loss:", train_window_loss.numpy())
            if config.run_inner_eval:
                val_window_loss = tf.reduce_mean(
                    list(traj_val_losses)
                )
                print("Meta val loss:", val_window_loss.numpy())
            else:
                val_window_loss = 0.0

            window_losses.append((train_window_loss, val_window_loss))

        if (inner_step + 1) % config.graph_update_step == 0:
            inner_net.update_graph()

        if (
            config.n_steps_per_epoch is not None and
            inner_step + 1 == config.n_steps_per_epoch
        ):
            break

    print("End of meta eval")

    return window_losses


def inner_train_step(
    batch, net, step,
    meta_training=None,
    writer=None,
):
    # First get the original gradient and the current loss
    with tf.GradientTape(persistent=True) as tape:
        # Since the weights are not variables anymore, need to watch
        # them. They need to be watched before calculations is applied
        # to them.
        # Also it seems like since the activations are intermediate
        # results anyway, I don't need to explicitly watch them.
        tape.watch(net.weights)
        net_out = net.forward(
            batch, training=True, meta_training=meta_training
        )
        losses = net.loss_fn(batch, net_out, step)

        mean_loss = tf.reduce_mean(losses['loss'])
        sum_loss = tf.reduce_sum(losses['loss'])

        mean_metric = tf.reduce_mean(net.metric_fn(batch, net_out))

    bmasks, inp_grads, used_grads = net.backward(tape, sum_loss)
    fmasks = net_out['masks']
    acts = net_out['acts']

    del tape

    if writer is not None:
        if 'task_id' in batch:
            with writer.as_default():
                tf.summary.scalar(
                    'diag/task_id',
                    tf.reduce_mean(batch['task_id']),
                    step,
                )
        if fmasks is not None:
            with writer.as_default():
                for mid, fm in enumerate(fmasks):
                    if fm is None:
                        continue
                    fm = fm.numpy()
                    tf.summary.scalar(
                        'diag/fmask{}_max'.format(mid),
                        fm.max(),
                        step
                    )
                    tf.summary.scalar(
                        'diag/fmask{}_min'.format(mid),
                        fm.min(),
                        step
                    )
                    tf.summary.scalar(
                        'diag/fmask{}_range'.format(mid),
                        fm.max() - fm.min(),
                        step
                    )

        if bmasks is not None:
            with writer.as_default():
                for mid, bm in enumerate(bmasks):
                    if bm is None:
                        continue
                    bm = bm.numpy()
                    tf.summary.scalar(
                        'diag/bmask{}_max'.format(mid),
                        bm.max(),
                        step
                    )
                    tf.summary.scalar(
                        'diag/bmask{}_min'.format(mid),
                        bm.min(),
                        step
                    )
                    tf.summary.scalar(
                        'diag/bmask{}_range'.format(mid),
                        bm.max() - bm.min(),
                        step
                    )

        with writer.as_default():
            for gid, g in enumerate(inp_grads):
                g = g.numpy()
                tf.summary.scalar(
                    'diag/input_g{}_max'.format(gid),
                    np.abs(g).max(),
                    step
                )
                tf.summary.scalar(
                    'diag/input_g{}_min'.format(gid),
                    np.abs(g).min(),
                    step
                )
                tf.summary.scalar(
                    'diag/input_g{}_mean'.format(gid),
                    g.mean(),
                    step
                )
                tf.summary.histogram(
                    'diag/input_g{}_hist'.format(gid),
                    g,
                    step
                )

            for gid, gs in enumerate(used_grads):
                gmax = np.max([g.numpy().max() for g in gs])
                gmin = np.min([g.numpy().min() for g in gs])
                tf.summary.scalar(
                    'diag/used_g{}_max'.format(gid),
                    gmax,
                    step
                )
                tf.summary.scalar(
                    'diag/used_g{}_min'.format(gid),
                    gmin,
                    step
                )

            for aid, act in enumerate(acts):
                act = act.numpy()
                tf.summary.scalar(
                    'diag/act{}_max'.format(aid),
                    np.abs(act).max(),
                    step
                )
                tf.summary.scalar(
                    'diag/act{}_min'.format(aid),
                    np.abs(act).min(),
                    step
                )
                tf.summary.histogram(
                    'diag/act{}_hist'.format(aid),
                    act,
                    step
                )

    return mean_loss, mean_metric, fmasks, bmasks


def inner_eval(batch, net, step, meta_training=None):
    out = net.forward(
        batch, training=False, meta_training=meta_training
    )
    losses = net.loss_fn(batch, out, step)
    metrics = net.metric_fn(batch, out)
    return tf.reduce_mean(losses['loss']), tf.reduce_mean(metrics)


def inner_pretrain(inner_net, dataset, steps, config, writer):
    # for pretraining, don't use masking
    fwd_setting, bwd_setting = inner_net.get_layer_mask_settings()
    inner_net.disable_fwd_masks()
    inner_net.disable_bwd_masks()

    # use pretraining learning rate
    inner_net.set_learning_rate(config.inner_pretrain_lr)

    tlosses = []
    for step, batch in enumerate(dataset):
        if step == steps:
            break

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inner_net.weights)
            net_out = inner_net.forward(
                batch, training=True, meta_training=False
            )
            losses = inner_net.loss_fn(batch, net_out, step)

            sum_loss = tf.reduce_sum(losses['loss'])
            mean_loss = tf.reduce_mean(losses['loss'])

        inner_net.backward(tape, sum_loss)

        del tape

        tlosses.append(mean_loss.numpy())

        if step % config.log_inner_every_n_steps == 0:
            with writer.as_default():
                tf.summary.scalar('loss', mean_loss, step)

    # set the inner net back
    inner_net.set_learning_rate(config.inner_lr)
    inner_net.set_layer_mask_settings(fwd_setting, bwd_setting)

    return tlosses


def _create_inner_pretrain_writer(inner_model_path, epoch_idx):
    if not type(epoch_idx) == int:
        subfolder = 'e{}'.format(epoch_idx)

    else:
        higher_folder = epoch_idx // 10
        lower_folder = epoch_idx % 10
        subfolder = 'e{}0/e{}{}'.format(
            higher_folder, higher_folder, lower_folder
        )

    writer = tf.summary.create_file_writer(os.path.join(
        inner_model_path, 'log', subfolder, 'pretrain')
    )
    return writer


def _create_inner_writers(inner_model_path, epoch_idx):
    if not type(epoch_idx) == int:
        subfolder = 'e{}'.format(epoch_idx)

    else:
        higher_folder = epoch_idx // 10
        lower_folder = epoch_idx % 10
        subfolder = 'e{}0/e{}{}'.format(
            higher_folder, higher_folder, lower_folder
        )

    train_writer = tf.summary.create_file_writer(os.path.join(
        inner_model_path, 'log', subfolder, 'train')
    )
    val_writer = tf.summary.create_file_writer(os.path.join(
        inner_model_path, 'log', subfolder, 'val')
    )
    return train_writer, val_writer


if __name__ == '__main__':
    fire.Fire(run)

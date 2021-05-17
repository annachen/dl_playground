from collections import namedtuple
import tensorflow as tf
import copy

from artistcritic.networks.meta.masked_layers import (
    MaskedLayer,
    MaskedDense,
    MaskedConv2D,
)
from artistcritic.utils.yaml_loadable import YAMLLoadable
from artistcritic.networks.classification.losses import (
    accuracy_metric
)


class StackedMasked(YAMLLoadable):
    def __init__(
        self,
        layer_configs,
        learning_rate,
        window_size=None,
        stop_gradient_from_meta=True,
        stop_gradient_at_update=False,
        mask_after_activation=True,
        use_fwd_mask=None,
        use_bwd_mask=None,
        pass_bwd_mask_thru_layers=False,
        record_masked_grads=False,
        bwd_return_grads=False,
        meta_net=None,
    ):
        self._local_layers = [
            _config_to_layer(
                config,
                window_size=window_size,
                stop_gradient_from_meta=stop_gradient_from_meta,
                stop_gradient_at_update=stop_gradient_at_update,
                mask_after_activation=mask_after_activation,
                use_fwd_mask=use_fwd_mask,
                use_bwd_mask=use_bwd_mask,
                bwd_return_grads=bwd_return_grads,
                meta_net=meta_net,
            ) for config in layer_configs
        ]
        self._lr = learning_rate
        self._pass_bwd_mask = pass_bwd_mask_thru_layers
        self._record_masked_grads = record_masked_grads

        self._n_classes = self._local_layers[-1]._fout

    def forward(self, batch, training=None, meta_training=None):
        if 'data' in batch:
            x = batch['data']
        elif 'image' in batch:
            x = batch['image']
        else:
            raise ValueError()

        masks = []
        acts = []
        for layer in self._local_layers:
            x = layer.call(
                x, training=training, meta_training=meta_training
            )
            if isinstance(layer, MaskedLayer):
                masks.append(layer._last_mask)
                acts.append(layer._last_act)
        return {
            'output': x,
            'masks': masks,
            'acts': acts,
        }

    def backward(self, tape, sum_loss):
        masks = []
        input_grads = []
        used_grads = []
        prev_mask = None
        for layer in self._local_layers[::-1]:
            # Layers without parameters, skip backward pass
            if not isinstance(layer, MaskedLayer):
                continue

            cur_grads = tape.gradient(sum_loss, layer._last_act)
            if prev_mask is not None and self._pass_bwd_mask:
                cur_grads = cur_grads * prev_mask

            if not self._record_masked_grads:
                layer.record_step(cur_grads)

            # for diagnosis
            input_grads.append(cur_grads)

            if self._pass_bwd_mask:
                grads, bmask, prev_mask = layer.calculate_gradients(
                    tape,
                    sum_loss,
                    cur_grads=cur_grads,
                    prev_mask=prev_mask,
                )
            else:
                grads, bmask, _ = layer.calculate_gradients(
                    tape, sum_loss, cur_grads=cur_grads,
                )

            if self._record_masked_grads:
                if bmask is None:
                    assert not layer._use_bwd_mask
                    layer.record_step(cur_grads)
                else:
                    layer.record_step(cur_grads * bmask)

            used_grads.append(grads)

            layer.update_weights(grads, self._lr)
            masks.append(bmask)
        return masks[::-1], input_grads[::-1], used_grads[::-1]

    @property
    def weights(self):
        ws = []
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                ws.extend(layer.weights)
        return ws

    @property
    def variables(self):
        vs = []
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                vs.extend(layer.variables)
        return vs

    def variables_updated(self):
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                layer.variables_updated()

    def update_graph(self):
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                layer.update_graph()

    def disable_fwd_masks(self):
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                layer._use_fwd_mask = False

    def disable_bwd_masks(self):
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                layer._use_bwd_mask = False

    def set_learning_rate(self, learning_rate):
        self._lr = learning_rate

    def get_layer_mask_settings(self):
        fwd_masks = []
        bwd_masks = []
        for layer in self._local_layers:
            if isinstance(layer, MaskedLayer):
                fwd_masks.append(layer._use_fwd_mask)
                bwd_masks.append(layer._use_bwd_mask)
            else:
                fwd_masks.append(None)
                bwd_masks.append(None)
        return fwd_masks, bwd_masks

    def set_layer_mask_settings(self, fwd_masks, bwd_masks):
        for lidx, layer in enumerate(self._local_layers):
            if isinstance(layer, MaskedLayer):
                layer._use_fwd_mask = fwd_masks[lidx]
                layer._use_bwd_mask = bwd_masks[lidx]

    def loss_fn(self, batch, pred, step):
        labels = batch['label']

        # (B,)
        class_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, depth=self._n_classes),
            logits=pred['output'],
        )

        loss = class_loss
        losses = {
            'loss': loss,
            'class_loss': class_loss,
        }
        return losses

    def metric_fn(self, batch, prediction):
        is_correct = accuracy_metric(
            prediction=prediction['output'],
            label=batch['label'],
            is_one_hot=False,
        )
        # (B,)
        return is_correct


class MaxPool2D:
    def __init__(self, kernel_size, padding):
        self._layer = tf.keras.layers.MaxPool2D(
            pool_size=kernel_size,
            strides=kernel_size,
            padding=padding,
        )

    def call(self, batch, training=None, meta_training=None):
        return self._layer.call(batch)


class Flatten:
    def __init__(self, fout):
        self._fout = fout

    def call(self, batch, training=None, meta_training=None):
        B = tf.shape(batch)[0]
        return tf.reshape(batch, (B, self._fout))


LayerConfig = namedtuple('LayerConfig', [
    'type',
    'window_size',
    'stop_gradient_from_meta',
    'stop_gradient_at_update',
    'mask_after_activation',
    'use_fwd_mask',
    'use_bwd_mask',
    'bwd_return_grads',
    'fin',  # dense and conv
    'fout',
    'activation',
    'act_fn_kwargs',
    'use_bias',
    'kernel_size',  # conv specific
    'stride',
    'padding',
], defaults=[None] * 15)


def _config_to_layer(
    config,
    window_size=None,
    stop_gradient_from_meta=True,
    stop_gradient_at_update=False,
    mask_after_activation=True,
    use_fwd_mask=None,
    use_bwd_mask=None,
    bwd_return_grads=False,
    meta_net=None,
):
    # do not change input
    config = copy.deepcopy(config)

    # better typo check..
    layer_config = LayerConfig(**config)

    # user layer-wise config if not None
    if layer_config.use_fwd_mask is None:
        config['use_fwd_mask'] = use_fwd_mask
    if layer_config.use_bwd_mask is None:
        config['use_bwd_mask'] = use_bwd_mask
    if layer_config.window_size is None:
        config['window_size'] = window_size
    if layer_config.stop_gradient_from_meta is None:
        config['stop_gradient_from_meta'] = stop_gradient_from_meta
    if layer_config.stop_gradient_at_update is None:
        config['stop_gradient_at_update'] = stop_gradient_at_update
    if layer_config.mask_after_activation is None:
        config['mask_after_activation'] = mask_after_activation

    # this can't have layer-wise config (as all layers share meta net)
    config['bwd_return_grads'] = bwd_return_grads

    config.pop('type')

    if layer_config.type == 'dense':
        layer = MaskedDense(meta_net=meta_net, **config)
    elif layer_config.type == 'conv2d':
        layer = MaskedConv2D(meta_net=meta_net, **config)
    elif layer_config.type == 'maxpool2d':
        layer = MaxPool2D(
            kernel_size=config['kernel_size'],
            padding=config['padding'],
        )
    elif layer_config.type == 'flatten':
        layer = Flatten(fout=config['fout'])
    else:
        raise ValueError()

    return layer

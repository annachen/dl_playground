import tensorflow as tf

from dl_playground.networks.layers.interface import BatchLayer
from dl_playground.networks.layers.cnn import ConvNet, LayerConfig
from dl_playground.networks.layers.mlp import MLP
from dl_playground.networks.classification.losses import (
    accuracy_metric
)


EPS = 1e-5


class ClassificationCNN(tf.keras.layers.Layer, BatchLayer):
    """A CNN with classification loss.

    Parameters
    ----------
    n_classes : int
    layer_configs : [LayerConfig]
        Configs for the convolutional layers.
    dense_filters : [int]
        Number of filters for the dense layers on top of the CNN.
        This does not include the last layer with `n_classes` outputs.
    dense_activity_regularizer : (str, float) | None
    n_heads : int
        Number of classification heads. One head for each
        classification task.
    heads_use_bias : bool

    """
    def __init__(
        self,
        n_classes,
        layer_configs,
        dense_filters,
        dense_activity_regularizer=None,
        n_heads=1,  # readouts
        heads_use_bias=True,
    ):
        super(ClassificationCNN, self).__init__()
        self._n_classes = n_classes
        self._layer_configs = layer_configs
        self._dense_filters = dense_filters
        self._n_heads = n_heads
        self._heads_use_bias = heads_use_bias

        self._n_conv_layers = len(self._layer_configs)
        # This does not include the readout layer
        self._n_dense_layers = len(self._dense_filters)

        self._cnn = ConvNet(layer_configs)
        self._dense = MLP(
            filters=dense_filters,
            activity_regularizer=dense_activity_regularizer,
        )
        self._clsf_heads = []
        for _ in range(self._n_heads):
            head = MLP(
                filters=[n_classes],
                last_layer_act_fn='linear',
                use_bias=heads_use_bias,
            )
            self._clsf_heads.append(head)

    def call(self, inputs, training=None, masks=None):
        """Runs the network.

        Parameters
        ----------
        inputs : dict
            "image": tf.Tensor, shape (B, H, W, C)
        training : bool | None
        masks : [tf.Tensor] | None
            Optionally mask the activations.

        Returns
        -------
        pred : tf.Tensor | [tf.Tensor]
            If n_heads is 1, the output of the head. Otherwise, a list
            of the outputs from the heads.

        """
        if masks is not None:
            conv_masks = masks[:self._n_conv_layers]
        else:
            conv_masks = None
        # (B, H', W', C')
        cnn_out = self._cnn(
            inputs['image'], training=training, masks=conv_masks
        )

        # (B, H'*W'*C')
        batch_size = tf.shape(cnn_out)[0]
        flatten = tf.reshape(cnn_out, (batch_size, -1))

        if masks is not None:
            dense_masks = masks[self._n_conv_layers : (
                self._n_conv_layers + self._n_dense_layers
            )]
        else:
            dense_masks = None
        # (B, F)
        dense_out = self._dense(
            flatten, training=training, masks=dense_masks
        )

        clsf_out = []
        for head in self._clsf_heads:
            out = head(dense_out, training=training)
            clsf_out.append(out)

        # for convenience
        if self._n_heads == 1:
            return clsf_out[0]

        return clsf_out

    def loss_fn(self, batch, prediction, step, task_id=None):
        """Loss calculation

        The loss is only calculated on the head indicated by `task_id`
        If `task_id` is None, then the first head would be used.

        Parameters
        ----------
        batch : dict
            "label" : tf.int32 (64?)
        prediction : tf.Tensor | [tf.Tensor]
            Output of the `call` function
        step : int
        task_id : int | None

        Returns
        -------
        losses : dict

        """
        labels = batch['label']
        if self._n_heads == 1:
            head_idx = 0
        elif task_id is None:
            prediction = prediction[0]
            head_idx = 0
        else:
            prediction = prediction[task_id]
            head_idx = task_id

        # (B,)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, depth=self._n_classes),
            logits=prediction,
        )
        losses = {'ce{}'.format(head_idx) : loss}

        reg = 0.0
        for layer in self._cnn._local_layers:
            for l in layer.losses:
                reg += l

        for layer in self._dense._local_layers:
            for l in layer.losses:
                reg += l
        losses['reg'] = reg

        loss += reg
        losses['loss'] = loss

        return losses

    def train_callback(self):
        pass

    def metric_fn(self, batch, prediction, task_id=None):
        if self._n_heads == 1:
            pass
        elif task_id is None:
            prediction = prediction[0]
        else:
            prediction = prediction[task_id]

        is_correct = accuracy_metric(
            prediction=prediction,
            label=batch['label'],
            is_one_hot=False,
        )
        # (B,)
        return is_correct

    def summary(self, writer, batch, step, training=None):
        _ = self.call(batch, training=training)
        cnn_acts = self._cnn.last_acts
        dense_acts = self._dense.last_acts

        zero_cnt = 0
        total_cnt = 0
        for act in cnn_acts + dense_acts:
            idxs = tf.where(act < EPS)
            cnt = tf.shape(idxs)[0]
            zero_cnt += cnt
            total_cnt += tf.math.reduce_prod(tf.shape(act))

        ratio = tf.cast(total_cnt - zero_cnt, tf.float32) / tf.cast(
            total_cnt, tf.float32
        )

        with writer.as_default():
            tf.summary.scalar(
                "diag/overall_activity_ratio", ratio, step
            )

        self._cnn.update_activity_monitor()
        self._dense.update_activity_monitor()

        cnn_ratio, cnn_cur_ratio =\
            self._cnn._act_monitor.activity_ratio()
        dense_ratio, dense_cur_ratio =\
            self._dense._act_monitor.activity_ratio()

        with writer.as_default():
            for name, ratio in cnn_ratio.items():
                tf.summary.histogram(
                    "diag/lifetime_act_ratio_{}".format(name),
                    tf.reshape(ratio, (-1,)),
                    step
                )
            for name, ratio in dense_ratio.items():
                tf.summary.histogram(
                    "diag/lifetime_act_ratio_{}".format(name),
                    tf.reshape(ratio, (-1,)),
                    step
                )
            for name, ratio in cnn_cur_ratio.items():
                tf.summary.histogram(
                    "diag/cur_act_ratio_{}".format(name),
                    tf.reshape(ratio, (-1,)),
                    step
                )
            for name, ratio in dense_cur_ratio.items():
                tf.summary.histogram(
                    "diag/cur_act_ratio_{}".format(name),
                    tf.reshape(ratio, (-1,)),
                    step
                )

    @classmethod
    def from_config(cls, config):
        layer_configs = config['layer_configs']
        layer_configs = [LayerConfig(**c) for c in layer_configs]
        return cls(
            layer_configs=layer_configs,
            **config['classification_cnn']
        )

import tensorflow as tf

from dl_playground.networks.layers.interface import BatchLayer
from dl_playground.networks.layers.mlp import MLP
from dl_playground.networks.classification.losses import (
    accuracy_metric
)


EPS = 1e-5


class ClassificationMLP(tf.keras.layers.Layer, BatchLayer):
    """A MLP with classification loss.

    Parameters
    ----------
    n_classes : int
    filters : [int]
    normalize_activation : bool
        Whether to normalize the activation such that it has norm == 1
    normalize_weights : bool
        Whether to normalize the weights

    """
    def __init__(
        self,
        n_classes,
        filters,
        act_fn='relu',
        normalize_activation=False,
        normalize_weights=False,
    ):
        super(ClassificationMLP, self).__init__()
        self._n_classes = n_classes
        self._filters = filters
        self._normalize_activation = normalize_activation
        self._normalize_weights = normalize_weights

        self._mlp = MLP(
            filters=filters + [n_classes],
            act_fn=act_fn,
            last_layer_act_fn='linear',
            normalize_weights=normalize_weights,
            normalize_activation=normalize_activation,
        )

    def call(self, batch, masks=None, training=None):
        """Runs the network.

        Parameters
        ----------
        batch : dict
            "data": tf.Tensor, shape (B, C)
        masks : [tf.Tensor] | None
        training : bool | None

        Returns
        -------
        pred : tf.Tensor, shape (B, n_classes)

        """
        return self._mlp(
            batch['data'],
            masks=masks,
            training=training,
        )

    def loss_fn(self, batch, prediction, step, task_id=None):
        """Loss calculation

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

        # (B,)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, depth=self._n_classes),
            logits=prediction,
        )
        losses = {'loss': loss}

        return losses

    def train_callback(self):
        self._mlp.train_callback()

    def metric_fn(self, batch, prediction):
        is_correct = accuracy_metric(
            prediction=prediction,
            label=batch['label'],
            is_one_hot=False,
        )
        # (B,)
        return is_correct

    def summary(self, writer, batch, step, training=None):
        pred = self.call(batch, training=training)
        is_correct = self.metric_fn(batch, pred)
        accuracy = tf.reduce_mean(is_correct)
        with writer.as_default():
            tf.summary.scalar('metric/accuracy', accuracy, step)

    @classmethod
    def from_config(cls, config):
        return cls(
            **config['classification_mlp']
        )

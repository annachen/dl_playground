"""Implements the layer interfaces used for batch learning"""

from enum import Enum
from artistcritic.utils.yaml_loadable import YAMLLoadable


class BatchLayer(YAMLLoadable):
    def predict(self, inputs):
        raise NotImplementedError()

    def metric_fn(self, batch, prediction):
        """
        Parameters
        ----------
        batch : tf.Tensor | dict
        prediction : tf.Tensor | dict

        Returns
        -------
        metric_type : MetricType
        metric_value : tf.Tensor, shape (B,)

        """
        raise NotImplementedError()

    def loss_fn(self, batch, prediction, step):
        """
        Parameters
        ----------
        batch : tf.Tensor | dict
        prediction : tf.Tensor | dict
            The output of the `call` function
        step : int

        Returns
        -------
        loss_values : dict
            'loss' : tf.Tensor, shape (B,)
                The total loss. This is used for optimiziation
            Other keys should contain different components of the
            loss. They should all have shape (B,) and will be
            displayed in TensorBoard.

        """
        raise NotImplementedError()

    def train_callback(self):
        raise NotImplementedError()

    def summary(self, writer, batch, step, training=None):
        raise NotImplementedError()


class MetricType(Enum):
    # type of metric: lower is better or higher is better
    LOW = 1
    HIGH = 2


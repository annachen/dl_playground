import tensorflow as tf

from dl_playground.networks.model import BatchModel, ContinualModel


class LayerModel(tf.keras.Model, BatchModel):
    """A model contains one layer.

    A thin wrapper around the layer to comply with model interface.

    Parameters
    ----------
    layer : tf.keras.layers.Layer & BatchLayer

    """
    def __init__(self, layer):
        super(LayerModel, self).__init__()
        self._layer = layer

    def call(self, inputs, training=None, **kwargs):
        return self._layer(inputs, training=training, **kwargs)

    def predict(self, inputs):
        return self._layer.predict(inputs)

    def loss_fn(self, batch, prediction, step):
        return self._layer.loss_fn(batch, prediction, step)

    def train_callback(self):
        return self._layer.train_callback()

    def metric_fn(self, batch, prediction):
        return self._layer.metric_fn(batch, prediction)

    def summary(self, writer, batch, step, training=None):
        return self._layer.summary(
            writer, batch, step, training=training
        )


class BatchLayerContinualModel(tf.keras.Model, ContinualModel):
    """A Model contains a single BatchLayer.

    Parameters
    ----------
    layer : BatchLayer

    """
    def __init__(self, layer, optimizer):
        super(BatchLayerContinualModel, self).__init__()
        self._layer = layer
        self._opt = optimizer
        self._step = 0

    def perceive(
        self,
        batch,
        freeze=False,
        return_eval=False,
        task_id=None,
    ):
        """Main function.

        Parameters
        ----------
        batch : dict | list(tf.Tensor) | tf.Tensor
        freeze : bool
        return_eval : bool
        task_id : int | None

        Returns
        -------
        pred : tf.Tensor | [tf.Tensor] | dict
            The output of the `call` function of the underlying layer
        perf : tf.Tensor
            Optional

        """
        with tf.GradientTape() as tape:
            pred = self._layer(batch, training=not freeze)
            losses = self._layer.loss_fn(
                batch=batch,
                prediction=pred,
                step=self._step,
                task_id=task_id,
            )
            loss = tf.reduce_mean(losses['loss'])

        if not freeze:
            grads = tape.gradient(loss, self._layer.trainable_weights)
            self._opt.apply_gradients(
                zip(grads, self._layer.trainable_weights)
            )

        self._step += 1

        if return_eval:
            perf = self._layer.metric_fn(
                batch=batch,
                prediction=pred,
                task_id=task_id,
            )
            return pred, losses, perf

        return pred

    def evaluate(self, batch, task_id=None):
        """Evaluates the batch.

        Parameters
        ----------
        batch : tf.Tensor | list | dict
        task_id : int | None

        Returns
        -------
        perf : tf.Tensor, shape (B,)

        """
        pred = self._layer.call(batch, training=False)
        perf = self._layer.metric_fn(
            batch=batch,
            prediction=pred,
            task_id=task_id,
        )
        return perf

    def eval_and_summary(self, writer, batch, step, task_id=None):
        perf = tf.reduce_mean(self.evaluate(batch, task_id=task_id))
        with writer.as_default():
            tf.summary.scalar('perf', tf.reduce_mean(perf), step=step)

    def summary(self, writer, batch, step, training=None):
        return self._layer.summary(
            writer, batch, step, training=training
        )

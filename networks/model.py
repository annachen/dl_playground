"""Implements the model interfaces used for continual learning"""


class BatchModel:
    def predict(self, inputs):
        raise NotImplementedError()

    def loss_fn(self, batch, prediction, step):
        raise NotImplementedError()

    def metric_fn(self, batch, prediction):
        raise NotImplementedError()

    def train_callback(self):
        raise NotImplementedError()

    def summary(self, writer, batch, step, training=None):
        raise NotImplementedError()


class ContinualModel:
    def perceive(self, batch, freeze=False, return_eval=False):
        """The main entry point of the model.

        This is a combination of the `call`, `evaluate` and optimize
        step in keras model.

        Parameters
        ----------
        batch : tf.Tensor | dict
        freeze : bool
            Whether to update the weights on the given batch
        return_eval : bool
            Whether to return an evaluation result.

        Returns
        -------
        output : tf.Tensor | dict(str -> tf.Tensor) | list(tf.Tensor)
            The output of the model
        perf : float
            Only returned if `return_eval` is True.

        """
        raise NotImplementedError()

    def evaluate(self, batch):
        """Runs evaluation on a set of data.

        Parameters
        ----------
        batch : tf.Tensor | dict

        Returns
        -------
        perf : float

        """
        raise NotImplementedError()

    def summary(self, writer, batch, step):
        """Runs the model and write summary.

        Parameters
        ----------
        writer : tf.FileWriter
        batch : tf.Tensor | dict
        step : int

        """
        raise NotImplementedError()

    def eval_and_summary(self, writer, batch, step):
        """Runs evaluation and log.

        Parameters
        ----------
        writer : tf.FileWriter
        batch : tf.Tensor | dict
        step : int

        """
        raise NotImplementedError()

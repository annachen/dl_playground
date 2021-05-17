import tensorflow as tf


class MaxAggregation(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        """Runs max aggregation

        Parameters
        ----------
        inputs : dict
          "value" : tf.Tensor, shape (B, N, F)
          "mask" : tf.Tensor, shape (B, N) | None

        Returns
        -------
        agg : tf.Tensor, shape (B, F)

        """
        if inputs['mask'] is None:
            filtered_value = inputs['value']
        else:
            # (B,)
            minv = tf.reduce_min(inputs['value'], axis=(1, 2)) - 1
            # (B, N, F)
            filtered_value = tf.where(
                inputs['mask'][..., tf.newaxis] > 0.5,
                inputs['value'],
                minv[:, tf.newaxis, tf.newaxis]
            )

        # (B, F)
        agg = tf.reduce_max(filtered_value, axis=1)

        return agg


class MeanAggregation(tf.keras.layers.Layer):
    def call(self, inputs, training=None):
        """Runs mean aggregation

        Parameters
        ----------
        inputs : dict
          "value" : tf.Tensor, shape (B, N, F)
          "mask" : tf.Tensor, shape (B, N) | None

        Returns
        -------
        agg : tf.Tensor, shape (B, F)

        """
        if inputs['mask'] is None:
            avg = tf.reduce_mean(inputs['value'], axis=1)
        else:
            # (B, F)
            s = tf.reduce_sum(
                inputs['value'] * inputs['mask'][..., tf.newaxis],
                axis=1
            )
            # (B,)
            mask_sum = tf.reduce_sum(inputs['mask'], axis=1)
            # This can be 0 if some samples in the batch are invalid
            # Make sure the mask is not 0 (the caller should be
            # responsible for filtering out the invalid samples)
            mask_sum = tf.maximum(mask_sum, 1.0)

            # (B, F)
            avg = s / mask_sum[:, tf.newaxis]
        return avg


class AttentionAggregation(tf.keras.layers.Layer):
    def __init__(self, n_feats):
        super(AttentionAggregation, self).__init__()
        self._F = n_feats

        initializer = tf.keras.initializers.GlorotNormal()
        self._w = tf.Variable(
            initial_value=initializer(shape=(self._F * 2, 1)),
            trainable=True,
        )

    def call(self, inputs, training=None):
        """Runs attention-like aggregation

        Parameters
        ----------
        inputs : dict
          "value" : tf.Tensor, shape (B, N, F)
          "mask" : tf.Tensor, shape (B, N) | None
          "query" : tf.Tensor, shape (B, F)

        Returns
        -------
        agg : tf.Tensor, shape (B, F)

        """
        N = tf.shape(inputs['value'])[1]
        query = tf.tile(inputs['query'][:, tf.newaxis], [1, N, 1])

        # this structure follows the graph attention network (GAT)
        # which concatenates the query with keys
        # (B, N, F*2)
        concated = tf.concat([inputs['value'], query], axis=2)
        # (B*N, F*2)
        concated = tf.reshape(concated, (-1, self._F * 2))

        # (B*N, 1)
        mat = tf.matmul(concated, self._w)
        # (B, N)
        mat = tf.reshape(mat, (-1, N))

        # following the GAT paper, adding leaky-relu
        mat = tf.nn.leaky_relu(mat)

        # (B, N)
        weights = tf.nn.softmax(mat, axis=1)

        if inputs['mask'] is not None:
            # mask and re-normalize
            masked = weights * inputs['mask']
            weights = masked / tf.reduce_sum(masked, axis=1)

        # (B, N, F)
        weighted = inputs['value'] * weights[..., tf.newaxis]
        # (B, F)
        avg = tf.reduce_sum(weighted, axis=1)

        return avg


LAYERS = {
    'max': MaxAggregation,
    'mean': MeanAggregation,
    'att': AttentionAggregation,
}

import tensorflow as tf

from dl_playground.networks.layers.interface import BatchLayer
from dl_playground.networks.layers.mlp import MLP
from dl_playground.networks.layers.gnn import GNNLayer
from dl_playground.networks.classification.losses import accuracy_metric

EPS = 1e-5


class GNNGlobalClassifier(tf.keras.layers.Layer, BatchLayer):
    def __init__(self, gnn, output_mlp, n_layers, n_classes, Din):
        super(GNNGlobalClassifier, self).__init__()
        self._gnn = gnn
        self._n_layers = n_layers
        self._n_classes = n_classes
        self._Din = Din  # input feature dims
        self._Dn = gnn._Dn  # intermediate feature dims
        self._output_mlp = output_mlp

    def call(self, batch, training=None):
        """Runs the network.

        Parameters
        ----------
        batch : dict
            "node_feats" : tf.RaggedTensor, (B, 1, N_i, Din)
            "adj_mat" : tf.SparseTensor, (B, max_N, max_N)

        """
        # (B, max_N, Din)
        node_feats = batch['node_feats'].to_tensor(
            shape=(None, 1, None, self._Din)
        )[:, 0]

        assert not self._Din > self._Dn

        # if Din is smaller than Dn, need to pad it since all layers
        # needs to have the same number of dimensions.
        if self._Din < self._Dn:
            node_feats = tf.pad(
                node_feats,
                [[0, 0], [0, 0], [0, self._Dn - self._Din]]
            )

        # (B, max_N)
        sum_nbrs = tf.sparse.reduce_sum(
            batch['adj_mat'], axis=2
        )
        # (B, max_N)
        mask = tf.where(
            sum_nbrs > EPS,
            tf.ones_like(sum_nbrs),
            tf.zeros_like(sum_nbrs),
        )

        x = batch.copy()
        x['node_feats'] = node_feats
        x['mask'] = mask
        x['edge_feats'] = None
        for _ in range(self._n_layers):
            x = self._gnn(x, training=training)
            x['adj_mat'] = batch['adj_mat']
            x['edge_feats'] = None

        # combine the nodes predictions and get output
        # (B, Dn)
        combined = tf.reduce_sum(x['node_feats'], axis=1) / (
            tf.reduce_sum(x['mask'], axis=1, keepdims=True)
        )
        combined.set_shape((None, self._Dn))

        # (B, n_classes)
        output = self._output_mlp.call(combined, training=training)

        return output

    def loss_fn(self, batch, prediction, step):
        assert self._output_mlp._last_layer_act_fn == 'linear'
        # (B,)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(batch['label'], depth=self._n_classes),
            logits=prediction,
        )
        return {'loss': loss}

    def metric_fn(self, batch, prediciton):
        is_correct = accuracy_metric(
            prediction=prediction,
            label=batch['label'],
            is_one_hot=False
        )
        return is_correct

    def train_callback(self):
        pass

    @classmethod
    def from_config(cls, config):
        message_mlp = MLP(**config['message_mlp'])
        node_update_mlp = MLP(**config['node_update_mlp'])
        gnn_layer = GNNLayer(
            message_mlp=message_mlp,
            node_update_mlp=node_update_mlp,
            **config['gnn_layer'],
        )
        output_mlp = MLP(**config['output_mlp'])
        classifier = cls(
            gnn=gnn_layer, output_mlp=output_mlp,
            **config['gnn_global_classifier']
        )
        return classifier

    def summary(self, writer, batch, step, training=None):
        pass

import tensorflow as tf


class GNNLayer(tf.keras.layers.Layer):
    """A Graph Neural Net layer, implements message passing.

    If we were to run PGM message passing (belief prop) with this,
    we'd treat the edges on the PGM as nodes here.

    """
    def __init__(
        self,
        Dn,
        message_mlp,
        node_update_mlp,
        message_mlp_input='concat',
        message_gather='mean',
        node_update_mlp_input='concat',
    ):
        super(GNNLayer, self).__init__()
        self._Dn = Dn
        self._F = node_update_mlp._filters[-1]
        self._message_mlp = message_mlp
        self._message_mlp_input = message_mlp_input
        self._message_gather = message_gather
        self._node_update_mlp = node_update_mlp
        self._node_update_mlp_input = node_update_mlp_input

    def call(self, batch, training=None):
        """Runs the network.

        Parameters
        ----------
        batch : dict
          "node_feats" : tf.RaggedTensor, (B, 1, N_i, Dn)
          "edge_feats" : tf.SparseTensor, (B, max_N, max_N, De)|None
          "adj_mat" : tf.SparseTensor, (B, max_N, max_N)
          "mask" : tf.Tensor, (B, max_N) | None

        """
        assert batch['edge_feats'] is None

        adj_mat = batch['adj_mat']

        if batch['mask'] is None:
            # (B, max_N)
            sum_nbrs = tf.sparse.reduce_sum(
                adj_mat, axis=2
            )
            # (B, max_N)
            mask = tf.where(
                sum_nbrs > EPS,
                tf.ones_like(sum_nbrs),
                tf.zeros_like(sum_nbrs),
            )
        else:
            mask = batch['mask']

        if isinstance(batch['node_feats'], tf.RaggedTensor):
            # (B, max_N, Dn)
            node_feats = batch['node_feats'].to_tensor(
                shape=(None, 1, None, self._Dn),
            )[:, 0]
        else:
            node_feats = batch['node_feats']
            node_feats.set_shape((None, None, self._Dn))

        # First, create new messages on each edges
        # (E, Dm); E: # of edges in the whole batch
        messages = self._create_messages(
            node_feats, adj_mat, training=training
        )

        # Gather incoming messages at each node
        # (B, max_N, Dm)
        gathered = self._gather_messages(
            messages, adj_mat, training=training
        )

        # Update the features at each node
        # (B, max_N, F)
        updated_node_feats = self._update_nodes(
            node_feats,
            gathered,
            training=training,
        )

        return {
            'node_feats': updated_node_feats,
            'mask': mask,
        }

    def _create_messages(self, node_feats, adj_mat, training):
        """Calculates messages on each edge.

        Parameters
        ----------
        node_feats : tf.Tensor, (B, max_N, Dn)
        adj_mat : tf.SparseTensor, (B, max_N, max_N)

        """
        # each message is a function of a node's feat and the nbrs'
        # feats
        # (E, 3)
        indices = adj_mat.indices
        # (E, 2)
        src_indices = indices[:, :2]
        # (E, 2)
        target_indices = tf.stack(
            [indices[:, 0], indices[:, 2]], axis=-1
        )
        # (E, Dn)
        src_feats = tf.gather_nd(
            params=node_feats,
            indices=src_indices,
        )
        # (E, Dn)
        target_feats = tf.gather_nd(
            params=node_feats,
            indices=target_indices,
        )

        if self._message_mlp_input == 'concat':
            # (E, Dn*2)
            feats = tf.concat([src_feats, target_feats], axis=-1)
        elif self._message_mlp_input == 'multiply':
            # (E, Dn)
            feats = src_feats * target_feats
        else:
            raise ValueError()

        # (E, Dm)
        messages = self._message_mlp.call(
            feats, training=training
        )
        return messages

    def _gather_messages(self, messages, adj_mat, training=None):
        """Gather messages from edges.

        Parameters
        ----------
        messages : tf.Tensor, (E, Dm)
        adj_mat : tf.SparseTensor, (B, max_N, max_N)

        """
        # (E, 3)
        idxs = tf.cast(adj_mat.indices, tf.int32)
        B = tf.shape(adj_mat)[0]
        max_N = tf.shape(adj_mat)[1]
        Dm = messages.shape[-1]

        # (E, Dm)
        weighted_m = messages * adj_mat.values[:, tf.newaxis]

        # convert the first 2 dims of the idxs into 1 number
        # (E,)
        B_N_idx = idxs[:, 0] * max_N + idxs[:, 1]

        # (B*max_N, max_N, Dm)
        ragged_messages = tf.RaggedTensor.from_value_rowids(
            values=weighted_m,
            value_rowids=B_N_idx,
            nrows=B * max_N,
            validate=False,
        )

        if self._message_gather == 'sum':
            # (B*max_N, Dm)
            gathered = tf.math.reduce_sum(ragged_messages, axis=1)
        elif self._message_gather == 'mean':
            # TODO: make sure these methods work correctly with ragged
            # TODO: this is giving me NaN
            gathered = tf.math.reduce_mean(ragged_messages, axis=1)
        else:
            raise ValueError()

        # (B, max_N, Dm) apparently already dense
        gathered = tf.reshape(
            gathered, (B, max_N, Dm)
        )
        return gathered

        """
        m_sp = tf.SparseTensor(
            indices=idxs,
            values=weighted_m,
            dense_shape=(B, max_N, max_N, Dm),
        )

        if self._message_gather == 'sum':
            # (B, max_N, Dm)
            gathered = tf.sparse.to_dense(tf.sparse.reduce_sum(
                m_sp, axis=2
            ))
        elif self._message_gather == 'mean':
            # (B, max_N, Dm)
            summed = tf.sparse.to_dense(tf.sparse.reduce_sum(
                m_sp, axis=2
            ))
            # (B, max_N)
            under = tf.sparse.to_dense(tf.sparse.reduce_sum(
                adj_mat, axis=2
            ))
            gathered = summed / under[..., tf.newaxis]
        else:
            raise ValueError()

        # (B, max_N, Dm)
        return gathered
        """

    def _update_nodes(self, node_feats, gathered, training=None):
        # node_feats : tf.Tensor, (B, max_N, Dn)
        # gathered : tf.Tensor, (B, max_N, Dm)
        if self._node_update_mlp_input == 'concat':
            # (B, max_N, Dn+Dm)
            feats = tf.concat([node_feats, gathered], axis=-1)
        else:
            raise ValueError()

        B = tf.shape(node_feats)[0]
        F = feats.shape[-1]
        updated = self._node_update_mlp.call(
            tf.reshape(feats, (-1, F)),
            training=training
        )

        Fout = updated.shape[-1]

        # (B, max_N, Fout)
        return tf.reshape(updated, (B, -1, Fout))

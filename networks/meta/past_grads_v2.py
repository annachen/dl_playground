"""Meta network using past gradients."""

import tensorflow as tf


class DualRNN(tf.keras.layers.Layer):
    """
    Pretty similar to LayerCompetition, except:
    1) Optionally aggregate features across batch before feeding into
       the RNN. Doing this because if the RNN states were to
       represent training state of the underlying network, the whole
       batch is used for the underlying network and not just one
       instance from the batch. Doing this also means that we'd be
       training the meta-network RNN with batch_size = 1.
    2) Extend the backward masking to be more similar to forward
       masking - the past masked gradients are passed into the RNN,
       while the current unmasked gradient is passed in through a
       separate branch. (This needs the inner network to pass in
       the masked gradient instead)
       Because the RNN states could potentially not have the batch
       dimension, we need to also pass in the current gradient at
       the end to get the mask output as (B, N)
    3) Added a few more options to experiment with different network
       design.

    Parameters
    ----------

    """
    def __init__(
        self,
        rnn_type,
        rnn_units,
        input_mlp,
        fwd_output_mlp,
        bwd_output_mlp,
        mask_thresh=0.1,
        dist_fn='none',
        use_bwd_mask=False,
        normalize_grads=False,
        normalize_acts=False,
        random_grads_stddev=None,
        use_nearest_grads=False,
        use_node_set=True,
        node_set_version='v3',
        use_batch_set=False,
        use_batch_summary=True,
        cur_reuse_branch=False,
        bwd_return_grads=False,
    ):
        super(DualRNN, self).__init__()
        assert rnn_type in ['simplernn', 'gru', 'lstm']

        if rnn_type == 'simplernn':
            self._rnn = tf.keras.layers.SimpleRNN(rnn_units)
        elif rnn_type == 'gru':
            self._rnn = tf.keras.layers.GRU(rnn_units)
        elif rnn_type == 'lstm':
            self._rnn = tf.keras.layers.LSTM(rnn_units)

        self._input_mlp = input_mlp
        self._fwd_output_mlp = fwd_output_mlp
        self._bwd_output_mlp = bwd_output_mlp
        self._mask_thresh = mask_thresh
        self._rnn_units = rnn_units
        self._dist_fn = dist_fn
        self._use_bwd_mask = use_bwd_mask
        self._normalize_grads = normalize_grads
        self._normalize_acts = normalize_acts
        self._use_node_set = use_node_set
        self._node_set_version = node_set_version
        self._use_batch_set = use_batch_set
        self._use_batch_summary = use_batch_summary
        self._random_grads_stddev = random_grads_stddev
        self._use_nearest_grads = use_nearest_grads
        self._cur_reuse_branch = cur_reuse_branch
        self._bwd_return_grads = bwd_return_grads

        self._last_input_mlp_input = None

        if self._fwd_output_mlp._last_layer_act_fn_str == 'linear':
            self._fwd_apply_sigmoid = True
        elif self._fwd_output_mlp._last_layer_act_fn_str == 'sigmoid':
            self._fwd_apply_sigmoid = False
        else:
            raise ValueError()

        if self._use_bwd_mask is False:
            assert self._bwd_output_mlp is None
        else:
            if self._bwd_output_mlp._last_layer_act_fn_str == 'linear':
                self._bwd_apply_sigmoid = True
            elif self._bwd_output_mlp._last_layer_act_fn_str == 'sigmoid':
                self._bwd_apply_sigmoid = False
            else:
                raise ValueError()

    def warm_start(self):
        batch = {
            'past_grads': tf.zeros((1, 1, 1)),
            'past_acts': tf.zeros((1, 1, 1)),
            'cur_acts': tf.zeros((1, 1)),
            'cur_grads': tf.zeros((1, 1)),
        }
        self.forward(batch, training=False)
        if self._use_bwd_mask:
            self.backward(batch, training=False)

    def first_forward(self, batch, training=None):
        """
        batch : (B, N)
            current activations.

        """
        B = tf.shape(batch)[0]
        N = tf.shape(batch)[1]

        # currently initial state is zeros
        h = tf.zeros((B * N, self._rnn_units), dtype=tf.float32)

        # prepare the branch from cur_acts
        if self._cur_reuse_branch:
            default_grads = self._get_default_grads(
                past_grads=None,
                past_acts=None,
                cur_acts=cur_acts,
            )
            # (B, 1, N, cur_F)
            cur_act_input, cur_F = self._prepare_input_mlp_input(
                past_acts=cur_acts[:, tf.newaxis],
                past_grads=default_grads
            )
            # (B*N, cur_F)
            cur_act_input = tf.reshape(cur_act_input, (B * N, cur_F))
            # (B*N, F)
            cur_act_feats = self._input_mlp.call(
                cur_act_input,
                training=training
            )
            F = self._input_mlp._filters[-1]
            cur_act_feats = tf.reshape(cur_act_feats, (B, 1, N, F))

            # also run set features on cur_acts
            # (B, 1, N, F')
            cur_act_feats, F_p = self._get_set_feature(
                cur_act_feats, F
            )
        else:
            if self._normalize_acts:
                # (B, N), (B, 1)
                nacts, norm = _safe_normalize(batch, axis=-1)
                norm = tf.tile(norm, [1, N])
                # (B, N, 2)
                cur_act_feats = tf.stack([nacts, norm], axis=-1)
                F_p = 2
            else:
                cur_act_feats = cur_acts
                F_p = 1

        # concat with current activation to feed into output_mlp
        # (B*N, U+F')
        feat = tf.concat([
            h, tf.reshape(cur_act_feats, (B * N, F_p))
        ], axis=-1)

        out = self._fwd_output_mlp(feat, training=training)
        # (B, N)
        out = tf.reshape(out, (B, N))

        if self._fwd_apply_sigmoid:
            mask = tf.nn.sigmoid(out)
        else:
            mask = out

        # to avoid gradient underflow in the inner net, make mask
        # smaller than `mask_thresh` 0s
        mask = tf.where(
            mask < self._mask_thresh,
            tf.zeros_like(mask),
            mask,
        )
        return mask

    def forward(self, batch, training=None):
        """Returns the mask for forward inner network

        Parameters
        ----------
        batch : dict
          "past_grads" : (B, T, N)
          "past_acts" : (B, T, N)
          "cur_acts" : (B, N)

        """
        past_grads = batch['past_grads']
        past_acts = batch['past_acts']
        cur_acts = batch['cur_acts']

        B = tf.shape(cur_acts)[0]
        N = tf.shape(cur_acts)[1]
        T = tf.shape(past_grads)[1]

        # (B, T, N, Fin)
        feat, Fin = self._prepare_input_mlp_input(
            past_grads=past_grads,
            past_acts=past_acts,
        )
        feat = tf.reshape(feat, (-1, Fin))
        #print("fwd Fin: {}".format(Fin))

        # (B * T * N, F)
        feat = self._input_mlp.call(feat, training=training)

        F = self._input_mlp._filters[-1]
        feat = tf.reshape(feat, (B, T, N, F))

        # (B, T, N, F')
        all_feats, F_p = self.get_set_feature(feat, F)
        #print("fwd Fp: {}".format(F_p))

        if self._use_batch_summary:
            # (T, N, F')
            all_feats, F_p = self._get_batch_summary(all_feats, F_p)
            # (N, T, F')
            seq = tf.transpose(all_feats, (1, 0, 2))
            # (N, U)
            last_h = self._rnn(seq, training=training)
            # (B, N, U)
            last_h = tf.tile(last_h[tf.newaxis], [B, 1, 1])
            last_h = tf.reshape(last_h, (B * N, self._rnn_units))
        else:
            # (B, N, T, F')
            seq = tf.transpose(all_feats, (0, 2, 1, 3))
            seq = tf.reshape(seq, (B * N, T, F_p))

            # (B*N, U)
            last_h = self._rnn(seq, training=training)

        # prepare the branch from cur_acts
        if self._cur_reuse_branch:
            default_grads = self._get_default_grads(
                past_grads=past_grads,
                past_acts=past_acts,
                cur_acts=cur_acts,
            )
            # (B, 1, N, cur_F)
            cur_act_input, cur_F = self._prepare_input_mlp_input(
                past_acts=cur_acts[:, tf.newaxis],
                past_grads=default_grads
            )
            # (B*N, cur_F)
            cur_act_input = tf.reshape(cur_act_input, (-1, cur_F))
            # (B*N, F)
            cur_act_feats = self._input_mlp.call(
                cur_act_input,
                training=training
            )
            F = self._input_mlp._filters[-1]
            cur_act_feats = tf.reshape(cur_act_feats, (B, 1, N, F))

            # also run set features on cur_acts
            # (B, 1, N, F')
            cur_act_feats, F_p = self._get_set_feature(
                cur_act_feats, F
            )
        else:
            if self._normalize_acts:
                # (B, N), (B, 1)
                nacts, norm = _safe_normalize(cur_acts, axis=-1)
                norm = tf.tile(norm, [1, N])
                cur_act_feats = tf.stack([nacts, norm], axis=-1)
                F_p = 2
            else:
                cur_act_feats = cur_acts
                F_p = 1

        # prepare inputs for output_mlp
        # (B*N, U + F')
        feat = tf.concat([
            last_h,
            tf.reshape(cur_act_feats, (B * N, F_p))
        ], axis=-1)

        out = self._fwd_output_mlp(feat, training=training)

        # (B, N)
        out = tf.reshape(out, (B, N))

        if self._fwd_apply_sigmoid:
            mask = tf.nn.sigmoid(out)
        else:
            mask = out

        # to avoid gradient underflow in the inner net, make mask
        # smaller than `mask_thresh` 0s
        # TODO: not sure if this is needed
        mask = tf.where(
            mask < self._mask_thresh,
            tf.zeros_like(mask),
            mask,
        )
        return mask

    def first_backward(self, batch, training=None):
        """Returns the mask for backward gradient masking

        Parameters
        ----------
        batch : dict
          "cur_acts" : (B, N)
          "cur_grads" : (B, N)

        """
        cur_acts = batch['cur_acts']
        cur_grads = batch['cur_grads']

        B = tf.shape(cur_acts)[0]
        N = tf.shape(cur_acts)[1]

        # currently initial state is zeros
        h = tf.zeros((B * N, self._rnn_units), dtype=tf.float32)

        # prepare the branch from cur_acts
        if self._cur_reuse_branch:
            # (B, 1, N, cur_F)
            cur_input, cur_F = self._prepare_input_mlp_input(
                past_acts=cur_acts[:, tf.newaxis],
                past_grads=cur_grads[:, tf.newaxis],
            )
            # (B*N, cur_F)
            cur_input = tf.reshape(cur_input, (B * N, cur_F))
            # (B*N, F)
            cur_feats = self._input_mlp.call(
                cur_input,
                training=training
            )
            F = self._input_mlp._filters[-1]
            cur_feats = tf.reshape(cur_feats, (B, 1, N, F))

            # also run set features on cur_feats
            # (B, 1, N, F')
            cur_feats, F_p = self._get_set_feature(
                cur_feats, F
            )
        else:
            if self._normalize_acts:
                # (B, N), (B, 1)
                nacts, norm = _safe_normalize(cur_acts, axis=-1)
                norm = tf.tile(norm, [1, N])
                # (B, N, 2)
                cur_feats = tf.stack([nacts, norm], axis=-1)
                F_p = 2
            else:
                cur_feats = cur_acts
                F_p = 1

            if self._normalize_grads:
                ngrads, norm = _safe_normalize(cur_grads, axis=-1)
                norm = tf.tile(norm, [1, N])
                cur_feats = tf.concat([
                    cur_feats,
                    ngrads[..., tf.newaxis],
                    norm[..., tf.newaxis]
                ], axis=-1)
                F_p += 2
            else:
                cur_feats = tf.concat([
                    cur_feats, cur_grads[..., tf.newaxis]
                ], axis=-1)
                F_p += 1

        # concat with current activation to feed into output_mlp
        # (B*N, U+F')
        feat = tf.concat([
            h, tf.reshape(cur_feats, (B * N, F_p))
        ], axis=-1)

        out = self._bwd_output_mlp(feat, training=training)
        # (B, N)
        out = tf.reshape(out, (B, N))

        if self._bwd_return_grads:
            weights = tf.nn.softmax(
                tf.reshape(out, (B, N, 4)), axis=-1
            )
            grads = self._bwd_weighted_grads(
                cur_grads=cur_grads,
                weights=weights,
            )
            return grads

        if self._bwd_apply_sigmoid:
            mask = tf.nn.sigmoid(out)
        else:
            mask = out

        # to avoid gradient underflow in the inner net, make mask
        # smaller than `mask_thresh` 0s
        mask = tf.where(
            mask < self._mask_thresh,
            tf.zeros_like(mask),
            mask,
        )
        return mask

    def backward(self, batch, training=None):
        """Returns the mask for backward gradient masking

        Parameters
        ----------
        batch : dict
          "past_grads" : (B, T, N)
          "past_acts" : (B, T, N)
          "cur_acts" : (B, N)
          "cur_grads" : (B, N)

        """
        past_grads = batch['past_grads']
        past_acts = batch['past_acts']
        cur_acts = batch['cur_acts']
        cur_grads = batch['cur_grads']

        B = tf.shape(cur_acts)[0]
        N = tf.shape(cur_acts)[1]
        T = tf.shape(past_grads)[1]

        # (B, T, N, Fin)
        feat, Fin = self._prepare_input_mlp_input(
            past_grads=past_grads,
            past_acts=past_acts,
        )
        feat = tf.reshape(feat, (-1, Fin))
        #print("bwd Fin: {}".format(Fin))

        # (B * T * N, F)
        feat = self._input_mlp.call(feat, training=training)

        F = self._input_mlp._filters[-1]
        feat = tf.reshape(feat, (B, T, N, F))

        # (B, T, N, F')
        all_feats, F_p = self.get_set_feature(feat, F)
        #print("bwd Fp: {}".format(F_p))

        if self._use_batch_summary:
            # (T, N, F')
            all_feats, F_p = self._get_batch_summary(all_feats, F_p)
            # (N, T, F')
            seq = tf.transpose(all_feats, (1, 0, 2))
            # (N, U)
            last_h = self._rnn(seq, training=training)
            # (B, N, U)
            last_h = tf.tile(last_h[tf.newaxis], [B, 1, 1])
            last_h = tf.reshape(last_h, (B * N, self._rnn_units))
        else:
            # (B, N, T, F')
            seq = tf.transpose(all_feats, (0, 2, 1, 3))
            seq = tf.reshape(seq, (B * N, T, F_p))

            # (B*N, U)
            last_h = self._rnn(seq, training=training)

        # prepare the branch from cur_acts
        if self._cur_reuse_branch:
            # (B, 1, N, cur_F)
            cur_input, cur_F = self._prepare_input_mlp_input(
                past_acts=cur_acts[:, tf.newaxis],
                past_grads=cur_grads[:, tf.newaxis],
            )
            # (B*N, cur_F)
            cur_input = tf.reshape(cur_input, (-1, cur_F))
            # (B*N, F)
            cur_feats = self._input_mlp.call(
                cur_input,
                training=training
            )
            F = self._input_mlp._filters[-1]
            cur_feats = tf.reshape(cur_feats, (B, 1, N, F))

            # also run set features on cur_acts
            # (B, 1, N, F')
            cur_feats, F_p = self._get_set_feature(
                cur_feats, F
            )
        else:
            if self._normalize_acts:
                # (B, N), (B, 1)
                nacts, norm = _safe_normalize(cur_acts, axis=-1)
                norm = tf.tile(norm, [1, N])
                cur_feats = tf.stack([nacts, norm], axis=-1)
                F_p = 2
            else:
                cur_feats = cur_acts
                F_p = 1

            if self._normalize_grads:
                ngrads, norm = _safe_normalize(cur_grads, axis=-1)
                norm = tf.tile(norm, [1, N])
                cur_feats = tf.concat([
                    cur_feats,
                    ngrads[..., tf.newaxis],
                    norm[..., tf.newaxis]
                ], axis=-1)
                F_p += 2
            else:
                cur_feats = tf.concat([
                    cur_feats, cur_grads[..., tf.newaxis]
                ], axis=-1)
                F_p += 1

        # prepare inputs for output_mlp
        # (B*N, U + F')
        feat = tf.concat([
            last_h,
            tf.reshape(cur_feats, (B * N, F_p))
        ], axis=-1)

        out = self._bwd_output_mlp(feat, training=training)

        if self._bwd_return_grads:
            weights = tf.nn.softmax(
                tf.reshape(out, (B, N, 4)), axis=-1
            )
            grads = self._bwd_weighted_grads(
                cur_grads=cur_grads,
                weights=weights,
            )
            return grads

        # (B, N)
        out = tf.reshape(out, (B, N))

        if self._bwd_apply_sigmoid:
            mask = tf.nn.sigmoid(out)
        else:
            mask = out

        # to avoid gradient underflow in the inner net, make mask
        # smaller than `mask_thresh` 0s
        # TODO: not sure if this is needed
        mask = tf.where(
            mask < self._mask_thresh,
            tf.zeros_like(mask),
            mask,
        )
        return mask

    def _prepare_input_mlp_input(self, past_grads, past_acts):
        if self._normalize_acts:
            # (B, T, N), (B, T, 1)
            nacts, norm = _safe_normalize(past_acts, axis=2)
            N = tf.shape(nacts)[-1]
            # (B, T, N)
            norm = tf.tile(norm, [1, 1, N])
            # (B, T, N, 2)
            feat = tf.stack([nacts, norm], axis=-1)
            F = 2
        else:
            # (B, T, N, 1)
            feat = past_acts[..., tf.newaxis]
            F = 1

        if self._normalize_grads:
            # (B, T, N), (B, T, 1)
            ngrads, norm = _safe_normalize(past_grads, axis=2)
            N = tf.shape(ngrads)[-1]
            # (B, T, N)
            norm = tf.tile(norm, [1, 1, N])
            # (B, T, N, F+2)
            feat = tf.concat([
                feat, ngrads[..., tf.newaxis], norm[..., tf.newaxis]
            ], axis=-1)
            F = F + 2
        else:
            feat = tf.concat([feat, past_grads[..., tf.newaxis]])
            F = F + 1

        return feat, F

    def get_set_feature(self, feat, F):
        """Returns the features extracted based on sets.

        Parameters
        ----------
        feat : tf.Tensor, shape (B, T, N, F)
            `N` is the dimension for the set
        F : int
            The number of channels for the input feature

        Returns
        -------
        set_feat : tf.Tensor, shape (B, T, N, F')
        F' : int
            The number of channels of the output feature

        """
        if not self._use_node_set and not self._use_batch_set:
            # if coordinate-wise, use original features
            return feat, F

        if self._use_node_set:
            if self._node_set_version == 'v1':
                feat, F = self._get_node_set_feature(feat, F)
            elif self._node_set_version == 'v2':
                feat, F = self._get_node_set_feature_v2(feat, F)
            elif self._node_set_version == 'v3':
                # (B, T, N, Fn)
                feat, F = self._get_node_set_feature_v3(feat, F)
            else:
                raise ValueError()

        if self._use_batch_set:
            feat_b, Fb = self._get_batch_set_feature(feat, F)
            if self._use_node_set:
                feat = tf.concat([feat, feat_b], axis=-1)
                F = F + Fb
            else:
                feat = feat_b
                F = Fb

        return feat, F

    def _get_node_set_feature(self, feat, F):
        """Returns the features extracted based on sets.

        Parameters
        ----------
        feat : tf.Tensor, shape (B, T, N, F)
            `N` is the dimension for the set
        F : int
            The number of channels for the input feature

        Returns
        -------
        set_feat : tf.Tensor, shape (B, T, N, F')
        F' : int
            The number of channels of the output feature

        """
        B = tf.shape(feat)[0]
        T = tf.shape(feat)[1]

        # (BT, N, F)
        feat = tf.reshape(feat, (B * T, -1, F))

        # obtain pair-wise feats for nodes
        # (BT, N, 1, F)
        src_feat = feat[:, :, tf.newaxis, :]
        # (BT, 1, N, F)
        dst_feat = feat[:, tf.newaxis, :, :]

        N = tf.shape(feat)[1]
        BT = B * T

        if self._dist_fn == 'diff':
            # (BT, N, N, F)
            dist = dst_feat - src_feat
            self_feat = feat
        elif self._dist_fn == 'dot':
            # (BT, N, N, F)
            dist = dst_feat * src_feat
            self_feat = feat
        elif self._dist_fn == 'norm_dot':
            n_dst_feat, _ = _safe_normalize(dst_feat, axis=-1)
            n_src_feat, _ = _safe_normalize(src_feat, axis=-1)
            dist = tf.reduce_sum(
                n_dst_feat * n_src_feat, axis=-1, keepdims=True
            )
            self_feat = tf.ones([BT, N, 1])
            F = 1
        elif self._dist_fn == 'concat':
            # (BT, N, N, F*2)
            dist = tf.concat([
                tf.tile(src_feat, [1, 1, N, 1]),
                tf.tile(dst_feat, [1, N, 1, 1])
            ], axis=-1)
            # (BT, N, F*2)
            self_feat = tf.concat([feat, feat], axis=-1)
            F = F * 2
        elif self._dist_fn == 'none':
            # need to tile the first `N` dimension and not the 2nd
            # (BT, N, N, F)
            dist = tf.tile(dst_feat, [1, N, 1, 1])
            # (BT, N, F)
            self_feat = feat
        else:
            raise ValueError()

        # (N, N, B*T, F)
        dist = tf.transpose(dist, (1, 2, 0, 3))

        # Aggregate over node features
        # Create an "other" mask
        mask = tf.ones((N, N)) - tf.eye(N)
        # (N * (N-1), 2)
        to_take = tf.where(mask > 0.5)
        # (N * (N-1), BT, F)
        gathered = tf.gather_nd(dist, to_take)
        # (N, N-1, BT, F)
        other_feat = tf.reshape(gathered, (N, N - 1, BT, F))

        # So, what are some options after here?
        # I have NxN pairwise distance, and eventually I want to
        # reduce to N and the RNN will share weights among the N
        # nodes.

        # It'd be quite intuitive to apply attention of some form to
        # see what are the other nodes that a node should pay
        # attention to. So feature for RNN input would be
        # concat(self_feat, att(other_feat))

        # I don't want to directly aggregate from NxN -> N without
        # distinguish self-vs-other because, well, seems like a
        # useful distinction.

        # But perhaps I'll start with some hard coded aggregation

        # (BT, N, N-1, F)
        other_feat = tf.transpose(other_feat, (2, 0, 1, 3))

        # (BT, N, F)
        other_mean = tf.reduce_mean(other_feat, axis=2)
        other_min = tf.reduce_min(other_feat, axis=2)
        other_max = tf.reduce_max(other_feat, axis=2)

        # put them together
        agg_feats = [self_feat, other_mean, other_min, other_max]
        n_agg_feats = len(agg_feats)

        # (BT, N, F*n_agg_feats)
        all_feats = tf.concat(agg_feats, axis=-1)
        all_feats = tf.reshape(all_feats, (B, T, N, F * n_agg_feats))

        return all_feats, F * n_agg_feats

    def _get_node_set_feature_v2(self, feat, F):
        """Returns the features extracted based on sets.

        Skip the pairwise distance as in v1 as it takes too much
        memory. Start looking at aggregation stats directly.

        Parameters
        ----------
        feat : tf.Tensor, shape (B, T, N, F)
            `N` is the dimension for the set
        F : int
            The number of channels for the input feature

        Returns
        -------
        set_feat : tf.Tensor, shape (B, T, N, F')
        F' : int
            The number of channels of the output feature

        """
        B = tf.shape(feat)[0]
        T = tf.shape(feat)[1]
        N = tf.shape(feat)[2]

        def _other_stats(self_idx):
            # (N,)
            self_idx_one_hot = tf.one_hot(self_idx, depth=N)
            # (N-1, B, T, F)
            other_feat = tf.gather(
                tf.transpose(feat, (2, 0, 1, 3)),  # (N, B, T, F)
                tf.where(self_idx_one_hot < 0.5)[:, 0],  # (N-1, 1)
            )
            # (B, T, N-1, F)
            other_feat = tf.transpose(other_feat, (1, 2, 0, 3))
            # (B, T, F)
            other_min = tf.reduce_min(other_feat, axis=2)
            other_max = tf.reduce_max(other_feat, axis=2)
            other_mean = tf.reduce_mean(other_feat, axis=2)

            # (B, T, F * 3)
            return tf.concat(
                [other_min, other_max, other_mean], axis=-1
            )

        self_idxs = tf.range(N)
        # (N, B, T, F * 3)
        other_feats = tf.map_fn(
            fn=_other_stats,
            elems=self_idxs,
            fn_output_signature=tf.float32,
        )
        # (B, T, N, F*3)
        other_feats = tf.transpose(other_feats, (1, 2, 0, 3))
        # (B, T, N, F*4)
        all_feats = tf.concat([feat, other_feats], axis=-1)

        return all_feats, F * 4

    def _get_node_set_feature_v3(self, feat, F):
        """Returns the features extracted based on sets.

        Skip the pairwise distance as in v1 as it takes too much
        memory. Start looking at aggregation stats directly.
        Skip self vs other and just use self vs all.

        Parameters
        ----------
        feat : tf.Tensor, shape (B, T, N, F)
            `N` is the dimension for the set
        F : int
            The number of channels for the input feature

        Returns
        -------
        set_feat : tf.Tensor, shape (B, T, N, F')
        F' : int
            The number of channels of the output feature

        """
        B = tf.shape(feat)[0]
        T = tf.shape(feat)[1]
        N = tf.shape(feat)[2]

        # (B, T, 1, F)
        all_min = tf.reduce_min(feat, axis=2, keepdims=True)
        all_max = tf.reduce_max(feat, axis=2, keepdims=True)
        all_mean = tf.reduce_mean(feat, axis=2, keepdims=True)
        # (B, T, 1, F*3)
        all_feats = tf.concat([all_min, all_max, all_mean], axis=-1)

        # (B, T, N, F*3)
        all_feats = tf.tile(all_feats, [1, 1, N, 1])

        # (B, T, N, F*4)
        all_feats = tf.concat([feat, all_feats], axis=-1)

        return all_feats, F * 4

    def _get_batch_summary(self, feat, F):
        """Returns some summary of the current batch.

        Reduces over the batch dimension

        Parameters
        ----------
        feat : tf.Tensor, shape (B, ..., F)
        F : int

        Returns
        -------
        summary : tf.Tensor, shape (..., F')
        F' : int

        """
        bmean = tf.reduce_mean(feat, axis=0)
        bmin = tf.reduce_min(feat, axis=0)
        bmax = tf.reduce_max(feat, axis=0)
        feat = tf.concat([bmean, bmin, bmax], axis=-1)
        F = F * 3
        return feat, F

    def _bwd_weighted_grads(self, cur_grads, weights):
        # cur_grads: (B, N)
        # weights: (B, N, 4)

        # (B, 1, N, 4)
        set_grads, F_p = self._get_node_set_feature_v2(
            cur_grads[:, tf.newaxis, :, tf.newaxis],
            F=1,
        )
        # (B, N)
        weighted_grads = tf.reduce_sum(
            weights * set_grads[:, 0], axis=-1
        )
        return weighted_grads

    def _get_batch_set_feature(self, feat, F):
        """Returns the features extracted based on sets.

        Parameters
        ----------
        feat : tf.Tensor, shape (B, T, N, F)
            `N` is the dimension for the set
        F : int
            The number of channels for the input feature

        Returns
        -------
        set_feat : tf.Tensor, shape (B, T, N, F')
        F' : int
            The number of channels of the output feature

        """
        B = tf.shape(feat)[0]
        T = tf.shape(feat)[1]
        N = tf.shape(feat)[2]

        # (B, TN, F)
        feat = tf.reshape(feat, (B, -1, F))

        # obtain pair-wise feats for nodes
        # (B, 1, TN, F)
        src_feat = feat[:, tf.newaxis, :, :]
        # (1, B, TN, F)
        dst_feat = feat[tf.newaxis, :, :, :]

        TN = T * N

        if self._dist_fn == 'diff':
            # (B, B, TN, F)
            dist = dst_feat - src_feat
            self_feat = feat
        elif self._dist_fn == 'dot':
            # (B, B, TN, F)
            dist = dst_feat * src_feat
            self_feat = feat
        elif self._dist_fn == 'norm_dot':
            n_dst_feat, _ = _safe_normalize(dst_feat, axis=-1)
            n_src_feat, _ = _safe_normalize(src_feat, axis=-1)
            dist = tf.reduce_sum(
                n_dst_feat * n_src_feat, axis=-1, keepdims=True
            )
            self_feat = tf.ones([B, TN, 1])
            F = 1
        elif self._dist_fn == 'concat':
            # (B, B, TN, F*2)
            dist = tf.concat([
                tf.tile(src_feat, [1, B, 1, 1]),
                tf.tile(dst_feat, [B, 1, 1, 1]),
            ], axis=-1)
            # (B, TN, F*2)
            self_feat = tf.concat([feat, feat], axis=-1)
            F = F * 2
        elif self._dist_fn == 'none':
            # (B, B, TN, F)
            dist = tf.tile(dst_feat, [B, 1, 1, 1])
            # (B, TN, F)
            self_feat = feat
        else:
            raise ValueError()

        # Aggregate over node features
        # Create an "other" mask
        mask = tf.ones((B, B)) - tf.eye(B)  # here
        # (B * (B-1), 2)
        to_take = tf.where(mask > 0.5)
        # (B * (B-1), TN, F)
        gathered = tf.gather_nd(dist, to_take)
        # (B, B-1, TN, F)
        other_feat = tf.reshape(gathered, (B, B - 1, TN, F))

        # (B, TN, F)
        other_mean = tf.reduce_mean(other_feat, axis=1)
        other_min = tf.reduce_min(other_feat, axis=1)
        other_max = tf.reduce_max(other_feat, axis=1)

        # put them together
        agg_feats = [self_feat, other_mean, other_min, other_max]
        n_agg_feats = len(agg_feats)

        # (B, TN, F*n_agg_feats)
        all_feats = tf.concat(agg_feats, axis=-1)

        all_feats = tf.reshape(all_feats, (B, T, N, F * n_agg_feats))

        return all_feats, F * n_agg_feats

    def _get_default_grads(self, past_grads, past_acts, cur_acts):
        if self._random_grads_stddev is not None:
            default_grads = tf.random.normal(
                shape=(B, 1, N),
                stddev=self._random_grads_stddev
            )
        elif self._use_nearest_grads:
            # TODO: can look at other batch instances too
            #       which would create a (B, B, T, N) diff
            # TODO: can limit the time window that we look back
            # (B, T, N)
            diff = tf.math.abs(cur_acts[:, tf.newaxis] - past_acts)
            # (B, N)
            closest_idx = tf.math.argmin(diff, axis=1)
            # (B * N, 1)
            closest_idx = tf.reshape(closest_idx, (B * N, 1))
            idx = tf.range(B * N)
            # (B * N, 2)
            closest_idx = tf.concat(
                [closest_idx, idx[..., tf.newaxis]], axis=-1
            )
            # (T, B, N)
            pg = tf.transpose(past_grads, (1, 0, 2))
            # (T, B * N)
            pg = tf.reshape(pg, (T, B * N))
            # (B * N)
            closest_grads = tf.gather_nd(pg, closest_idx)
            default_grads = tf.reshape(closest_grads, (B, 1, N))
        else:
            default_grads = tf.zeros((B, 1, N), dtype=tf.float32)

        return default_grads

    def train_callback(self):
        self._input_mlp.train_callback()
        if self._fwd_output_mlp is not None:
            self._fwd_output_mlp.train_callback()
        if self._bwd_output_mlp is not None:
            self._bwd_output_mlp.train_callback()


def _safe_normalize(tensor, axis, eps=1e-8):
    tensor, norm = tf.linalg.normalize(tensor + eps, axis=axis)
    return tensor, norm

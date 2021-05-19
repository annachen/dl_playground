import tensorflow as tf
import numpy as np
from collections import deque


from dl_playground.networks.utils import (
    activation_function,
    random_init
)


class MaskedLayer:
    def __init__(
        self,
        window_size,
        meta_net=None,
        stop_gradient_from_meta=True,
    ):
        self._meta_net = meta_net
        self._sg_from_meta = stop_gradient_from_meta

        self._past_acts = deque(maxlen=window_size)
        self._past_grads = deque(maxlen=window_size)

        self._last_act = None

    def record_step(self, grads):
        fout = self._last_act.shape[-1]  # this should be static
        last_act = tf.reshape(self._last_act, (-1, fout))
        grads = tf.reshape(grads, (-1, fout))
        if self._sg_from_meta:
            la = tf.stop_gradient(last_act)
            lg = tf.stop_gradient(grads)
        else:
            la = last_act
            lg = grads

        self._past_acts.append(la)
        self._past_grads.append(lg)

    def _prepare_meta_net_inputs(self, cur_acts=None, cur_grads=None):
        batch = {
            'past_acts': tf.stack(list(self._past_acts), axis=1),
            'past_grads': tf.stack(list(self._past_grads), axis=1),
            'cur_acts': cur_acts,
            'cur_grads': cur_grads,
        }
        return batch

    def get_forward_mask(self, cur_acts, meta_training=None):
        assert self._meta_net is not None

        inp_shape = tf.shape(cur_acts)

        if self._sg_from_meta:
            cur_acts = tf.stop_gradient(cur_acts)

        fout = cur_acts.shape[-1]
        cur_acts = tf.reshape(cur_acts, (-1, fout))

        if len(self._past_acts) == 0:
            mask = self._meta_net.first_forward(
                cur_acts, training=meta_training
            )
        else:
            meta_batch = self._prepare_meta_net_inputs(cur_acts)
            mask = self._meta_net.forward(
                meta_batch, training=meta_training,
            )

        mask = tf.reshape(mask, inp_shape)

        return mask

    def get_backward_mask(
        self,
        cur_grads=None,
        meta_training=None
    ):
        assert self._meta_net is not None

        if self._sg_from_meta:
            # backward pass in DualRNN uses masked act as last input
            cur_acts = tf.stop_gradient(self._last_act)
        else:
            cur_acts = self._last_act

        inp_shape = tf.shape(cur_acts)
        fout = cur_acts.shape[-1]
        cur_acts = tf.reshape(cur_acts, (-1, fout))

        if cur_grads is not None and self._sg_from_meta:
            cur_grads = tf.stop_gradient(cur_grads)

        if cur_grads is not None:
            cur_grads = tf.reshape(cur_grads, (-1, fout))

        if len(self._past_acts) == 0:
            meta_batch = {
                'cur_acts': cur_acts,
                'cur_grads': cur_grads,
            }
            mask = self._meta_net.first_backward(
                meta_batch, training=meta_training
            )
        else:
            meta_batch = self._prepare_meta_net_inputs(
                cur_acts=cur_acts,
                cur_grads=cur_grads,
            )
            mask = self._meta_net.backward(
                meta_batch, training=meta_training
            )
        mask = tf.reshape(mask, inp_shape)
        return mask


class MaskedDense(MaskedLayer):
    """A dense layer that supports masking.

    This layer supports:
    1) masking with forward meta-net mask
    2) masking with backward meta-net mask
    3) pass gradients through weight updates
    4) be stacked together as a larger network

    Parameters
    ----------
    fin : int
    fout : int
    activation : str
    window_size : int
        The length of history to use to feed into meta-net
    act_fn_kwargs : dict | None
    use_bias : bool
    stop_gradient_from_meta : bool
        Whether to stop the gradients coming back from the masks from
        the meta-net. In reality the stop_gradient op is put where
        the activation and gradients are passed to meta-net, as the
        meta-net do want gradient through the masks.
    stop_gradient_at_update : bool
        Whether to stop the gradients at weight updates. If so, the
        weight updates are
          w_new = sg(w_old) - grads * lr
    mask_after_activation : bool
        Whether the masking should happen before or after activation
        function
    use_fwd_mask : bool
        Whether to use the forward mask
    use_bwd_mask : bool
        Whether to use the backward mask
    bwd_return_grads : bool
        Whether meta-net returns a mask or gradients in bwd pass
    meta_net : LayerCompetition | None

    """
    def __init__(
        self,
        fin,
        fout,
        activation,
        window_size,
        act_fn_kwargs=None,
        use_bias=True,
        stop_gradient_from_meta=True,
        stop_gradient_at_update=False,
        mask_after_activation=True,
        use_fwd_mask=False,
        use_bwd_mask=False,
        bwd_return_grads=False,
        meta_net=None,
    ):
        super(MaskedDense, self).__init__(
            meta_net=meta_net,
            window_size=window_size,
            stop_gradient_from_meta=stop_gradient_from_meta,
        )

        self._fin = fin
        self._fout = fout
        self._act_fn = activation_function(activation, act_fn_kwargs)
        self._use_bias = use_bias
        self._sg_at_update = stop_gradient_at_update
        self._sg_from_meta = stop_gradient_from_meta
        self._mask_after_act = mask_after_activation
        self._window_size = window_size
        self._use_fwd_mask = use_fwd_mask
        self._use_bwd_mask = use_bwd_mask
        self._bwd_return_grads = bwd_return_grads

        self._meta_net = meta_net

        self._last_x = None
        self._last_act = None
        self._last_pre_act = None
        self._last_pre_mask = None
        self._last_mask = None

        self._past_acts = deque(maxlen=window_size)
        self._past_grads = deque(maxlen=window_size)

        self.build((fin, fout))

    def build(self, input_shape):
        # input_shape : (int, int) input_channels, output_channels

        init_w = random_init(
            init_fn='GlorotNormal',
            init_kwargs=None,
            fin=input_shape[0],
            fout=input_shape[1],
        )
        self._w_var = tf.Variable(init_w, dtype=tf.float32)
        self._w = tf.identity(self._w_var)

        if self._use_bias:
            self._b_var = tf.Variable(
                np.zeros(self._fout), dtype=tf.float32
            )
            self._b = tf.identity(self._b_var)

    def variables_updated(self):
        """Callback function when the variables are modified.

        This copies the new values of the variables to the weight
        tensors. This function should be called when variables are
        restored via checkpoints.

        """
        self._w = tf.identity(self._w_var)
        if self._use_bias:
            self._b = tf.identity(self._b_var)

    def call(self, batch, training=None, meta_training=None):
        # batch : (B, fin)
        # (B, fout)
        pre_act = tf.matmul(batch, self._w)
        if self._use_bias:
            pre_act = pre_act + self._b[tf.newaxis]

        if not self._mask_after_act:
            pre_mask = pre_act
            if self._meta_net is not None and self._use_fwd_mask:
                mask = self.get_forward_mask(
                    pre_mask, meta_training=meta_training
                )
                pre_act = pre_mask * mask
            else:
                mask = None

        out = self._act_fn(pre_act)

        if self._mask_after_act:
            pre_mask = out
            if self._meta_net is not None and self._use_fwd_mask:
                mask = self.get_forward_mask(
                    pre_mask, meta_training=meta_training
                )
                out = pre_mask * mask
            else:
                mask = None

        # store the last activations
        self._last_x = batch
        self._last_act = out
        self._last_pre_act = pre_act
        self._last_pre_mask = pre_mask
        self._last_mask = mask

        return out

    @property
    def weights(self):
        if self._use_bias:
            return [self._w, self._b]
        else:
            return [self._w]

    @property
    def variables(self):
        if self._use_bias:
            return [self._w_var, self._b_var]
        else:
            return [self._w_var]

    def set_weights(self, new_weights):
        self._w = new_weights[0]
        if self._use_bias:
            self._b = new_weights[1]

    def update_graph(self):
        self._w_var.assign(self._w)
        self._w = tf.identity(self._w_var)
        if self._use_bias:
            self._b_var.assign(self._b)
            self._b = tf.identity(self._b_var)

    def calculate_gradients(
        self,
        tape,
        sum_loss,
        cur_grads=None,
        prev_mask=None,
    ):
        if self._use_bwd_mask:
            # (B, F_out)
            mask = self.get_backward_mask(
                cur_grads=cur_grads,
            )

            if prev_mask is not None:
                mask = mask * prev_mask

            if self._bwd_return_grads:
                # (B, F_out)
                grads = mask
                # dL/dw0 = dL/dact * dact/dpre_act * dpre_act/dw0
                if self._mask_after_act:
                    # (B, F_out)
                    dact_dpre_act = tape.gradient(
                        self._last_act, self._last_pre_act
                    )
                else:
                    dact_dpreact = tape.gradient(
                        self._last_act, self._last_pre_mask
                    )
                # (B, F_in)
                x0 = self._last_x

                # (B, F_in, F_out)
                dw0 = (
                    grads[:, tf.newaxis] *
                    dact_dpreact[:, tf.newaxis] *
                    x0[..., tf.newaxis]
                )
                dw0 = tf.reduce_mean(dw0, axis=0)
                ret_grads = [dw0]

                if self._use_bias:
                    # dL/db0 = dL/dact * dact/dpre_act * 1
                    # (B, F_out)
                    db0 = grads * dact_dpreact
                    db0 = tf.reduce_mean(db0, axis=0)
                    ret_grads.append(db0)

                return ret_grads, grads, None

            # dL/dw0 = dL/dpre_act * dpre_act/dw0
            #        = dL/dpre_act * x0
            if self._mask_after_act:
                # (B, F_out)
                d_pre_act = tape.gradient(
                    sum_loss, self._last_pre_act
                )
            else:
                # (B, F_out)
                d_pre_act = tape.gradient(
                    sum_loss, self._last_pre_mask
                )

            # (B, F_in)
            x0 = self._last_x

            # (B, F_in, F_out)
            dw = x0[..., tf.newaxis] * d_pre_act[:, tf.newaxis]

            masked_dw = mask[:, tf.newaxis] * dw
            # (F_in, F_out)
            masked_dw = tf.reduce_mean(masked_dw, axis=0)

            grads = [masked_dw]

            if self._use_bias:
                # dL/db0 = dL/dpre_act * dpre_act/db0
                #        = dL/dpre_act * 1
                db = d_pre_act

                masked_db = mask * db
                # (F_out)
                masked_db = tf.reduce_mean(masked_db, axis=0)

                grads.append(masked_db)

            # calculate mask to pass to previous layer
            # (B, Fin)
            prev_mask = tf.matmul(mask, tf.transpose(self._w))

            return grads, mask, prev_mask

        else:
            B = tf.cast(tf.shape(self._last_act)[0], tf.float32)
            grads = tape.gradient(sum_loss, self.weights)
            grads = [g / B for g in grads]
            return grads, None, None

    def update_weights(self, gradients, learning_rate):
        if self._sg_at_update:
            old_w = tf.stop_gradient(self._w)
        else:
            old_w = self._w
        self._w = old_w - gradients[0] * learning_rate

        if self._use_bias:
            if self._sg_at_update:
                old_b = tf.stop_gradient(self._b)
            else:
                old_b = self._b
            self._b = old_b - gradients[1] * learning_rate


class MaskedConv2D(MaskedLayer):
    """A conv2d layer that supports masking.

    This layer supports:
    1) masking with forward meta-net mask
    2) masking with backward meta-net mask
    3) pass gradients through weight updates
    4) be stacked together as a larger network

    Parameters
    ----------
    kernel_size : int
    fin : int
    fout : int
    activation : str
    window_size : int
        The length of history to use to feed into meta-net
    act_fn_kwargs : dict | None
    use_bias : bool
    stop_gradient_from_meta : bool
        Whether to stop the gradients coming back from the masks from
        the meta-net. In reality the stop_gradient op is put where
        the activation and gradients are passed to meta-net, as the
        meta-net do want gradient through the masks.
    stop_gradient_at_update : bool
        Whether to stop the gradients at weight updates. If so, the
        weight updates are
          w_new = sg(w_old) - grads * lr
    mask_after_activation : bool
        Whether the masking should happen before or after activation
        function
    use_fwd_mask : bool
        Whether to use the forward mask
    use_bwd_mask : bool
        Whether to use the backward mask
    bwd_return_grads : bool
    meta_net : LayerCompetition | None

    """
    def __init__(
        self,
        kernel_size,
        fin,
        fout,
        stride,
        padding,
        activation,
        window_size,
        act_fn_kwargs=None,
        use_bias=True,
        stop_gradient_from_meta=True,
        stop_gradient_at_update=False,
        mask_after_activation=True,
        use_fwd_mask=False,
        use_bwd_mask=False,
        bwd_return_grads=False,
        meta_net=None,
    ):
        super(MaskedConv2D, self).__init__(
            meta_net=meta_net,
            window_size=window_size,
            stop_gradient_from_meta=stop_gradient_from_meta,
        )

        self._ks = kernel_size
        self._fin = fin
        self._fout = fout
        self._stride = stride
        self._padding = padding
        self._act_fn = activation_function(activation, act_fn_kwargs)
        self._use_bias = use_bias
        self._sg_at_update = stop_gradient_at_update
        self._sg_from_meta = stop_gradient_from_meta
        self._mask_after_act = mask_after_activation
        self._window_size = window_size
        self._use_fwd_mask = use_fwd_mask
        self._use_bwd_mask = use_bwd_mask
        self._bwd_return_grads = bwd_return_grads

        self._meta_net = meta_net

        self._last_x = None
        self._last_act = None
        self._last_pre_act = None
        self._last_pre_mask = None
        self._last_mask = None

        self._past_acts = deque(maxlen=window_size)
        self._past_grads = deque(maxlen=window_size)

        self.build()

    def build(self):
        init_w = random_init(
            init_fn='GlorotNormal',
            init_kwargs=None,
            fin=self._fin * self._ks * self._ks,
            fout=self._fout,
        )
        init_w = tf.reshape(
            init_w, (self._ks, self._ks, self._fin, self._fout)
        )
        self._w_var = tf.Variable(init_w, dtype=tf.float32)
        self._w = tf.identity(self._w_var)

        if self._use_bias:
            self._b_var = tf.Variable(
                np.zeros(self._fout), dtype=tf.float32
            )
            self._b = tf.identity(self._b_var)

    def variables_updated(self):
        """Callback function when the variables are modified.

        This copies the new values of the variables to the weight
        tensors. This function should be called when variables are
        restored via checkpoints.

        """
        self._w = tf.identity(self._w_var)
        if self._use_bias:
            self._b = tf.identity(self._b_var)

    def call(self, batch, training=None, meta_training=None):
        # batch : (B, H, W, fin)
        # (B, H', W', fout)
        pre_act = tf.nn.conv2d(
            input=batch,
            filters=self._w,
            strides=self._stride,
            padding=self._padding,
        )
        if self._use_bias:
            pre_act = pre_act + self._b[
                tf.newaxis, tf.newaxis, tf.newaxis
            ]

        B = tf.shape(pre_act)[0]
        Hp = tf.shape(pre_act)[1]
        Wp = tf.shape(pre_act)[2]

        if not self._mask_after_act:
            pre_mask = pre_act
            if self._meta_net is not None and self._use_fwd_mask:
                mask = self.get_forward_mask(
                    pre_mask,
                    meta_training=meta_training
                )
                pre_act = pre_mask * mask
            else:
                mask = None

        out = self._act_fn(pre_act)

        if self._mask_after_act:
            pre_mask = out
            if self._meta_net is not None and self._use_fwd_mask:
                mask = self.get_forward_mask(
                    pre_mask,
                    meta_training=meta_training
                )
                out = pre_mask * mask
            else:
                mask = None

        # store the last activations
        self._last_x = batch
        self._last_act = out
        self._last_pre_act = pre_act
        self._last_pre_mask = pre_mask
        self._last_mask = mask

        return out

    @property
    def weights(self):
        if self._use_bias:
            return [self._w, self._b]
        else:
            return [self._w]

    @property
    def variables(self):
        if self._use_bias:
            return [self._w_var, self._b_var]
        else:
            return [self._w_var]

    def set_weights(self, new_weights):
        self._w = new_weights[0]
        if self._use_bias:
            self._b = new_weights[1]

    def update_graph(self):
        self._w_var.assign(self._w)
        self._w = tf.identity(self._w_var)
        if self._use_bias:
            self._b_var.assign(self._b)
            self._b = tf.identity(self._b_var)

    def calculate_gradients(
        self,
        tape,
        sum_loss,
        cur_grads=None,
        prev_mask=None
    ):
        if self._use_bwd_mask:
            # (B, H', W', F_out)
            mask = self.get_backward_mask(cur_grads=cur_grads)

            if prev_mask is not None:
                mask = mask * prev_mask

            if self._bwd_return_grads:
                # (B, H', W', F_out)
                grads = mask
                # dL/dw0 = dL/dact * dact/dpre_act * dpre_act/dw0
                if self._mask_after_act:
                    # (B, H', W', F_out)
                    dact_dpre_act = tape.gradient(
                        self._last_act, self._last_pre_act
                    )
                else:
                    dact_dpreact = tape.gradient(
                        self._last_act, self._last_pre_mask
                    )

                # (B, H', W', ks*ks*F_in)
                x0_patches = tf.image.extract_patches(
                    images=self._last_x,
                    sizes=[1, self._ks, self._ks, 1],
                    strides=[1, self._stride, self._stride, 1],
                    padding=self._padding,
                    rates=[1, 1, 1, 1],
                )

                # (B, H', W', ks*ks*F_in, F_out)
                dw0 = (
                    grads[:, :, :, tf.newaxis] *
                    dact_dpreact[:, :, :, tf.newaxis] *
                    x0_patches[..., tf.newaxis]
                )
                # (B, ks*ks*F_in, F_out)
                dw0 = tf.reduce_sum(dw0, axis=(1, 2))
                dw0 = tf.reduce_mean(dw0, axis=0)
                dw0 = tf.reshape(
                    dw0, (self._ks, self._ks, self._fin, self._fout)
                )
                ret_grads = [dw0]

                if self._use_bias:
                    # dL/db0 = dL/dact * dact/dpre_act * 1
                    # (B, H', W', F_out)
                    db0 = grads * dact_dpreact
                    db0 = tf.reduce_sum(db0, axis=(1, 2))
                    db0 = tf.reduce_mean(db0, axis=0)

                    ret_grads.append(db0)

                return ret_grads, grads, None

            # dL/dw0 = dL/dpre_act * dpre_act/dw0
            # dpre_act(b, y, x) / dw0 = x0 (local patch)
            if self._mask_after_act:
                # (B, H', W', F_out)
                d_pre_act = tape.gradient(
                    sum_loss, self._last_pre_act
                )
            else:
                # (B, H', W', F_out)
                d_pre_act = tape.gradient(
                    sum_loss, self._last_pre_mask
                )

            # dpre_act/dw0 at each location
            # (B, H', W', ks*ks*F_in)
            x0_patches = tf.image.extract_patches(
                images=self._last_x,
                sizes=[1, self._ks, self._ks, 1],
                strides=[1, self._stride, self._stride, 1],
                padding=self._padding,
                rates=[1, 1, 1, 1],
            )

            # (B, H', W', ks*ks*F_in, F_out)
            dw = (
                x0_patches[..., tf.newaxis] *
                d_pre_act[:, :, :, tf.newaxis]
            )
            masked_dw = mask[:, :, :, tf.newaxis] * dw
            #test_dw = tape.gradient(sum_loss, self._w)
            #test_dw2 = tf.reduce_sum(dw, axis=(0, 1, 2))
            #print(test_dw)
            #print(test_dw2)

            # (B, ks*ks*F_in, F_out)
            masked_dw = tf.reduce_sum(masked_dw, axis=(1, 2))
            # (ks*ks*F_in, F_out)
            masked_dw = tf.reduce_mean(masked_dw, axis=0)
            # (ks, ks, F_in, F_out)
            masked_dw = tf.reshape(
                masked_dw, (self._ks, self._ks, self._fin, self._fout)
            )

            grads = [masked_dw]

            if self._use_bias:
                # dL/db0 = dL/dpre_act * dpre_act/db0
                #        = dL/dpre_act * 1
                # (B, H', W', F_out)
                db = d_pre_act

                masked_db = mask * db
                # (B, F_out)
                masked_db = tf.reduce_sum(masked_db, axis=(1, 2))
                # (F_out)
                masked_db = tf.reduce_mean(masked_db, axis=0)

                grads.append(masked_db)

            # calculate mask to pass to previous layer
            # TODO: implement this
            # I have a feeling that this is gonna be a convolution
            #prev_mask = tf.matmul(mask, tf.transpose(self._w))

            return grads, mask, None

        else:
            B = tf.cast(tf.shape(self._last_act)[0], tf.float32)
            grads = tape.gradient(sum_loss, self.weights)
            grads = [g / B for g in grads]
            return grads, None, None

    def update_weights(self, gradients, learning_rate):
        if self._sg_at_update:
            old_w = tf.stop_gradient(self._w)
        else:
            old_w = self._w
        self._w = old_w - gradients[0] * learning_rate

        if self._use_bias:
            if self._sg_at_update:
                old_b = tf.stop_gradient(self._b)
            else:
                old_b = self._b
            self._b = old_b - gradients[1] * learning_rate

import tensorflow as tf

def accuracy_metric(prediction, label, is_one_hot=False):
    """Calculates classification accuracy.

    Parameters
    ----------
    prediction : tf.Tensor, shape (B, n_classes)
    label : tf.Tensor
        If `is_one_hot` is True, shape (B, n_classes)
        Otherwise, shape (B,)
    is_one_hot : bool
        Whether `label` is in one_hot vector format.

    Returns
    -------
    is_correct : tf.Tensor, shape (B,)

    """
    if is_one_hot:
        label = tf.argmax(label, axis=-1)

    label = tf.cast(label, tf.int32)
    pred = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

    return tf.cast(tf.math.equal(label, pred), tf.float32)

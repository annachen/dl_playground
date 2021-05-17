import tensorflow as tf


OPTIMIZER_CLS = {
    'adam': tf.keras.optimizers.Adam,
    'sgd': tf.keras.optimizers.SGD,
}


def create_optimizer(config):
    learning_rate = config['learning_rate']
    if type(learning_rate) == dict:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            **config['learning_rate']
        )
    else:
        lr = learning_rate
    optimizer_name = config['optimizer']
    optimizer = OPTIMIZER_CLS[optimizer_name](
        learning_rate=lr
    )
    return optimizer

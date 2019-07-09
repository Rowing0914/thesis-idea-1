import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def create_model(kernel_fn):
    dtype = np.float64
    num_inducing_points = 40
    loss = lambda y, rv_y: rv_y.variational_loss(y)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[658], dtype=dtype),
        # tf.keras.layers.Dense(379, kernel_initializer='ones', use_bias=False),
        # tf.keras.layers.Dense(169, kernel_initializer='ones', use_bias=False),
        tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
        tfp.layers.VariationalGaussianProcess(
            num_inducing_points=num_inducing_points,
            kernel_provider=kernel_fn(dtype=dtype),
            event_shape=[1],
            # inducing_index_points_initializer=tf.constant_initializer(
            # 	np.linspace(*x_range, num=num_inducing_points,
            # 				dtype=x.dtype)[..., np.newaxis]),
            unconstrained_observation_noise_variance_initializer=(
                tf.constant_initializer(np.array(0.54).astype(dtype))),
        ),
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
    return model

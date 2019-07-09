import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils.common import flatten_weight


def create_model(weights, kernel_fn):
    """
    Instantiate the GP model corresponding to the weight matrix used in the policy network

    Usage:
        ```python
        agent = DQN(...)
        init_state = env.reset() # reset
        agent.predict(init_state) # burn the format of the input matrix to get the weight matrices!!
        gp_model = create_model(weights=agent.main_model.get_weights(), kernel_fn=RBFKernelFn)
        ```
    """
    input_shape = flatten_weight(weights).shape[0]
    print("GP Model: Input Shape is {}".format(input_shape))
    dtype = np.float64
    num_inducing_points = 40
    loss = lambda y, rv_y: rv_y.variational_loss(y)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[input_shape], dtype=dtype),
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

# class Model(tf.keras.Model):
# 	""" Variational Gaussian Process Network """
#
# 	def __init__(self, num_inducing_points, kernel, dtype=np.float64):
# 		# def __init__(self, _max, _min, num_inducing_points, kernel, dtype=np.float64): # if we need inducing_index then, open this!
# 		super(Model, self).__init__()
# 		self.dense = tf.keras.layers.Dense(1, activation='linear')
# 		self.gp = tfp.layers.VariationalGaussianProcess(
# 			num_inducing_points=num_inducing_points,
# 			kernel_provider=kernel(dtype=dtype),
# 			event_shape=[1],
# 			# inducing_index_points_initializer=tf.constant_initializer(
# 			# 	np.linspace(_min, _max, num=num_inducing_points, dtype=dtype)[..., np.newaxis]),
# 			unconstrained_observation_noise_variance_initializer=(
# 				tf.constant_initializer(np.array(0.54).astype(dtype))),
# 		)
#
# 	# @tf.contrib.eager.defun(autograph=False)
# 	def call(self, inputs):
# 		x = self.dense(inputs)
# 		return self.gp(x)
#
# gp_model = Model(num_inducing_points=40, kernel=RBFKernelFn)
# optimiser = tf.train.AdamOptimizer()
# # https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
# loss_func = lambda y, rv_y: rv_y.variational_loss(y)  # temp loss func
#
#
# def update(model, x, y):
# 	""" Temp function to update the weights of GP net """
# 	with tf.GradientTape() as tape:
# 		pred = model(x)
# 		print(pred)
# 		loss = loss_func(y, pred)
# 	grads = tape.gradient(loss, model.trainable_weights)  # get gradients
# 	optimiser.apply_gradients(zip(grads, model.trainable_weights))  # apply gradients to the network
# 	return model, loss

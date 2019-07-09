import tensorflow as tf
import tensorflow_probability as tfp


class RBFKernelFn(tf.keras.layers.Layer):
	# https://medium.com/tensorflow/regression-with-probabilistic-layers-in-tensorflow-probability-e46ff5d37baf
	def __init__(self, **kwargs):
		super(RBFKernelFn, self).__init__(**kwargs)
		dtype = kwargs.get('dtype', None)

		self._amplitude = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='amplitude')

		self._length_scale = self.add_variable(
			initializer=tf.constant_initializer(0),
			dtype=dtype,
			name='length_scale')

	def call(self, x):
		# Never called -- this is just a layer so it can hold variables
		# in a way Keras understands.
		return x

	@property
	def kernel(self):
		return tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
			amplitude=tf.nn.softplus(0.1 * self._amplitude),
			length_scale=tf.nn.softplus(5. * self._length_scale)
		)


class Model(tf.keras.Model):
	""" Variational Gaussian Process Network """

	def __init__(self, num_inducing_points, kernel, dtype=np.float64):
		# def __init__(self, _max, _min, num_inducing_points, kernel, dtype=np.float64): # if we need inducing_index then, open this!
		super(Model, self).__init__()
		self.dense = tf.keras.layers.Dense(1, activation='linear')
		self.gp = tfp.layers.VariationalGaussianProcess(
			num_inducing_points=num_inducing_points,
			kernel_provider=kernel(dtype=dtype),
			event_shape=[1],
			# inducing_index_points_initializer=tf.constant_initializer(
			# 	np.linspace(_min, _max, num=num_inducing_points, dtype=dtype)[..., np.newaxis]),
			unconstrained_observation_noise_variance_initializer=(
				tf.constant_initializer(np.array(0.54).astype(dtype))),
		)

	# @tf.contrib.eager.defun(autograph=False)
	def call(self, inputs):
		x = self.dense(inputs)
		return self.gp(x)

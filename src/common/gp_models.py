import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from src.common.utils import flatten_weight
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

tfd = tfp.distributions

def create_variational_GP_model(weights, kernel_fn):
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
    dtype = np.float32
    num_inducing_points = 40
    loss = lambda y, rv_y: rv_y.variational_loss(y)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[input_shape], dtype=dtype),
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


def temp(kernel_fn):
    class Model(tf.keras.Model):
        """ Variational Gaussian Process Network """

        def __init__(self, num_inducing_points=40): # default value on the article
            super(Model, self).__init__()
            self.dense = tf.keras.layers.Dense(1, activation='linear')
            self.gp = tfp.layers.VariationalGaussianProcess(
                num_inducing_points=num_inducing_points,
                kernel_provider=kernel_fn(dtype=np.float64),
                event_shape=[1],
                # inducing_index_points_initializer=tf.constant_initializer(
                # 	np.linspace(*x_range, num=num_inducing_points,
                # 				dtype=x.dtype)[..., np.newaxis]),
                unconstrained_observation_noise_variance_initializer=(
                    tf.constant_initializer(np.array(0.54).astype(np.float64))),
            )

        # @tf.contrib.eager.defun(autograph=False)
        def call(self, inputs):
            x = self.dense(inputs)
            return self.gp(x)

    def update(model, optimiser, x, y):
        """ Temp function to update the weights of Bayes net """
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = pred.variational_loss(y)
        grads = tape.gradient(loss, model.trainable_weights)  # get gradients
        optimiser.apply_gradients(zip(grads, model.trainable_weights))  # apply gradients to the network
        return model, loss

    return Model(), update


def sklearn_GP_model(random_state=0):
    """ SkLearn implementation of GPR """
    KERNEL = DotProduct() + WhiteKernel()
    return GaussianProcessRegressor(kernel=KERNEL, random_state=random_state)


def create_TF_GP_model(weights, kernel_fn):
    """
    TF implementation of GPR is utterly useless
    they require us to set the index points in advance, we cannot apply the model to unknown data.
    how stupid TF team is....
    """
    input_shape = flatten_weight(weights).shape[0]
    print("GP Model: Input Shape is {}".format(input_shape))
    dtype = np.float64
    loss = lambda y, rv_y: rv_y.variational_loss(y)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[input_shape], dtype=dtype),
        tfp.distributions.GaussianProcess(
            kernel=kernel_fn
        ),
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
    return model


def create_bayes_net(batch_size=32):
    class Model(tf.keras.Model):
        """ Variational Gaussian Process Network """

        def __init__(self):
            super(Model, self).__init__()
            self.dense = tfp.layers.DenseFlipout(84, activation=tf.nn.relu) # TODO: make it changeable from upper layer
            self.mean = tfp.layers.DenseFlipout(1)
            self.std = tf.Variable(10.0, dtype=tf.float64) # TODO: how can we back-prop the error to STD?

        # @tf.contrib.eager.defun(autograph=False)
        def call(self, inputs):
            x = self.dense(inputs)
            mean_ = self.mean(x)
            return tfd.Normal(loc=mean_, scale=self.std)  # assumed to follow Gaussian Distribution

    def update(model, optimiser, x, y, num_update=10):
        """ Temp function to update the weights of Bayes net """
        for _ in range(num_update):
            indices = np.random.randint(0, x.shape[0], size=batch_size)
            x = x[indices, ...]
            with tf.GradientTape() as tape:
                pred = model(x)
                neg_log_likelihood = -tf.reduce_mean(input_tensor=pred.log_prob(y)) # TODO: Is NLL an appropriate loss func?
            grads = tape.gradient(neg_log_likelihood, model.trainable_weights)  # get gradients
            optimiser.apply_gradients(zip(grads, model.trainable_weights))  # apply gradients to the network
        return model, neg_log_likelihood

    return Model(), update

from tf_rl.common.utils import eager_setup
from src.common.gp_models import *
from src.common import RBFKernelFn

eager_setup()

temp_w = np.random.rand(3, 3, 3)
temp_w_size = len(temp_w.flatten().tolist())
num_episodes = num_sample = batch_size = num_epochs = 10

def data_generator():
    weights = np.random.rand(100, temp_w_size).astype(dtype=np.float64)
    scores = np.random.randint(0, 10, size=100)*1.0
    new_weight = np.random.rand(1, temp_w_size).astype(dtype=np.float64)
    return weights, scores, new_weight

def train_loop(model, update, optimiser):
    for episode in range(num_episodes):
        weights, scores, new_weight = data_generator()

        pred = model(weights).sample(num_sample).numpy().mean()
        update(model, optimiser, weights, scores)
        sample_ = model(new_weight).sample(num_sample).numpy().mean()
        print(sample_, pred)

def sk_gp_loop(model):
    for episode in range(num_episodes):
        weights, scores, new_weight = data_generator()

        model.fit(weights, scores)
        sample_ = model.sample_y(new_weight, n_samples=num_sample).mean()
        print(sample_)

if __name__ == '__main__':
    # model, update = create_bayes_net()
    # optimiser = tf.compat.v1.train.AdamOptimizer()
    # train_loop(model, update, optimiser)
    #
    # model = sklearn_GP_model()
    # sk_gp_loop(model)

    model, update = temp(kernel_fn=RBFKernelFn)
    optimiser = tf.compat.v1.train.AdamOptimizer()
    train_loop(model, update, optimiser)
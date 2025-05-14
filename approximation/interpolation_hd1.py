import sys

sys.path.append('../')
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import time
import optax
from jax.nn import gelu, silu, tanh
from jax.lax import scan, stop_gradient
from jax import random, jit, vmap, grad
import os
import scipy
import matplotlib.pyplot as plt
import argparse

from data import get_data
from networks import get_network
from utils import normalization
from numpy import pi

import argparse

parser = argparse.ArgumentParser(description="SincKAN")
parser.add_argument("--mode", type=str, default='train', help="mode of the network, "
                                                              "train: start training, eval: evaluation")
parser.add_argument("--datatype", type=str, default='scaling', help="type of data")
parser.add_argument("--npoints", type=int, default=5000, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=1000, help="the number of testing dataset")
parser.add_argument("--ntrain", type=int, default=1000, help="the number of training dataset for each epochs")
parser.add_argument("--dim", type=int, default=10, help="dim of the problem")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=5000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--activation", type=str, default='tanh', help='the activation function')
parser.add_argument("--interval", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, 1: add normalization")
parser.add_argument("--network", type=str, default="mlp", help="type of network")
parser.add_argument("--kanshape", type=str, default="16", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=100, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=6, help='length of k for sinckan')
parser.add_argument("--init_h", type=float, default=2.0, help='initial value of h')
parser.add_argument("--device", type=int, default=7, help="cuda number")
parser.add_argument("--decay", type=str, default='inverse', help='exponent for h')
parser.add_argument("--skip", type=int, default=0, help='1: use skip connection for sinckan')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--initialization", type=str, default='Xavier', help='random initialization of the parameters')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

class training_points():
    def __init__(self, dim, interval=(-1,1)):
        self.dim = dim
        self.points = jnp.linspace(interval[0], interval[1], 20000)

    def sample(self, num, key):
        keys = random.split(key, self.dim)
        points = jnp.concatenate([random.choice(key, self.points, shape=(num, 1), replace=True) for key in keys], -1)
        return points


def net(model, frozen_para, *x):
    return model(jnp.stack(*x), frozen_para)[0]

def compute_loss(model, ob_x, frozen_para, dim):
    output = vmap(net, (None, None, 0))(model, frozen_para, ob_x[:, :dim])  # vmap applies net to a batch of inputs simultaneously, efficient for batch processing in GPUs, (None, 0, None) specifies no action on model and frozen_para, only act on ob_x[:, :n_points]

    return 100 * ((output - ob_x[:, dim]) ** 2).mean()
# if output and ob_x[:,2] are both batches of vectors, mean() will take average over their square difference, otherwise it is just the same as not having mean()


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss) #eqx library provides a function to filter trainable parameters from nontrainable parameters, this computes compute_loss and its grad wrt trainable parameters


@eqx.filter_jit
def make_step(model, ob_x, frozen_para, optim, opt_state, dim):
    loss, grads = compute_loss_and_grads(model, ob_x, frozen_para, dim)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array)) #updates the trainable parameters along the grads, opt_state is the current optimizer state containing momentum term or learning rate. updates is the change need to be added to the param, opt_stat is the latest optimizer state
    model = eqx.apply_updates(model, updates) #updating model parameters by adding the changes contained in updates
    return loss, model, opt_state


def train(key):
    # get hyperparameters
    dim = args.dim
    ntest = args.ntest
    N_train = args.ntrain
    N_epochs = args.epochs
    learning_rate = args.lr
    ite = args.ite

    # Generate sampled data
    keys = random.split(key, 3)
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_in_set = training_points(dim=dim, interval=interval)
    x_train = x_in_set.sample(num=N_train, key=keys[0])
    x_test = x_in_set.sample(num=ntest, key=keys[1])
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    y_target = y_train.copy()
    # Add noise
    if args.noise == 1:
        sigma = 0.01
        y_train += np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test)

    normalizer = normalization(interval, dim, args.normalization)

    input_dim = dim
    output_dim = 1

    # Choosing the model
    keys = random.split(key, 3)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # parameters of optimizer
    N_drop = 10000
    gamma = 0.95
    sc = optax.exponential_decay(learning_rate, N_drop, gamma) #produce decayed_value = init_value * (decay_rate ** rate_factor) rate_factor = ((count - transition_begin) / transition_steps)
    optim = optax.adam(learning_rate=sc) #adam optimizer with exponential decay learning rate

    #Piecewise learning rate schedule
    # lr_schedule = optax.join_schedules(
    #     schedules=[
    #         # First phase: decay rate 0.99
    #         optax.exponential_decay(
    #             init_value=learning_rate,
    #             transition_steps=30000,
    #             decay_rate=0.99
    #         ),
    #         # Second phase: decay rate 0.65
    #         optax.exponential_decay(
    #             init_value=learning_rate * (0.99 ** 3),  # Match last LR of first phase
    #             transition_steps=40000,  # Ends at iteration 70,000
    #             decay_rate=0.65
    #         ),
    #         # Third phase: decay rate 0.99 again
    #         optax.exponential_decay(
    #             init_value=learning_rate * (0.99 ** 3) * (0.65 ** 4),  # Match last LR of second phase
    #             transition_steps=30000,
    #             decay_rate=1.2
    #         )
    #     ],
    #     boundaries=[30000, 70000]  # Boundaries for transitions
    # )
    #
    # # Adam optimizer with the custom schedule
    # optim = optax.adam(learning_rate=lr_schedule)
    #
    opt_state = optim.init(eqx.filter(model, eqx.is_array))  # extract the model trainable parameters, then initialize them

    keys = random.split(keys[-1], 100) #take the last element in keys and split it into two new independent keys
    ob_x = jnp.concatenate([x_train, y_train.reshape(-1,1)],-1)

    history = []
    T = []
    relative_errors = []
    mse_errors = []
    for j in range(ite * N_epochs):
        T1 = time.time()
        loss, model, opt_state = make_step(model, ob_x, frozen_para, optim, opt_state, dim) #for each j in range, make_step will train the model using training points and labels provided in ob_x
        T2 = time.time()
        T.append(T2 - T1)
        history.append(loss.item())
        if j % N_epochs == 0: # for every N_epochs epochs, resample the training points and labels, evaluate the trained model on x_test, record and print the relative error and mse error
            keys = random.split(keys[-1], 3)
            x_train = x_in_set.sample(N_train, keys[0])
            y_train = generate_data(x_train)
            #Add noise
            if args.noise == 1:
                sigma = 0.01
                y_train += np.random.normal(0, sigma, y_train.shape)
            ob_x = jnp.concatenate([x_train, y_train.reshape(-1, 1)], -1)
            y_pred = vmap(net, (None, None, 0))(model, frozen_para, x_test)
            mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
            relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(
                y_test.flatten())
            relative_errors.append(relative_error)
            mse_errors.append(mse_error)
            print(f'ite:{j},testing mse:{mse_error:.2e},relative:{relative_error:.2e}')

    # eval: the previous for loop only train ite*N_epochs - 1 time, need to eval the last time
    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')
    y_pred = vmap(net, (None, None, 0))(model, frozen_para, x_test)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(
        y_test.flatten())
    relative_errors.append(relative_error)
    mse_errors.append(mse_error)
    print(f'ite:{ite * N_epochs},testing mse:{mse_error:.2e},relative:{relative_error:.2e}')



    # save model and results
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.dim}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.dim}.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=y_pred, y_test=y_test, relative_errors=relative_errors, mse_errors=mse_errors)

    # print the parameters
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # write the results on csv file
    header = "datatype, network, seed, final_loss_mean, training_time, total_ite, mse, relative"
    save_here = "results_fractal.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.datatype},{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},{ite * N_epochs},{mse_errors[-1]},{relative_errors[-1]}"
    with open(save_here, "a") as f:
        f.write(res)


def eval(key): # evaluate the trained model on a new test set and plot it
    # Generate sample data
    dim = args.dim
    ntest = args.ntest
    N_train = args.ntrain
    N_epochs = args.epochs
    learning_rate = args.lr
    ite = args.ite

    # Generate sampled data
    keys = random.split(key, 3)
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_in_set = training_points(dim=dim, interval=interval)
    x_train = x_in_set.sample(num=N_train, key=keys[0])
    x_test = x_in_set.sample(num=ntest, key=keys[1])
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    y_target = y_train.copy()
    # Add noise
    if args.noise == 1:
        sigma = 0.01
        y_train += np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test)
    normalizer = normalization(interval, dim, args.normalization)
    input_dim = dim
    output_dim = 1
    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    path = f'{args.datatype}_{args.network}_{args.seed}.eqx'
    frozen_para = model.get_frozen_para()
    # model = eqx.tree_deserialise_leaves(path, model)
    y_pred = vmap(net, (None, None, 0))(model, frozen_para, x_test)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f'testing mse: {mse_error},relative: {relative_error}')

    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test, 'r', label='Original Data')
    plt.plot(x_test, y_pred, 'b-', label='SincKAN')
    plt.title('Comparison of SincKAN and MLP Interpolations f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    path = f'{args.datatype}_{args.network}_{args.seed}.png'
    plt.savefig(path)





if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    train(key)
    #eval(key)

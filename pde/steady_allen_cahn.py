import sys

sys.path.append('../')
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
import time
from jax.nn import gelu, silu, tanh
from jax.lax import scan, stop_gradient
from jax import random, jit, vmap, grad
import os
import scipy
import matplotlib.pyplot as plt
import argparse
import jax
from data import get_data
from networks import get_network
from utils import normalization

parser = argparse.ArgumentParser(description="SincKAN")
parser.add_argument("--mode", type=str, default='train', help="mode of the network, "
                                                              "train: start training, eval: evaluation")
parser.add_argument("--datatype", type=str, default='allen_cahn', help="type of data")
parser.add_argument("--ntest", type=int, default=10000, help="the number of testing dataset")
parser.add_argument("--n_interior", type=int, default=2000,
                    help="the number of interior training dataset for each epochs")
parser.add_argument("--n_boundary", type=int, default=100,
                    help="the number of boundary training dataset for each epochs")
parser.add_argument("--dim", type=int, default=40, help="dimension of the problem")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=1000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--activation", type=str, default='tanh', help='the activation function')
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--interval", type=str, default="0.0,1.0", help='boundary of the interval')
parser.add_argument("--network", type=str, default="sinckan", help="type of network")
parser.add_argument("--kanshape", type=str, default="8", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=8, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=1, help='lenth of k for sinckan')
parser.add_argument("--init_h", type=float, default=2.0, help='initial value of h')
parser.add_argument("--decay", type=str, default='inverse', help='decay type for h')
parser.add_argument("--skip", type=int, default=1, help='1: use skip connection for sinckan')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--alpha", type=float, default=10, help='parameters for the width of poission')
parser.add_argument("--device", type=int, default=3, help="cuda number")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def solution_jax(x, alpha, c):
    A = jnp.mean(c * jnp.cos(x[:-1] + jnp.sin(x[1:]) + x[1:] * jnp.sin(x[:-1])))
    # A = jnp.mean(jnp.exp(-c * x[:-1] * x[1:]))
    B = - alpha * np.sum(x ** 2)
    return A * jnp.exp(B)
    # return jnp.exp(B)


def right_hand_side(x, alpha, c, dim):
    u = lambda model, alpha, c, *x: model(jnp.stack([*x]), alpha, c)
    f = jnp.sum(
        jnp.stack([grad(grad(u, argnums=i + 3), argnums=i + 3)(solution_jax, alpha, c, *x) for i in range(dim)])) \
        + u(solution_jax, alpha, c, *x) - u(solution_jax, alpha, c, *x) ** 3
    return f


class interior_points():
    def __init__(self, dim, interval=(-1, 1)):
        self.dim = dim
        self.points = jnp.linspace(interval[0], interval[1], 1000)[1:-1]

    def sample(self, num, key):
        keys = random.split(key, self.dim)
        points = jnp.concatenate([random.choice(key, self.points, shape=(num, 1), replace=True) for key in keys], -1)
        return points


class boundary_points():
    def __init__(self, dim, generate_data, interval=(-1, 1), alpha=100, c=0):
        self.dim = dim
        self.points = jnp.linspace(interval[0], interval[1], 100)
        self.interval = interval
        self.generate_data = generate_data
        self.alpha = alpha
        self.c = c

    def sample(self, num, key):
        keys = random.split(key, self.dim + 1)
        x = jnp.concatenate([random.choice(key, self.points, shape=(num, 1), replace=True) for key in keys[:-1]], -1)
        keys = random.split(keys[-1], 2)
        boundary = jax.random.randint(keys[0], num, 0, 2) * (self.interval[1] - self.interval[0]) + self.interval[0]
        idx_bd = jax.random.randint(keys[1], num, 0, self.dim)
        vset = lambda p, idx, value: p.at[idx].set(value)
        x = vmap(vset, (0, 0, 0))(x, idx_bd, boundary)
        y = self.generate_data(x, self.alpha, self.c)
        return x, y


def net(model, frozen_para, *x):
    return model(jnp.stack([*x]), frozen_para)[0]


def residual(model, x, frozen_para, r_s):
    dim = x.shape[0]
    u_output = net(model, frozen_para, *x)
    f = jnp.sum(jnp.stack([grad(grad(net, argnums=i + 2), argnums=i + 2)(model, frozen_para, *x) for i in range(dim)])) \
        + u_output - u_output ** 3
    return f - r_s


def boundary(model, x, frozen_para):
    return net(model, frozen_para, *x)


def output_test(model, x, frozen_para):
    return net(model, frozen_para, *x)


def compute_loss(model, ob_x, ob_sup, frozen_para):
    res = vmap(residual, (None, 0, None, 0))(model, ob_x[:, :-1], frozen_para, ob_x[:, -1])
    r = (res ** 2).mean()
    ob_b = vmap(boundary, (None, 0, None))(model, ob_sup[:, :-1], frozen_para)
    l_b = ((ob_b - ob_sup[:, -1]) ** 2).mean()
    return r + 100 * l_b


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def make_step(model, ob_x, ob_sup, frozen_para, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, ob_x, ob_sup, frozen_para)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train(key):
    keys = random.split(key, 3)
    # Get hyterparameters
    interval = args.interval.split(',')
    dim = args.dim
    alpha = args.alpha / dim
    vec_c = np.random.normal(0, 1, (dim - 1,))
    vec_c = vec_c
    ntest = args.ntest
    N_interior = args.n_interior
    N_b = args.n_boundary * dim
    N_epochs = args.epochs
    ite = args.ite
    learning_rate = args.lr
    generate_data = get_data(args.datatype)
    # Generate sampled data
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_b_set = boundary_points(dim=dim, generate_data=generate_data, interval=interval, alpha=alpha, c=vec_c)
    x_in_set = interior_points(dim=dim, interval=interval)
    x_test = jnp.concatenate([x_in_set.sample(num=int(ntest * 0.8), key=keys[0]),
                              x_b_set.sample(num=int(ntest * 0.2), key=keys[1])[0]], 0)

    y_test = generate_data(x_test, alpha=alpha, c=vec_c)
    normalizer = normalization(interval, dim, args.normalization)
    input_dim = dim
    output_dim = 1

    # Choose the model
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()

    # parameters of optimizer
    N_drop = 10000
    gamma = 0.95
    sc = optax.exponential_decay(learning_rate, N_drop, gamma)
    optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    history = []
    T = []
    errors = []
    for j in range(ite * N_epochs):
        if j % N_epochs == 0:
            # sample
            keys = random.split(keys[-1], 3)
            input_points = x_in_set.sample(N_interior, keys[0])
            ob_x = jnp.concatenate([input_points,
                                    vmap(right_hand_side, (0, None, None, None))(input_points, alpha, vec_c, dim,
                                                                                 ).reshape(-1, 1)],
                                   -1)
            x_b, y_b = x_b_set.sample(N_b, keys[1])
            ob_sup = jnp.concatenate([x_b, y_b], -1)

        T1 = time.time()
        loss, model, opt_state = make_step(model, ob_x, ob_sup, frozen_para, optim, opt_state, )
        T2 = time.time()
        T.append(T2 - T1)
        history.append(loss.item())
        if j % N_epochs == 0:
            # test
            y_pred = vmap(output_test, (None, 0, None))(model, x_test, frozen_para)
            mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
            relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
            errors.append(relative_error)
            print(f'testing mse: {mse_error:.2e},relative: {relative_error:.2e}')

    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')

    y_pred = vmap(output_test, (None, 0, None))(model, x_test, frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    errors.append(relative_error)
    print(f'testing mse: {mse_error:.2e},relative: {relative_error:.2e}')

    # save model and results
    path = f'./results/allen_cahn/{args.datatype}_{args.network}_{args.seed}_{args.alpha}_{args.dim}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'./results/allen_cahn/{args.datatype}_{args.network}_{args.seed}_{args.alpha}_{args.dim}.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=y_pred, y_test=y_test, x_test=x_test, errors=errors)

    # print the parameters
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # write the reuslts on csv file
    header = "datatype, network, seed, alpha, dim, final_loss_mean, training_time, total_ite,total_param, fine_mse, fine_relative"
    save_here = "results.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.datatype},{args.network},{args.seed},{args.alpha},{args.dim},{history[-1]},{np.sum(np.array(T))},{param_count},{ite * N_epochs},{mse_error},{relative_error}"
    with open(save_here, "a") as f:
        f.write(res)


def eval(key):
    # Generate sampled data
    dim = args.dim
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    keys = random.split(key, 3)
    # Get hyterparameters
    vec_c = random.normal(keys[0], shape=(dim - 1,))
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]
    generate_data = get_data(args.datatype)
    y_test = generate_data(x_test, alpha=args.alpha, c=vec_c)
    normalizer = normalization(x_test, args.normalization)

    input_dim = 1
    output_dim = 1

    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}.eqx'
    model = eqx.tree_deserialise_leaves(path, model)

    y_pred = vmap(net, (None, 0, None))(model, x_test[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f'testing mse: {mse_error:.2e},relative: {relative_error:.2e}')

    plt.figure(figsize=(10, 5))
    plt.plot(x_test, y_test, 'r', label='exact solution')
    plt.plot(x_test, y_pred, 'b-', label='SincKAN')
    plt.title('Comparison of SincKAN')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend()
    path = f'{args.datatype}_{args.network}_{args.seed}.png'
    plt.savefig(path)


if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    if args.mode == 'train':
        train(key)
    elif args.mode == 'eval':
        eval(key)

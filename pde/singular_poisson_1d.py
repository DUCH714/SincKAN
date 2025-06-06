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
from jax.scipy.special import gamma
import os
import scipy
import matplotlib.pyplot as plt
import argparse
import jax
from data import get_data
from networks import get_network
from utils import normalization, get_matrix_1d

parser = argparse.ArgumentParser(description="SincKAN")
parser.add_argument("--mode", type=str, default='train', help="mode of the network, "
                                                              "train: start training, eval: evaluation")
parser.add_argument("--datatype", type=str, default='singular_frac', help="type of data")
parser.add_argument("--npoints", type=int, default=101, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=1000, help="the number of testing dataset")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=50000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--activation", type=str, default=None, help='the activation function')
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--interval", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--network", type=str, default="sinckan", help="type of network")
parser.add_argument("--kanshape", type=str, default="8", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=100, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=1, help='lenth of k for sinckan')
parser.add_argument("--init_h", type=float, default=4.0, help='initial value of h')
parser.add_argument("--decay", type=str, default='inverse', help='decay type for h')
parser.add_argument("--skip", type=int, default=1, help='1: use skip connection for sinckan')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--alpha", type=float, default=1.9, help='boundary layer parameters')
parser.add_argument("--initialization", type=str, default=None, help='the type of initialization of SincKAN')
parser.add_argument("--device", type=int, default=6, help="cuda number")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def net(model, x, frozen_para):
    return model(jnp.stack([x]), frozen_para)[0]


def residual(model, x, frozen_para, alpha, matrix_A):
    '''
    u_xx/alpha+u_x=0
    :param model:
    :param x:
    :param frozen_para:
    :param alpha:
    :return:
    '''
    u = vmap(net, (None, 0, None))(model, x, frozen_para)
    l_s = matrix_A @ u
    # r_s = (
    #         gamma(4) / gamma(4 - alpha) * (x ** (3 - alpha) + (1 - x) ** (3 - alpha))
    #         - 3 * gamma(5) / gamma(5 - alpha) * (x ** (4 - alpha) + (1 - x) ** (4 - alpha))
    #         + 3 * gamma(6) / gamma(6 - alpha) * (x ** (5 - alpha) + (1 - x) ** (5 - alpha))
    #         - gamma(7) / gamma(7 - alpha) * (x ** (6 - alpha) + (1 - x) ** (6 - alpha))
    # )
    return l_s - 1


def compute_loss(model, ob_x, ob_sup, frozen_para, alpha, matrix_A):
    res = residual(model, ob_x[:, 0], frozen_para, alpha, matrix_A)
    r = (res ** 2).mean()
    ob_b = vmap(net, (None, 0, None))(model, ob_sup[:, 0], frozen_para)
    l_b = ((ob_b - ob_sup[:, 1]) ** 2).mean()
    return r + 100 * l_b


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def make_step(model, ob_x, ob_sup, frozen_para, optim, opt_state, alpha, matrix_A):
    loss, grads = compute_loss_and_grads(model, ob_x, ob_sup, frozen_para, alpha, matrix_A)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train(key):
    # Generate sampled data
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train, args.alpha)
    y_test = generate_data(x_test, args.alpha)
    normalizer = normalization(x_train, args.normalization, 0)

    index_b = [-1, 0]
    x_b = x_train[index_b, :]
    y_b = y_train[index_b, :]
    ob_sup = jnp.concatenate([x_b, y_b], -1)

    input_dim = 1
    output_dim = 1

    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    matrix_A = get_matrix_1d(args.alpha, args.npoints, interval)/2/np.cos(np.pi*args.alpha/2)

    # Hyperparameters
    N_epochs = args.epochs
    ite = args.ite

    # parameters of optimizer
    learning_rate = args.lr
    N_drop = 10000
    gamma = 0.95
    sc = optax.exponential_decay(learning_rate, N_drop, gamma)
    optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    input_points = np.roll(x_train, -1)
    input_points = np.vstack((x_b, input_points))
    history = []
    T = []
    errors = []
    for j in range(ite * N_epochs):
        T1 = time.time()
        loss, model, opt_state = make_step(model, input_points, ob_sup, frozen_para, optim, opt_state,
                                           alpha=args.alpha, matrix_A=matrix_A)
        T2 = time.time()
        T.append(T2 - T1)
        history.append(loss.item())
        if j % N_epochs == 0:
            train_y_pred = vmap(net, (None, 0, None))(model, x_train[:, 0], frozen_para)
            train_mse_error = jnp.mean((train_y_pred.flatten() - y_train.flatten()) ** 2)
            train_relative_error = jnp.linalg.norm(train_y_pred.flatten() - y_train.flatten()) / jnp.linalg.norm(
                y_train.flatten())
            print(f'ite:{j},mse:{train_mse_error:.2e},relative:{train_relative_error:.2e}')
            errors.append(train_relative_error)

    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')

    train_y_pred = vmap(net, (None, 0, None))(model, x_train[:, 0], frozen_para)
    train_mse_error = jnp.mean((train_y_pred.flatten() - y_train.flatten()) ** 2)
    train_relative_error = jnp.linalg.norm(train_y_pred.flatten() - y_train.flatten()) / jnp.linalg.norm(
        y_train.flatten())
    print(f'training mse: {train_mse_error:.2e},relative: {train_relative_error:.2e}')
    errors.append(train_relative_error)

    y_pred = vmap(net, (None, 0, None))(model, x_test[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f'testing mse: {mse_error:.2e},relative: {relative_error:.2e}')

    # save model and results
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=y_pred, y_test=y_test, errors=errors)

    # print the parameters
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # write the reuslts on csv file
    header = "datatype, network, seed, final_loss_mean, training_time, total_ite,total_param, mse, relative, fine_mse, fine_relative"
    save_here = "results.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.datatype},{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},{param_count},{ite * N_epochs},{train_mse_error},{train_relative_error},{mse_error},{relative_error}"
    with open(save_here, "a") as f:
        f.write(res)


def eval(key):
    # Generate sampled data
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]
    generate_data = get_data(args.datatype)
    y_test = generate_data(x_test, args.alpha)
    normalizer = normalization(x_train, args.normalization, 0)

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

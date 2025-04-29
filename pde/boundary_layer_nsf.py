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
from networks_old import get_network
from utils import normalization

parser = argparse.ArgumentParser(description="SincKAN")
parser.add_argument("--datatype", type=str, default='bl', help="type of data")
parser.add_argument("--npoints", type=int, default=1000, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=1000, help="the number of testing dataset")
parser.add_argument("--ntrain", type=int, default=500, help="the number of training dataset for each epochs")
parser.add_argument("--ite", type=int, default=30, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=50000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--weight", type=float, default=100.0, help='weight of the boundary term')
parser.add_argument("--interval", type=str, default="0.0,1.0", help='boundary of the interval')
parser.add_argument("--network", type=str, default="sinckan", help="type of network")
parser.add_argument("--kanshape", type=str, default="8", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=8, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=1, help='lenth of k for sinckan')
parser.add_argument("--init_h", type=float, default=2.0, help='inital value of h')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--alpha", type=float, default=100, help='boundary layer parameters')
parser.add_argument("--device", type=int, default=2, help="cuda number")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def net(model, x, frozen_para):
    return model(jnp.stack([x]), frozen_para)[0]


def residual(model, x, frozen_para, alpha):
    '''
    u_xx/alpha+u_x=0
    :param model:
    :param x:
    :param frozen_para:
    :param alpha:
    :return:
    '''

    u_x = grad(net, argnums=1)(model, x, frozen_para)
    u_xx = grad(grad(net, argnums=1), argnums=1)(model, x, frozen_para)
    f = u_xx / alpha + u_x
    return f


def compute_loss(model, ob_x, ob_sup, frozen_para, weight, alpha):
    res = vmap(residual, (None, 0, None, None))(model, ob_x[:, 0], frozen_para, alpha)
    r = (res ** 2).mean()
    ob_b = vmap(net, (None, 0, None))(model, ob_sup[:, 0], frozen_para)
    l_b = ((ob_b - ob_sup[:, 1]) ** 2).mean()
    return r + weight[0]*l_b

def compute_loss_item(model, ob_x, ob_sup, frozen_para, alpha):
    res = vmap(residual, (None, 0, None, None))(model, ob_x[:, 0], frozen_para, alpha)
    r = (res ** 2).mean()
    ob_b = vmap(net, (None, 0, None))(model, ob_sup[:, 0], frozen_para)
    l_b = ((ob_b - ob_sup[:, 1]) ** 2).mean()
    return r, l_b

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def make_step(model, ob_x, ob_sup, frozen_para, optim, opt_state,weight, alpha):
    loss, grads = compute_loss_and_grads(model, ob_x, ob_sup, frozen_para,weight,alpha)
    loss_r, loss_b = compute_loss_item(model, ob_x, ob_sup, frozen_para, alpha)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [loss,loss_r,loss_b], model, opt_state

@eqx.filter_value_and_grad
def loss_r(model, ob_x,frozen_para,alpha):
    res = vmap(residual, (None, 0,  None, None))(model, ob_x[:,0], frozen_para, alpha)
    return (res ** 2).mean()

@eqx.filter_value_and_grad
def loss_b(model, ob_sup, frozen_para):
    ob_b = vmap(net, (None, 0, None))(model, ob_sup[:, 0], frozen_para)
    return ((ob_b - ob_sup[:, 1]) ** 2).mean()

#@eqx.filter_jit
def get_weights(model, ob_x, ob_sup, frozen_para, lams, alpha=0.1):
    _, grads_r = loss_r(model, ob_x, frozen_para, args.alpha)
    _, grads_b = loss_b(model, ob_sup,frozen_para)
    J_rs = []
    for i in range(len(model.matrices)):
        # J_r = jnp.concatenate([grads_r.matrices[i].reshape(-1), grads_r.biases[i].reshape(-1)], -1)
        J_rs.append(grads_r.matrices[i].reshape(-1))
    J_rs = jnp.concatenate(J_rs, -1)

    J_bs = []
    for i in range(len(model.matrices)):
        # J_b = jnp.concatenate([grads_b.matrices[i].reshape(-1), grads_b.biases[i].reshape(-1)], -1)
        J_bs.append(grads_b.matrices[i].reshape(-1))
    J_bs = jnp.concatenate(J_bs, -1)

    l_r = jnp.mean(jnp.abs(J_rs))

    l_b = jnp.mean(jnp.abs(J_bs))
    lamb = (1 - alpha) * lams[0] + alpha * l_r / l_b
    lams = [lamb]

    return stop_gradient(lams)

def train(key):
    # Generate sample data
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train, alpha=args.alpha)

    y_test = generate_data(x_test, alpha=args.alpha)
    # eps=1e-4
    # trans = lambda x: np.log((x-lowb+eps)/(upb+eps-x))
    # x_train = trans(x_train)
    # x_test = trans(x_test)
    normalizer = normalization(x_train, 0, args.normalization)
    ob_x = x_train  # np.concatenate([x_train, y_train], -1)
    index_b = [0, -1]
    x_b = x_train[index_b, :]
    y_b = y_train[index_b, :]
    ob_sup = jnp.concatenate([x_b, y_b], -1)
    ob_x = ob_x[1:-1]
    input_dim = 1
    output_dim = 1
    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    # Hyperparameters
    N_train = args.ntrain
    N_epochs = args.epochs
    ite = args.ite

    # parameters of optimizer
    learning_rate = args.lr
    N_drop = 10000
    gamma = 0.95
    sc = optax.exponential_decay(learning_rate, N_drop, gamma)
    #optim = optax.lion(learning_rate=sc)
    optim = optax.adamw(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    keys = random.split(keys[-1], 2)
    input_points = random.choice(keys[0], ob_x, shape=(N_train,), replace=False)
    history = []
    T = []
    erros=[]
    loss_rs=[]
    loss_bs=[]
    weight = [1]

    for j in range(ite * N_epochs):
        T1 = time.time()
        loss, model, opt_state = make_step(model, input_points, ob_sup, frozen_para, optim, opt_state,weight,
                                           alpha=args.alpha)
        T2 = time.time()
        T.append(T2 - T1)
        if j % ite == 0:
            history.append(loss[0].item())
            loss_rs.append(loss[1].item())
            loss_bs.append(loss[2].item())
        if j % N_epochs == 0:
            if j!=0:
                weight = get_weights(model, input_points, ob_sup, frozen_para, weight)
                print(f'{weight[0]:.2e}')
            keys = random.split(keys[-1], 2)
            input_points = random.choice(keys[0], ob_x, shape=(N_train,), replace=False)
            train_y_pred = vmap(net, (None, 0, None))(model, x_train[:, 0], frozen_para)
            train_mse_error = jnp.mean((train_y_pred.flatten() - y_train.flatten()) ** 2)
            train_relative_error = jnp.linalg.norm(train_y_pred.flatten() - y_train.flatten()) / jnp.linalg.norm(
                y_train.flatten())
            print(f'ite:{j},mse:{train_mse_error:.2e},relative:{train_relative_error:.2e}')
            erros.append(train_relative_error)
    # eval
    # if args.network == 'sinckan':
    #     netlayer = lambda model, x, frozen_para: model(jnp.stack([x]), frozen_para)
    #     z0 = vmap(netlayer, (None, 0, None))(model.layers[0], x_train[:, 0], frozen_para[0])
    #     # z1 = vmap(netlayer, (None, 0, None))(model.layers[1], x_train[:, 0], frozen_para[1])
    #     np.savez('inter.npz', z0=z0)
    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')
    train_y_pred = vmap(net, (None, 0, None))(model, x_train[:, 0], frozen_para)
    train_mse_error = jnp.mean((train_y_pred.flatten() - y_train.flatten()) ** 2)
    train_relative_error = jnp.linalg.norm(train_y_pred.flatten() - y_train.flatten()) / jnp.linalg.norm(
        y_train.flatten())
    print(f'training mse: {train_mse_error:.2e},relative: {train_relative_error:.2e}')
    erros.append(train_relative_error)
    y_pred = vmap(net, (None, 0, None))(model, x_test[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f'testing mse: {mse_error:.2e},relative: {relative_error:.2e}')

    # save model and results
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}_nsf.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}_nsf.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=y_pred, y_test=y_test,errors=erros,loss_b=loss_bs,loss_r=loss_rs)

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
    # Generate sample data
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_train = np.linspace(lowb, upb, num=args.npoints)[:, None]
    x_test = np.linspace(lowb, upb, num=args.ntest)[:, None]
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train, alpha=args.alpha)
    # Add noise
    if args.noise == 1:
        sigma = 0.1
        y_target = y_train.copy()
        y_train += np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test, alpha=args.alpha)
    input_dim = 1
    output_dim = 1
    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, keys)
    frozen_para = model.get_frozen_para()
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.alpha}.eqx'
    model = eqx.tree_deserialise_leaves(path, model)

    y_pred = vmap(net, (None, 0, None))(model, x_test[:, 0], frozen_para)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(y_test.flatten())
    print(f'mse: {mse_error},relative: {relative_error}')

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













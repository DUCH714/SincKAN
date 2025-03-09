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
parser.add_argument("--datatype", type=str, default='ns_tg', help="type of data")
parser.add_argument("--npoints", type=int, default=100, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=100, help="the number of testing dataset")
parser.add_argument("--ntrain", type=int, default=50000, help="the number of training dataset for each epochs")
parser.add_argument("--dim", type=int, default=3, help="dimension of the problem")
parser.add_argument("--ite", type=int, default=30, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=50000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--noise", type=int, default=0, help="add noise or not, 0: no noise, 1: add noise")
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--interval", type=str, default="0.0,1.0", help='boundary of the interval')
parser.add_argument("--network", type=str, default="sinckan", help="type of network")
parser.add_argument("--kanshape", type=str, default="8", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=6, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=1, help='lenth of k for sinckan')
parser.add_argument("--init_h", type=float, default=2.0, help='initial value of h')
parser.add_argument("--decay", type=str, default='inverse', help='decay type for h')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--device", type=int, default=3, help="cuda number")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def u_net(model, x, y, t, frozen_para):
    u = model(jnp.stack([x, y, t]), frozen_para)[0]
    return u


def v_net(model, x, y, t, frozen_para):
    v = model(jnp.stack([x, y, t]), frozen_para)[1]
    return v


def p_net(model, x, y, t, frozen_para):
    p = model(jnp.stack([x, y, t]), frozen_para)[2]
    return p


def residual(model, x, y, t, frozen_para, nu):
    '''

    :param model:
    :param x:
    :param frozen_para:
    :param nu:
    :return:
    '''

    u = u_net(model, x, y, t, frozen_para)
    u_t = grad(u_net, argnums=3)(model, x, y, t, frozen_para)
    u_x = grad(u_net, argnums=1)(model, x, y, t, frozen_para)
    u_xx = grad(grad(u_net, argnums=1), argnums=1)(model, x, y, t, frozen_para)
    u_y = grad(u_net, argnums=2)(model, x, y, t, frozen_para)
    u_yy = grad(grad(u_net, argnums=2), argnums=2)(model, x, y, t, frozen_para)

    v = v_net(model, x, y, t, frozen_para)
    v_t = grad(v_net, argnums=3)(model, x, y, t, frozen_para)
    v_x = grad(v_net, argnums=1)(model, x, y, t, frozen_para)
    v_xx = grad(grad(v_net, argnums=1), argnums=1)(model, x, y, t, frozen_para)
    v_y = grad(v_net, argnums=2)(model, x, y, t, frozen_para)
    v_yy = grad(grad(v_net, argnums=2), argnums=2)(model, x, y, t, frozen_para)

    p_x = grad(p_net, argnums=1)(model, x, y, t, frozen_para)
    p_y = grad(p_net, argnums=2)(model, x, y, t, frozen_para)

    f_u = u_t + (u * u_x + v * u_y) + p_x - nu * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + p_y - nu * (v_xx + v_yy)
    f_e = u_x + v_y
    return f_u, f_v, f_e


def compute_loss(model, ob_xyt, ob_sup, frozen_para, nu):
    f_u, f_v, f_e = vmap(residual, (None, 0, 0, 0, None, None))(model, ob_xyt[:, 0], ob_xyt[:, 1], ob_xyt[:, 2],
                                                                frozen_para,
                                                                nu)
    r = (f_u ** 2).mean() + (f_v ** 2).mean() + (f_e ** 2).mean()
    u_sup = vmap(u_net, (None, 0, 0, 0, None))(model, ob_sup[:, 0], ob_sup[:, 1], ob_sup[:, 2], frozen_para)
    v_sup = vmap(v_net, (None, 0, 0, 0, None))(model, ob_sup[:, 0], ob_sup[:, 1], ob_sup[:, 2], frozen_para)

    l_b = ((u_sup - ob_sup[:, 3]) ** 2).mean() + ((v_sup - ob_sup[:, 4]) ** 2).mean()
    return r + 100 * l_b


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


@eqx.filter_jit
def make_step(model, ob_xyt, ob_sup, frozen_para, optim, opt_state, nu):
    loss, grads = compute_loss_and_grads(model, ob_xyt, ob_sup, frozen_para, nu)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


def train(key):
    # Generate sample data
    N_t = 11
    T_end = 0.1
    Re = 400
    nu = 1 / Re
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x1_train, x2_train = np.meshgrid(*[np.linspace(lowb, upb, num=args.npoints)] * 2)
    x1_test, x2_test = np.meshgrid(*[np.linspace(lowb, upb, num=args.ntest)] * 2)
    TT = jnp.linspace(0, T_end, N_t)

    XX = jnp.tile(x1_train[:, :, None], (1, 1, N_t))  # N×N×T
    YY = jnp.tile(x2_train[:, :, None], (1, 1, N_t))  # N×N×T
    TT = jnp.tile(TT[None, None, :], (args.npoints, args.npoints, 1))  # N×N×T
    generate_data = get_data(args.datatype)
    UU, VV, PP = generate_data(XX, YY, TT, nu=nu)  # N×N×T

    u_test, v_test, p_test = generate_data(x1_test, x2_test, T_end, nu=nu)

    index_b = np.zeros((args.npoints, args.npoints))
    index_b[:, 0] = 1
    index_b[:, -1] = 1
    index_b[0, :] = 1
    index_b[-1, :] = 1
    index_b = (index_b == 1)

    x_train = XX[~index_b, :].reshape(-1, 1)
    y_train = YY[~index_b, :].reshape(-1, 1)
    t_train = TT[~index_b, :].reshape(-1, 1)
    x1_test = x1_test.reshape(-1, 1)
    x2_test = x2_test.reshape(-1, 1)

    x0 = XX[:, :, 0:1].reshape(-1, 1)
    y0 = YY[:, :, 0:1].reshape(-1, 1)
    t0 = TT[:, :, 0:1].reshape(-1, 1)
    u0 = UU[:, :, 0:1].reshape(-1, 1)
    v0 = VV[:, :, 0:1].reshape(-1, 1)
    ob_0 = np.concatenate([x0, y0, t0, u0, v0], -1)

    ob_xyt = np.concatenate([x_train, y_train, t_train], -1)

    x_b = XX[index_b, :].reshape(-1, 1)
    y_b = YY[index_b, :].reshape(-1, 1)
    t_b = TT[index_b, :].reshape(-1, 1)
    u_b = UU[index_b, :].reshape(-1, 1)
    v_b = VV[index_b, :].reshape(-1, 1)
    ob_b = jnp.concatenate([x_b, y_b, t_b, u_b, v_b], -1)
    ob_sup = jnp.concatenate([ob_b, ob_0], 0)
    input_dim = 3
    output_dim = 3
    # Choose the model
    normalizer = normalization(interval,args.dim, args.normalization,is_t=1)
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
    # optim = optax.lion(learning_rate=1e-4)
    optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    keys = random.split(keys[-1], 2)
    input_points = random.choice(keys[0], ob_xyt, shape=(N_train,), replace=False)
    history = []
    T = []
    for j in range(ite * N_epochs):
        T1 = time.time()
        loss, model, opt_state = make_step(model, input_points, ob_sup, frozen_para, optim, opt_state,
                                           nu=nu)
        T2 = time.time()
        T.append(T2 - T1)
        history.append(loss.item())
        if j % N_epochs == 0:
            keys = random.split(keys[-1], 2)
            input_points = random.choice(keys[0], ob_xyt, shape=(N_train,), replace=False)
            u_pred = vmap(u_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end,
                                                           frozen_para)
            train_mse_error_u = jnp.mean((u_pred.flatten() - u_test.flatten()) ** 2)
            train_relative_error_u = jnp.linalg.norm(u_pred.flatten() - u_test.flatten()) / jnp.linalg.norm(
                u_test.flatten())

            v_pred = vmap(v_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end,
                                                           frozen_para)
            train_mse_error_v = jnp.mean((v_pred.flatten() - v_test.flatten()) ** 2)
            train_relative_error_v = jnp.linalg.norm(v_pred.flatten() - v_test.flatten()) / jnp.linalg.norm(
                v_test.flatten())

            print(f'ite:{j},mse_u:{train_mse_error_u:.2e},relative_u:{train_relative_error_u:.2e},'
                  f'mse_v:{train_mse_error_v:.2e},relative_v:{train_relative_error_v:.2e}')
    # eval
    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')

    u_pred = vmap(u_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end, frozen_para)
    train_mse_error_u = jnp.mean((u_pred.flatten() - u_test.flatten()) ** 2)
    train_relative_error_u = jnp.linalg.norm(u_pred.flatten() - u_test.flatten()) / jnp.linalg.norm(
        u_test.flatten())
    v_pred = vmap(v_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end, frozen_para)
    train_mse_error_v = jnp.mean((v_pred.flatten() - v_test.flatten()) ** 2)
    train_relative_error_v = jnp.linalg.norm(v_pred.flatten() - v_test.flatten()) / jnp.linalg.norm(
        v_test.flatten())
    print(f'ite:{ite * N_epochs},mse_u:{train_mse_error_u:.2e},relative_u:{train_relative_error_u:.2e},'
          f'mse_v:{train_mse_error_v:.2e},relative_v:{train_relative_error_v:.2e}')

    # save model and results
    path = f'{args.datatype}_{args.network}_{args.seed}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.datatype}_{args.network}_{args.seed}.npz'
    np.savez(path, loss=history, avg_time=avg_time, u_pred=u_pred, u_test=u_test, v_pred=v_pred, v_test=v_test)

    # print the parameters
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')
    # write the reuslts on csv file
    header = "datatype, network, seed, final_loss_mean, training_time, total_ite,total_param, mse_u, relative_u, mse_v, relative_v"
    save_here = "results.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.datatype},{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},{param_count},{ite * N_epochs},{train_mse_error_u},{train_relative_error_u},{train_mse_error_v},{train_relative_error_v}"
    with open(save_here, "a") as f:
        f.write(res)


def eval(key):
    # Generate sample data
    N_t = 11
    T_end = 0.1
    Re = 400
    nu = 1 / Re
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x1_test, x2_test = np.meshgrid(*[np.linspace(lowb, upb, num=args.ntest)] * 2)
    generate_data = get_data(args.datatype)
    u_test, v_test, p_test = generate_data(x1_test, x2_test, T_end, nu=nu)

    x1_test = x1_test.reshape(-1, 1)
    x2_test = x2_test.reshape(-1, 1)

    input_dim = 3
    output_dim = 3
    # Choose the model
    normalizer = normalization(interval,args.dim, args.normalization,is_t=1)
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    path = f'{args.datatype}_{args.network}_{args.seed}.eqx'
    model = eqx.tree_deserialise_leaves(path, model)

    u_pred = vmap(u_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end, frozen_para)
    train_mse_error_u = jnp.mean((u_pred.flatten() - u_test.flatten()) ** 2)
    train_relative_error_u = jnp.linalg.norm(u_pred.flatten() - u_test.flatten()) / jnp.linalg.norm(
        u_test.flatten())
    v_pred = vmap(v_net, (None, 0, 0, None, None))(model, x1_test[:, 0], x2_test[:, 0], T_end, frozen_para)
    train_mse_error_v = jnp.mean((v_pred.flatten() - v_test.flatten()) ** 2)
    train_relative_error_v = jnp.linalg.norm(v_pred.flatten() - v_test.flatten()) / jnp.linalg.norm(
        v_test.flatten())
    print(f'mse_u:{train_mse_error_u:.2e},relative_u:{train_relative_error_u:.2e},'
          f'mse_v:{train_mse_error_v:.2e},relative_v:{train_relative_error_v:.2e}')

    plt.figure(figsize=(10, 5))
    plt.contourf(x1_test.reshape(100, 100), x2_test.reshape(100, 100), u_test.reshape(100, 100))
    plt.title('target')
    path = f'{args.datatype}_{args.network}_{args.seed}_test_u.png'
    plt.savefig(path)

    plt.figure(figsize=(10, 5))
    plt.contourf(x1_test.reshape(100, 100), x2_test.reshape(100, 100), u_pred.reshape(100, 100))
    plt.title('prediction')
    path = f'{args.datatype}_{args.network}_{args.seed}_pred_u.png'
    plt.savefig(path)

    plt.figure(figsize=(10, 5))
    plt.contourf(x1_test.reshape(100, 100), x2_test.reshape(100, 100), v_test.reshape(100, 100))
    plt.title('target')
    path = f'{args.datatype}_{args.network}_{args.seed}_test_v.png'
    plt.savefig(path)

    plt.figure(figsize=(10, 5))
    plt.contourf(x1_test.reshape(100, 100), x2_test.reshape(100, 100), v_pred.reshape(100, 100))
    plt.title('prediction')
    path = f'{args.datatype}_{args.network}_{args.seed}_pred_v.png'
    plt.savefig(path)


if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    if args.mode == 'train':
        train(key)
    elif args.mode == 'eval':
        eval(key)

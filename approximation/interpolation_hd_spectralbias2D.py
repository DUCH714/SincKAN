import sys

sys.path.append('../')
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import time
import optax
import jax
from jax.nn import gelu, silu, tanh
from jax.lax import scan, stop_gradient
from jax import random, jit, vmap, grad
import os
import scipy
import matplotlib
import matplotlib.pyplot as plt
import argparse

from data import get_data
from networks import get_network
from utils import normalization_hd
from numpy import pi

import argparse
from jax.lax import scan
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser(description="SincKAN")
parser.add_argument("--mode", type=str, default='train', help="mode of the network, "
                                                              "train: start training, eval: evaluation")
parser.add_argument("--datatype", type=str, default='scaling', help="type of data")
parser.add_argument("--npoints", type=int, default=2000, help="the number of total dataset")
parser.add_argument("--ntest", type=int, default=256, help="the number of testing dataset")
parser.add_argument("--ntrain", type=int, default=10000, help="the number of training dataset for each epochs")
parser.add_argument("--dim", type=int, default=2, help="dim of the problem")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=5000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="the name")
parser.add_argument("--activation", type=str, default='tanh', help='the activation function')
parser.add_argument("--interval", type=str, default="-2.0,2.0", help='boundary of the interval')
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
    def __init__(self, dim, interval=(-2,2)):
        self.dim = dim
        self.points = jnp.linspace(interval[0], interval[1], 40000)

    def sample(self, num, key):
        keys = random.split(key, self.dim)
        points = jnp.concatenate([random.choice(key, self.points, shape=(num, 1), replace=True) for key in keys], -1)
        return points


def net(model, frozen_para, *x):
    return model(jnp.stack(*x), frozen_para)[0]

def compute_loss(model, ob_x, frozen_para, dim):
    output = vmap(net, (None, None, 0))(model, frozen_para, ob_x[:, :dim])  # vmap applies net to a batch of inputs simultaneously, efficient for batch processing in GPUs, (None, 0, None) specifies no action on model and frozen_para, only act on ob_x[:, :n_points]

    return ((output - ob_x[:, dim]) ** 2).mean()
# if output and ob_x[:,2] are both batches of vectors, mean() will take average over their square difference, otherwise it is just the same as not having mean()


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss) #eqx library provides a function to filter trainable parameters from nontrainable parameters, this computes compute_loss and its grad wrt trainable parameters


@eqx.filter_jit
def make_step(model, ob_x, frozen_para, optim, opt_state, dim):
    loss, grads = compute_loss_and_grads(model, ob_x, frozen_para, dim)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array)) #updates the trainable parameters along the grads, opt_state is the current optimizer state containing momentum term or learning rate. updates is the change need to be added to the param, opt_stat is the latest optimizer state
    model = eqx.apply_updates(model, updates) #updating model parameters by adding the changes contained in updates
    return loss, model, opt_state

def make_eval_fn(model, frozen_para):
    @jit
    def eval_batch(carry, xb):
        yb = vmap(net, (None, None, 0))(model, frozen_para, xb)
        return carry, yb
    return eval_batch
def batched_predict(model, frozen_para, x, batch_size=4_096):
    eval_batch = make_eval_fn(model, frozen_para)   # <-- create jit-ed fn
    n = x.shape[0]
    n_batch = n // batch_size
    x_batched = x[:n_batch * batch_size].reshape(n_batch, batch_size, -1)
    _, y_batch = scan(eval_batch, None, x_batched)
    y_pred = y_batch.reshape(-1, 1)
    rem = x[n_batch * batch_size:]
    if rem.size:
        y_rem = vmap(net, (None, None, 0))(model, frozen_para, rem)
        y_pred = jnp.concatenate([y_pred, y_rem])
    return y_pred

def train(key):
    # get hyperparameters
    dim = args.dim
    ntest = args.ntest
    N_train = args.ntrain
    EPOCHS = args.epochs           # 经典定义：覆盖全数据 EPOCHS 次
    base_lr = args.lr
    batch_size = int(args.npoints) # 用 npoints 作为正统的 batch size

    # -------- 数据与模型（保持不变） --------
    keys = random.split(key, 3)
    interval = args.interval.split(',')
    lowb, upb = float(interval[0]), float(interval[1])
    interval = [lowb, upb]
    x_in_set = training_points(dim=dim, interval=interval)
    x_train = x_in_set.sample(num=N_train, key=keys[0])
    res = 256
    x = jnp.linspace(-2, 2, res)  # 1-D grid
    xx, yy = jnp.meshgrid(x, x, indexing='ij')  # shape (256, 256)
    x_test = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)  # (256*256, 2)
    generate_data = get_data(args.datatype)
    y_train = generate_data(x_train)
    y_target = y_train.copy()
    if args.noise == 1:
        sigma = 0.01
        y_train += np.random.normal(0, sigma, y_train.shape)

    y_test = generate_data(x_test)

    normalizer = normalization_hd(interval, dim, args.normalization)

    input_dim = dim
    output_dim = 1

    keys = random.split(key, 3)
    model = get_network(args, input_dim, output_dim, interval, normalizer, keys)
    frozen_para = model.get_frozen_para()
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # -------- 训练集拼接与 batch 准备 --------
    ob_x = jnp.concatenate([x_train, y_train.reshape(-1, 1)], -1)
    batch_size = max(1, min(batch_size, ob_x.shape[0]))
    steps_per_epoch = int(np.ceil(ob_x.shape[0] / batch_size))
    total_steps = EPOCHS * steps_per_epoch

    # -------- 学习率：Warmup + Cosine --------
    warmup_steps = max(1, int(0.05 * total_steps))  # 前 5% 线性 warmup
    warmup = optax.linear_schedule(init_value=0.0, end_value=base_lr, transition_steps=warmup_steps)
    cosine = optax.cosine_decay_schedule(init_value=base_lr,
                                         decay_steps=max(1, total_steps - warmup_steps),
                                         alpha=0.1)  # 衰减到 0.1 * base_lr
    lr_schedule = optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])

    # -------- 优化器：梯度裁剪 + AdamW --------
    weight_decay = 1e-4
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # -------- 训练循环（正统 mini-batch）--------
    history = []
    T = []
    relative_errors = []
    mse_errors = []

    step = 0
    for epoch in range(EPOCHS):
        # 每个 epoch 打乱一次
        keys = random.split(keys[-1], 2)
        perm = random.permutation(keys[0], ob_x.shape[0])
        ob_x_shuffled = ob_x[perm]

        # 遍历整个数据集（mini-batch）
        for s in range(steps_per_epoch):
            start = s * batch_size
            end = min((s + 1) * batch_size, ob_x_shuffled.shape[0])
            input_points = ob_x_shuffled[start:end]

            T1 = time.time()
            loss, model, opt_state = make_step(model, input_points, frozen_para, optim, opt_state, dim)
            T2 = time.time()
            T.append(T2 - T1)
            history.append(loss.item())
            step += 1

        # 每个 epoch 结束在 x_test 上评估（y_pred 必须在 x_test）
        y_pred = batched_predict(model, frozen_para, x_test, batch_size=4_096)
        mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
        relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(
            y_test.flatten())
        relative_errors.append(relative_error)
        mse_errors.append(mse_error)
        print(f'epoch:{epoch+1}/{EPOCHS}, testing mse:{mse_error:.2e}, relative:{relative_error:.2e}')

    # -------- 训练结束后的最终评估与打印频率 --------
    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')  # 每步迭代频率（与原先风格一致）

    # 最终评估（保持计算方式不变）
    y_pred = batched_predict(model, frozen_para, x_test, batch_size=4_096)
    mse_error = jnp.mean((y_pred.flatten() - y_test.flatten()) ** 2)
    relative_error = jnp.linalg.norm(y_pred.flatten() - y_test.flatten()) / jnp.linalg.norm(
        y_test.flatten())
    relative_errors.append(relative_error)
    mse_errors.append(mse_error)
    print(f'final, testing mse:{mse_error:.2e}, relative:{relative_error:.2e}')

    # -------- 保存模型与结果（保持原样） --------
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.dim}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.datatype}_{args.network}_{args.seed}_{args.dim}.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=y_pred, y_test=y_test,
             relative_errors=relative_errors, mse_errors=mse_errors)

    # 打印参数数量（保持不变）
    param_count = sum(x.size if eqx.is_array(x) else 0 for x in jax.tree.leaves(model))
    print(f'total parameters: {param_count}')

    # 写 CSV（字段不变；total_ite 用总步数替代）
    header = "datatype, network, seed, final_loss_mean, training_time, total_ite, mse, relative"
    save_here = "results_fractal.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.datatype},{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},{total_steps},{mse_errors[-1]},{relative_errors[-1]}"
    with open(save_here, "a") as f:
        f.write(res)

def plot_three_pdfs(npz_path: str):
    data = np.load(npz_path)
    pred = data['y_pred']
    ref  = data['y_test']

    res = 256
    x = np.linspace(-2, 2, res)
    xx, yy = np.meshgrid(x, x, indexing='ij')

    reference  = ref.reshape(xx.shape)
    prediction = pred.reshape(xx.shape)
    error      = np.abs(prediction - reference)

    kw_im = dict(cmap='jet', extent=[-2, 2, -2, 2], origin='lower', aspect='auto')

    # ---- Prediction ----
    fig0, ax0 = plt.subplots(figsize=(4.2, 3.6))
    im0 = ax0.imshow(prediction, **kw_im)
    ax0.set_xlabel('$x$', fontsize=12); ax0.set_ylabel('$y$', fontsize=12)
    ax0.set_xticks([-2, 0, 2]); ax0.set_yticks([-2, 0, 2])
    ax0.grid(True, alpha=0.3)
    fig0.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig0.savefig('spectral_bias2D_pred.pdf', dpi=300)
    plt.close(fig0)

    # ---- Reference ----
    fig1, ax1 = plt.subplots(figsize=(4.2, 3.6))
    im1 = ax1.imshow(reference, **kw_im)
    ax1.set_xlabel('$x$', fontsize=12); ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_xticks([-2, 0, 2]); ax1.set_yticks([-2, 0, 2])
    ax1.grid(True, alpha=0.3)
    fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.savefig('spectral_bias2D_ref.pdf', dpi=300)
    plt.close(fig1)

    # ---- Error (log) ----
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    im2 = ax2.imshow(error, norm=LogNorm(vmin=1e-6, vmax=1.2), **kw_im)
    ax2.set_xlabel('$x$', fontsize=12); ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_xticks([-2, 0, 2]); ax2.set_yticks([-2, 0, 2])
    ax2.grid(True, alpha=0.3)
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig2.savefig('spectral_bias2D_err.pdf', dpi=300)
    plt.close(fig2)

    print(f'[plot] 3 PDFs saved -> spectral_bias2D_*.pdf')





if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    train(key)
    npz_file = f'{args.datatype}_{args.network}_{args.seed}_{args.dim}.npz'
    plot_three_pdfs(npz_file)

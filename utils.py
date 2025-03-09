import numpy as np
import jax


def split_kanshape(input_dim, output_dim, shape):
    z = shape.split(',')
    features = []
    features.append(input_dim)
    for i in z:
        features.append(int(i))
    features.append(output_dim)
    return features


def normalization(interval, dim, is_normalization):
    if is_normalization == 1:
        max = interval[1] * jnp.ones(dim)
        min = interval[0] * jnp.ones(dim)
        if max != 1 or min != -1:
            mean = (max + min) / 2
            x_fun = lambda x: 2 * (x - mean) / (max - min)
        else:
            x_fun = lambda x: x
    else:
        x_fun = lambda x: x
    return x_fun

## matrix for fractional PDEs
def get_matrix_1d(alpha, N_x,interval, num_bc=2):
    weights = [1.0]
    for j in range(1, N_x):
        weights.append(weights[-1] * (j - 1 - alpha) / j)
    weights = np.stack(weights)
    int_mat = np.zeros((N_x, N_x), )
    diam = interval[-1]-interval[0]
    h = diam / (N_x - 1)  # self.geom.diam / (N_x - 1)
    for i in range(1, N_x - 1):
        # first order
        # int_mat[i, 1: i + 2] = np.flipud(weights[:(i + 1)])
        # int_mat[i, i - 1: -1] += weights[:(N_x - i)]
        # second order
        int_mat[i, 0:i+2] = np.flipud(modify_second_order(alpha=alpha, w=weights[:(i + 1)]))
        int_mat[i, i-1:] += modify_second_order(alpha=alpha, w=weights[:(N_x - i)])
        # third order
        # int_mat[i, 0:i+2] = np.flipud(self.modify_third_order(w=self.get_weight(i)))
        # int_mat[i, i-1:] += self.modify_third_order(w=self.get_weight(N_x-1-i))
    int_mat = h ** (-alpha) * int_mat

    int_mat = np.roll(int_mat, -1, 1)
    int_mat = int_mat[1:-1]
    int_mat = np.pad(int_mat, ((num_bc, 0), (num_bc, 0)))
    return int_mat

def modify_second_order(alpha, w=None):
    w0 = np.hstack(([0.0], w))
    w1 = np.hstack((w, [0.0]))
    beta = 1 - alpha / 2
    w = beta * w0 + (1 - beta) * w1
    return w

import numpy as np
import jax.numpy as jnp
import jax


def split_kanshape(input_dim, output_dim, shape):
    z = shape.split(',')
    features = []
    features.append(input_dim)
    for i in z:
        features.append(int(i))
    features.append(output_dim)
    return features


def normalization(interval, dim, is_normalization,is_t=0):
    if is_normalization == 1:
        max = interval[1] * jnp.ones(dim-is_t)
        min = interval[0] * jnp.ones(dim-is_t)
        if is_t==0:
            x_fun = lambda x: 2 * (x - mean) / (max - min)
        else:
            x_fun = lambda x: jnp.stack([2 * (x[:-1] - mean) / (max - min), x[-1]])
    else:
        x_fun = lambda x: x
    return x_fun

def normalization_by_points(x,is_normalization):
    max=x.max()
    min=x.min()
    if is_normalization==1:
        if max!=1 or min!=-1:
            mean=(max+min)/2
            x_fun=lambda x: 2*(x-mean)/(max-min)
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

def sample_points_on_square_boundary(num_pts_per_side):
    # Sample points along the top side (x=1 to x=0, y=1)
    top_coords = jnp.linspace(0, 1, num_pts_per_side)
    top = jnp.column_stack((top_coords, jnp.ones_like(top_coords)))

    # Sample points along the bottom side (x=0 to x=1, y=0)
    bottom_coords = jnp.linspace(0, 1, num_pts_per_side)
    bottom = jnp.column_stack((bottom_coords, jnp.zeros_like(bottom_coords)))

    # Sample points along the left side (x=0, y=1 to y=0)
    left_coords = jnp.linspace(0, 1, num_pts_per_side)[1:-1]
    left = jnp.column_stack((jnp.zeros_like(left_coords), left_coords))

    # Sample points along the right side (x=1, y=0 to y=1)
    right_coords = jnp.linspace(0, 1, num_pts_per_side)[1:-1]
    right = jnp.column_stack((jnp.ones_like(right_coords), right_coords))

    # Combine the points from all sides
    points = jnp.vstack((top, bottom, left, right))

    return points[:,0],points[:,1]

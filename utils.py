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


def normalization(x, is_normalization):
    max = np.array(x.max())
    min = np.array(x.min())
    if is_normalization == 1:
        if max != 1 or min != -1:
            mean = (max + min) / 2
            x_fun = lambda x: 2 * (x - mean) / (max - min)
        else:
            x_fun = lambda x: x
    else:
        x_fun = lambda x: x
    return x_fun

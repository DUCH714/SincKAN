import numpy as np
from scipy.special import ellipj, ellipkinc, ellipeinc, jn, yn, lpmv, sph_harm
from numpy import arange, exp, cos, sin, e, pi, absolute, meshgrid



def get_data(datatype):
    if datatype == 'sqrt':
        generate_data = sqrt
    elif datatype == 'bl':
        generate_data = boundary_layer
    elif datatype == 'bl_2d':
        generate_data = boundary_layer_2d
    elif datatype == 'sin_low':
        generate_data = sin_low
    elif datatype == 'sin_high':
        generate_data = sin_high
    elif datatype == 'double_exponential':
        generate_data = double_exponential
    elif datatype == 'spectral_bias':
        generate_data = spectral_bias
    elif datatype == 'spectral_bias2D':
        generate_data = spectral_bias2D
    elif datatype == 'piecewise':
        generate_data = piece_wise_function
    elif datatype == 'multi_sqrt':
        generate_data = multi_sqrt_function
    elif datatype == 'ellipj':
        generate_data = jacobian_elliptic_function
    elif datatype == 'ellipkinc':
        generate_data = incomplete_elliptic_integral_of_the_first_kind
    elif datatype == 'ellipeinc':
        generate_data = incomplete_elliptic_integral_of_the_second_kind
    elif datatype == 'jn':
        generate_data = bessel_function_of_the_first_kind
    elif datatype == 'yn':
        generate_data = bessel_function_of_the_second_kind
    elif datatype == 'lpmv':
        generate_data = associated_legendre_function_of_the_first_kind
    elif datatype == 'sph_harm01':
        generate_data = spherical_harmonics01
    elif datatype == 'sph_harm11':
        generate_data = spherical_harmonics11
    elif datatype == 'sph_harm02':
        generate_data = spherical_harmonics02
    elif datatype == 'sph_harm12':
        generate_data = spherical_harmonics12
    elif datatype == 'sph_harm22':
        generate_data = spherical_harmonics22
    elif datatype == 'fractal':
        generate_data = fractal_function
    elif datatype == 'multimodal1':
        generate_data = multimodal_function1
    elif datatype == 'multimodal2':
        generate_data = multimodal_function2
    elif datatype == 'multimodal3':
        generate_data = multimodal_function3
    elif datatype == 'multimodal4':
        generate_data = multimodal_function4
    elif datatype == '4D':
        generate_data = function_4D
    elif datatype == 'square4D':
        generate_data = function_square4D
    elif datatype == '100D':
        generate_data = function_100D_smooth
    elif datatype == '100D_osc':
        generate_data = function_100D_osc
    else:
        assert False, f'{datatype} does not exist'
    return generate_data

def piece_wise_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0.5
    y[mask1] = np.sin(20 * np.pi * x[mask1]) + x[mask1] ** 2
    mask2 = (0.5 <= x) & (x < 1.5)
    y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))
    mask3 = x >= 1.5
    y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])
    return y

def sqrt(x):
    y = np.zeros_like(x)
    mask1 = x < 0
    y[mask1] = 0
    mask2 = x >= 0
    y[mask2] = x[mask2]**0.5
    return y

def boundary_layer(x, alpha=100):
    y = np.exp(-x*alpha)
    return y

def boundary_layer_2d(x, y, alpha=100):
    z = np.exp(-(x + y)*alpha)
    return z

def sin_low(x):
    y = np.zeros_like(x)
    mask1 = (-1 <= x) & (x <= 1)
    y[mask1] = np.sin(4*np.pi*x[mask1])
    return y

def sin_high(x):
    y = np.zeros_like(x)
    mask = (-1 <= x) & (x <= 1)
    y[mask] = np.sin(400 * np.pi * x[mask])
    return y

def double_exponential(x):
    y = np.zeros_like(x)
    mask1 = x < 0
    y[mask1] = 0
    mask2 = (0 <= x) & (x <= 1)
    y[mask2] = (x[mask2] * (1 - x[mask2]) * e**(-x[mask2]))/(0.5**2 + (x[mask2] - 0.5)**2)
    mask3 = x > 1
    y[mask3] = 0
    return y

def spectral_bias(x):
    y = np.zeros_like(x)
    mask1 = x < -1
    y[mask1] = 0
    mask2 = (-1 <= x) & (x <= 0)
    y[mask2] = sin(x[mask2]) + sin(2 * x[mask2]) + sin(3 * x[mask2]) + sin(4 * x[mask2]) + 5
    mask3 = (0 <= x) & (x <= 1)
    y[mask3] = cos(10 * x[mask3])
    mask4 = x > 1
    y[mask4] = 0
    return y

def spectral_bias2D(X: np.ndarray) -> np.ndarray:
    """
    2-D discontinuous test function from §4.1.1.
    X : array-like, shape (..., 2)   # last axis = (x, y)
    returns array of shape X.shape[:-1]
    """
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 2:
        raise ValueError('Last dimension of X must be 2 (x and y coordinates).')

    x = X[..., 0]          # every x-coordinate
    y = X[..., 1]          # every y-coordinate

    def h(t):
        """1-D piecewise definition h(t) from the paper."""
        out = np.empty_like(t)
        mask = t < 0
        out[mask] = 5 + (np.sin(t[mask]) +
                         np.sin(2*t[mask]) +
                         np.sin(3*t[mask]) +
                         np.sin(4*t[mask]))
        out[~mask] = np.cos(10*t[~mask])
        return out

    return h(x) * h(y)     # tensor-product f(x,y)=h(x)·h(y)

def multi_sqrt_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0
    y[mask1] = 0
    mask2 = (0 <= x) & (x <= 1)
    y[mask2] = x[mask2]**0.5 * (1-x[mask2])**(3/4)
    mask3 = x > 1
    y[mask3] = 0
    return y

def jacobian_elliptic_function(x, k):
    sn, cn, dn, ph = ellipj(x, k)
    y = sn
    return y

def incomplete_elliptic_integral_of_the_first_kind(x, k):
    y = ellipkinc(x, k)
    return y

def incomplete_elliptic_integral_of_the_second_kind(x, k):
    y = ellipeinc(x, k)
    return y

def bessel_function_of_the_first_kind(x, n):
    y = jn(n, x)
    return y

def bessel_function_of_the_second_kind(x, n):
    y = yn(n, x)
    return y

def associated_legendre_function_of_the_first_kind(x, n):
    y = lpmv(1, n, x)
    return y

def spherical_harmonics01(theta, phi):
    #theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    #phi: Polar (colatitudinal) coordinate; must be in [0, pi]
    l = 1
    m = 0

    y = sph_harm(m, l, theta, phi).real
    return y

def spherical_harmonics11(theta, phi):
    # theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    # phi: Polar (colatitudinal) coordinate; must be in [0, pi]
    l = 1
    m = 1

    y = sph_harm(m, l, theta, phi).real
    return y

def spherical_harmonics02(theta, phi):
    # theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    # phi: Polar (colatitudinal) coordinate; must be in [0, pi]
    l = 2
    m = 0

    y = sph_harm(m, l, theta, phi).real
    return y

def spherical_harmonics12(theta, phi):
    # theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    # phi: Polar (colatitudinal) coordinate; must be in [0, pi]
    l = 2
    m = 1

    y = sph_harm(m, l, theta, phi).real
    return y

def spherical_harmonics22(theta, phi):
    # theta: Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    # phi: Polar (colatitudinal) coordinate; must be in [0, pi]
    l = 2
    m = 2

    y = sph_harm(m, l, theta, phi).real
    return y

def fractal_function(x):
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    z = np.sin(10 * np.pi * x_1) * np.cos(10 * np.pi * x_2) + np.sin(np.pi * np.sum(x ** 2, axis=1))
    z += np.abs(x_1 - x_2) + (np.sin(5 * x_1 * x_2) / (0.1 + np.abs(x_1 + x_2)))
    z *= np.exp(-0.1 * np.sum(x ** 2,axis=1))
    return z

def multimodal_function1(x):
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    z = -np.abs(np.sin(x_1) * np.cos(x_2) * np.exp(np.abs(1 - (np.sqrt(np.sum(x ** 2, axis=1)) / np.pi))))
    return z[:, None]  # Reshape to column vector


def multimodal_function2(x, alpha):
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    z = (-20.0 * np.exp(-0.2 * np.sqrt(alpha * np.sum(x ** 2, axis=1))) -
         np.exp(alpha * (np.cos(2 * np.pi * x_1) + np.cos(2 * np.pi * x_2))) +
         np.e + 20)
    return z[:, None]

def multimodal_function3(x):
    x_1 = x[:,0]
    x_2 = x[:,1]
    x_3 = x[:,2]
    x_4 = x[:,3]
    z = np.exp(np.sin(100 * (x_1 ** 2 + x_2 ** 2)) + np.sin(100 * (x_3 ** 2 + x_4 ** 2)))

    return z[:, None]

def multimodal_function4(x):
    x_1 = x[:,0]
    x_2 = x[:,1]
    z = np.cos(100 * (np.sin(np.pi * x_1) + x_2 ** 2))
    return z[:,None]

def function_4D(x, alpha): # x is of batch size * 4  alpha = 0.5
    # Split the input into two pairs of coordinates: (x1, x2) and (x3, x4)
    x1_x2 = x[:, :2]  # First two columns
    x3_x4 = x[:, 2:]  # Last two columns

    # Compute the function using the given formula
    z = np.exp(alpha * np.sin(np.pi * np.sum(x1_x2 ** 2, axis=1)) +
               alpha * np.sin(np.pi * np.sum(x3_x4 ** 2, axis=1)))[:, None]

    return z

def function_square4D(x):
    x_1 = x[:, 0]
    x_2 = x[:, 1]
    x_3 = x[:, 2]
    x_4 = x[:, 3]

    z = np.sin(100 * np.sqrt((x_1 - x_2) ** 2 + (x_3 - x_4) ** 2))[:,None]

    return z


def function_100D_smooth(x, alpha): #used alpha = 0.001
    print("x type:", type(x))
    print("x content:", x)

    z = np.exp(-alpha * np.sum(x ** 2,axis=1))[:,None] #[:, None] is used to convert a 1D array with shape (1000,) into a 2D column vector with shape (1000, 1). This is done to ensure that the array has the correct shape for subsequent operations, especially when performing element-wise multiplication and broadcasting.

    return z

def function_100D_osc(x, alpha):
    print("x type:", type(x))
    print("x content:", x)

    z = np.exp(-alpha * np.sum(np.sin(x ** 2),axis=1))[:,None] #[:, None] is used to convert a 1D array with shape (1000,) into a 2D column vector with shape (1000, 1). This is done to ensure that the array has the correct shape for subsequent operations, especially when performing element-wise multiplication and broadcasting.

    return z




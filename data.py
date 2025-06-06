import numpy as np
import scipy
from scipy.special import ellipj, ellipkinc, ellipeinc, jn, yn, lpmv, sph_harm, gamma
from numpy import arange, exp, cos, sin, e, pi, absolute, meshgrid


def get_data(datatype):
    if datatype == '100D':
        generate_data = function_100D
    elif datatype == '4D':
        generate_data = function_4D
    elif datatype == 'allen_cahn':
        generate_data = allen_cahn
    elif datatype == 'bl':
        generate_data = boundary_layer
    elif datatype == 'bl2d':
        generate_data = boundary_layer2d
    elif datatype == 'burgers_1d':
        generate_data = burgers_1d
    elif datatype == 'cdiff':
        generate_data = cdiff
    elif datatype == 'double_exponential':
        generate_data = double_exponential
    elif datatype == 'ellipeinc':
        generate_data = incomplete_elliptic_integral_of_the_second_kind
    elif datatype == 'ellipkinc':
        generate_data = incomplete_elliptic_integral_of_the_first_kind
    elif datatype == 'ellipj':
        generate_data = jacobian_elliptic_function
    elif datatype == 'endpoint':
        generate_data = endpoint_singularity_function
    elif datatype == 'fractal':
        generate_data = fractal_function
    elif datatype == 'fraction':
        generate_data = fraction
    elif datatype == 'jn':
        generate_data = bessel_function_of_the_first_kind
    elif datatype == 'lpmv':
        generate_data = associated_legendre_function_of_the_first_kind
    elif datatype == 'multimodal1':
        generate_data = multimodal_function1
    elif datatype == 'multimodal2':
        generate_data = multimodal_function2
    elif datatype == 'multi_sqrt':
        generate_data = multi_sqrt_function
    elif datatype == 'nonlinear':
        generate_data = nonlinear
    elif datatype == 'ns_tg':
        generate_data = ns_tg
    elif datatype == 'pbl':
        generate_data = pbl
    elif datatype == 'piecewise':
        generate_data = piece_wise_function
    elif datatype == 'poisson':
        generate_data = poisson
    elif datatype == 'poisson_sin':
        generate_data = poisson_sin
    elif datatype == 'schrodinger':
        generate_data = schrodinger
    elif datatype == 'sine_gordon':
        generate_data = sine_gordon
    elif datatype == 'sin_high':
        generate_data = sin_high
    elif datatype == 'sin_low':
        generate_data = sin_low
    elif datatype == 'spectral_bias':
        generate_data = spectral_bias
    elif datatype == 'sph_harm01':
        generate_data = spherical_harmonics01
    elif datatype == 'sph_harm02':
        generate_data = spherical_harmonics02
    elif datatype == 'sph_harm11':
        generate_data = spherical_harmonics11
    elif datatype == 'sph_harm12':
        generate_data = spherical_harmonics12
    elif datatype == 'sph_harm22':
        generate_data = spherical_harmonics22
    elif datatype == 'sqrt':
        generate_data = sqrt
    elif datatype == 'singular_frac':
        generate_data = singular_frac
    elif datatype == 't_nonlinear':
        generate_data = t_nonlinear
    elif datatype == 'yn':
        generate_data = bessel_function_of_the_second_kind
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
    y[mask2] = x[mask2] ** 0.5
    return y


def boundary_layer(x, alpha=100):
    y = np.exp(-x * alpha)
    return y


def boundary_layer2d(x, y, alpha=100):
    y = np.exp(-x * alpha) + np.exp(-y * alpha)
    return y


def endpoint_singularity_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0
    y[mask1] = 0
    mask2 = (0 <= x) & (x <= 1)
    y[mask2] = x[mask2] ** 0.5 * (1 - x[mask2]) ** (3 / 4)
    mask3 = x > 1
    y[mask3] = 0
    return y


def sin_low(x):
    y = np.zeros_like(x)
    mask1 = (-1 <= x) & (x <= 1)
    y[mask1] = np.sin(4 * np.pi * x[mask1])
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
    y[mask2] = (x[mask2] * (1 - x[mask2]) * e ** (-x[mask2])) / (0.5 ** 2 + (x[mask2] - 0.5) ** 2)
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


def multi_sqrt_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0
    y[mask1] = 0
    mask2 = (0 <= x) & (x <= 1)
    y[mask2] = x[mask2] ** 0.5 * (1 - x[mask2]) ** (3 / 4)
    mask3 = x > 1
    y[mask3] = 0
    return y


def jacobian_elliptic_function(x, k=0.5):
    sn, cn, dn, ph = ellipj(x, k)
    y = sn
    return y


def incomplete_elliptic_integral_of_the_first_kind(x, k=0.5):
    y = ellipkinc(x, k)
    return y


def incomplete_elliptic_integral_of_the_second_kind(x, k=0.5):
    y = ellipeinc(x, k)
    return y


def bessel_function_of_the_first_kind(x, n=3):
    y = jn(n, x)
    return y


def bessel_function_of_the_second_kind(x, n=3):
    y = yn(n, x)
    return y


def associated_legendre_function_of_the_first_kind(x, n=3):
    y = lpmv(1, n, x)
    return y


def spherical_harmonics01(theta):
    l = 1
    m = 0
    phi = 0
    y = sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics11(theta):
    l = 1
    m = 1
    phi = 0
    y = sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics02(theta):
    l = 2
    m = 0
    phi = 0
    y = sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics12(theta):
    l = 2
    m = 1
    phi = 0
    y = sph_harm(m, l, phi, theta).real
    return y


def spherical_harmonics22(theta):
    l = 2
    m = 2
    phi = 0
    y = sph_harm(m, l, phi, theta).real
    return y


def fractal_function(x, y):
    z = np.sin(10 * np.pi * x) * np.cos(10 * np.pi * y) + np.sin(np.pi * (x ** 2 + y ** 2))
    z += np.abs(x - y) + (np.sin(5 * x * y) / (0.1 + np.abs(x + y)))
    z *= np.exp(-0.1 * (x ** 2 + y ** 2))
    return z


def multimodal_function1(x, y):
    z = -absolute(sin(x) * cos(y) * exp(absolute(1 - (np.sqrt(x ** 2 + y ** 2) / pi))))
    return z


def multimodal_function2(x, y):
    z = -20.0 * exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20
    return z


def function_4D(x1, x2, x3, x4):
    z = np.exp(0.5 * np.sin(np.pi * (x1 ** 2 + x2 ** 2)) + 0.5 * np.sin(np.pi * (x3 ** 2 + x4 ** 2)))
    return z


def function_100D(x):
    x = np.asarray(x)
    if len(x) != 100:
        raise ValueError("Input should be a 100-dimensional vector.")
    z = exp(0.01 * np.sum(sin(pi * x / 2) ** 2))
    return z


def pbl(x, alpha=100):
    eps = 1 / alpha
    z = 1 + x + (np.exp(x / eps) - 1) / (np.exp(1 / eps) - 1)
    return z


def nonlinear(x):
    z = x ** (5 / 2) * (1 - x) ** 2 + x ** 3 + 1
    return z


def burgers_1d(x, t, a=0.1, nu=0.01):
    z = a / 2 - a / 2 * np.tanh(a * (x - a * t / 2) / 4 / nu)
    return z


def ns_tg(x, y, t, nu, k=1):
    u = -np.cos(k * x) * np.sin(k * y) * np.exp(-2 * t * nu)
    v = np.sin(k * x) * np.cos(k * y) * np.exp(-2 * t * nu)
    p = -(np.cos(2 * k * x) + np.sin(2 * k * y)) * np.exp(-4 * t * nu) / 4
    return u, v, p


def t_nonlinear(x, t):
    z = np.cos((x + 2) * (t + 1))
    return z


def cdiff(x, t, a, eps, N=6):
    Z = 0
    for k in range(N):
        Z = Z + np.sin(k * (x - a * t)) * np.exp(-eps * k ** 2 * t)
    return Z


def poisson(x, alpha):
    y = np.exp(-alpha * np.sum(x ** 2, axis=1))[:, None]
    return y


def allen_cahn(x, alpha, c):
    B = -alpha * np.sum(x ** 2, axis=1)
    return np.exp(B)[:, None]


def sine_gordon(x, alpha, c):
    A = np.mean(np.exp(-c * x[:, :-2] * x[:, 1:-1] * x[:, 2:]), axis=1)
    B = -alpha * np.sum(x ** 2, axis=1)
    return (A * np.exp(B))[:, None]


def poisson_sin(x, dim):
    temp = np.sum(x, axis=1) / dim
    y = (temp) ** 2 + np.sin(temp)
    return y[:, None]


def schrodinger(x, coeffs):
    hbar = coeffs['hbar']
    m = coeffs['m']
    omega = coeffs['omega']
    vec_s = coeffs['vec_s']
    vec_mu = coeffs['vec_mu']
    x0 = coeffs['x0']
    x = x0*x
    alpha = vec_mu - vec_s / 2
    # Associated Laguerre Polynomials for n=1
    # L = lambda x: x ** (-alpha) * np.exp(x) * (-x ** (1 + alpha) * np.exp(-x) + (1 + alpha) * x ** (alpha) * np.exp(-x))
    L = lambda x: 1 + alpha - x
    func_plus = lambda x: (np.exp(-m * omega / 2 / hbar * x ** 2) * x ** ((1 - 1) / 2) * (
                1 + vec_mu[0] - 1 / 2 - m * omega / hbar * x ** 2)) ** 2
    int_plus, _ = scipy.integrate.quad(func_plus, -np.inf, np.inf)

    func_minus = lambda x: (np.exp(-m * omega / 2 / hbar * x ** 2) * x ** ((1 + 1) / 2) * (
                1 + vec_mu[0] + 1 / 2 - m * omega / hbar * x ** 2)) ** 2
    int_minus, _ = scipy.integrate.quad(func_minus, -np.inf, np.inf)

    c_plus = np.sqrt(1 / int_plus)
    c_minus = np.sqrt(1 / int_minus)

    # c_plus = 1
    # c_minus = 1

    vec_psi = (c_plus * (vec_s == 1) + c_minus * (vec_s == -1)) * np.exp(-m * omega / 2 / hbar * x ** 2) * x ** (
                (1 - vec_s) / 2) * L(m * omega / hbar * x ** 2)

    psi = np.prod(vec_psi, axis=1)

    return psi[:, None]


def fraction(x):
    # return x * (np.abs(1 - x**2)) ** (alpha / 2)
    return x ** 3 * (1 - x) ** 3


def singular_frac(x, alpha, d=1):
    s = alpha / 2
    u = 2 ** (-2 * s) * gamma(d / 2) / gamma(d / 2 + s) / gamma(1 + s) \
        * (1 - np.sum(x ** 2, axis=1)) ** s
    return u[:, None]

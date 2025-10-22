import numpy as np
from math import pi


# =============================================================================
# define fitting functions
# x represents time axis (should be year - reference year)
# the baseline algorithm assigns first guess values to the first two
# fit paramters assuming that they are a constant offset and a linear term
# pay attention when defining new functions for it
# =============================================================================
def linear(x, a, b):
    return a + b*x


def quadratic(x, a, b, c):
    return a + b*x + c*x**2


def cubic(x, a, b, c, d):
    return a + b*x + c*x**2 + d*x**3


def simple(x, a, b, c, e, f):
    return a+b*x+c*x**2 +e*np.cos(2*pi*x) + f*np.sin(2*pi*x)


def simple_3rd(x, a, b, c, d, e, f,):
    return a+b*x+c*x**2\
                + d*x**3\
                + e*np.cos(2*pi*x)+f*np.sin(2*pi*x)
                

def higher(x, a, b, c, e, f, g, h):
    return a+b*x + c*x**2\
                + e*np.cos(2*pi*x)+f*np.sin(2*pi*x) \
                + g*np.cos(4*pi*x)+h*np.sin(4*pi*x)


def const(x, a):
    return a + 0.*x


def lin_sin(x, a, b, c, d):
    return a+b*x + c*np.sin(2*pi*x + d)


def poly(x, *poly_params):
    # p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]
    return np.polyval(poly_params[::-1], x)


# modified from ccgcrv
def harmonics(x, *harm_params):
    """"
    calculate the harmonic part of the function at time x
    length of harm_params should be an even number
    """

    if not harm_params is None:
        numharm = round(len(harm_params)/2)
    else:
        numharm=0

    if numharm > 0:
        # will not work with arrays if not np.sin/np.cos are used
        h = harm_params[0]*np.sin(2*pi*x) + harm_params[1]*np.cos(2*pi*x)

        # do additional harmonics (nharm > 1)
        for i in range(1, numharm):
            ix = 2*i            # index into harm_params for harmonic coefficients
            h += harm_params[ix]*np.sin((i+1)*2*pi*x) + harm_params[ix+1]*np.cos((i+1)*2*pi*x)

        return h

    return


# modified from ccgcrv
def poly_harm(x, *params):
    # last two entries of params should contain number of polynomial terms (incl. constant) and number of harmonics
    numpoly = params[-2]    # number of polynomial terms, incl. constant term
    numharm = params[-1]  # number of harmonic parameters, i.e. sum of all cos and sin terms, should be an even number

    poly_params = params[0:int(numpoly)]
    harm_params = params[int(numpoly):-2]   # harm_params should be 2*numharm (one cos and sin term per harmonic order)

    # get polynomial part of function
    poly_term = poly(x, *poly_params)

    # get harmonic part of function
    harm_term = harmonics(x, *harm_params)

    return poly_term + harm_term
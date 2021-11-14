import dolfin as dlfn
from dolfin import inner, Dx, div, grad, outer
import numpy as np

# Entropy pair
def entropy(J):
    """
    Returns entropy E(J)=0.5 * J^2

    :param J:
    :return:
    """
    return .5 * J * J

def entropy_flux(J, beta):
    """
    Returns entropy flux F(J)=beta*E(J)

    :param J:
    :param beta:
    :return:
    """
    return beta * entropy(J)

# State Problems
def F_spatial(sol_, del_, beta):
    """
    Return spatial operation, i.e. flux divergence, of transport equation in its weak form.

    :param sol_: TrialFunction,
    :param del_: TestFunction,
    :param beta: float, transport velocity
    :return: Form,
    """
    if sol_.ufl_shape == ():
        return inner(Dx(beta * sol_, 0), del_)
    else:
        return inner(div(outer(sol_, beta)), del_)

def F_spatial_visc(sol_, del_, beta, Nu):
    """
    Return spatial operation of transport equation with diffusion in its weak form.

    :param sol_: TrialFunction,
    :param del_: TestFunction,
    :param beta: float, transport velocity
    :param Nu: Function,
    :return: Form,
    """
    return F_spatial(sol_, del_, beta) + Nu * inner(grad(sol_), grad(del_))


def calculate_NormE(Jh, n_dofs):
    """
    Returns maximum value of entropy deviation ||E(J) - avg(E(J))||_max as FEA-approximation of infinity-norm on domain.

    :param Jh: Function, Inertia Tensor
    :return: float, maximum entropy deviation
    """
    E_var = entropy(Jh.vector()) - entropy(Jh.vector()).sum() / n_dofs
    return np.sqrt(E_var * E_var).max()


def min_func(f1, f2):
    """
    Return minimum of two functions.

    :param f1:
    :param f2:
    :return:
    """
    return (f1+f2-abs(f1-f2))/dlfn.Constant(2.)

def macaulay_func(f1, f2):
    return (f1-f2+abs(f1-f2))/dlfn.Constant(2.)

class PeriodicBoundary(dlfn.SubDomain):
    def __init__(self, l):
        dlfn.SubDomain.__init__(self)
        self.l = l

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < dlfn.DOLFIN_EPS and x[0] > -dlfn.DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - self.l

class Initial_Condition(dlfn.UserExpression):
    def __init__(self, degree, l, offset=0):
        self._offset = offset
        self._l = l
        super().__init__(degree=degree)
    def eval(self, values, x):
        x_ = (x[0] - self._offset) % self._l
        if abs(2 * x_ - .3) <= 0.25 + dlfn.DOLFIN_EPS:
            values[0] = dlfn.exp(-300 * (2 * x_ - .3) ** 2)
        elif abs(2 * x_ - .9) <= 0.2 + dlfn.DOLFIN_EPS:
            values[0] = 1
        elif abs(2 * x_ - 1.6) <= 0.2:
            values[0] = dlfn.sqrt(1 - ((2 * x_ - 1.6) / .2) ** 2)
        else:
            values[0] = 0

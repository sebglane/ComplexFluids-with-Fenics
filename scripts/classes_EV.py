import os.path

import dolfin as dlfn
from dolfin import dot, inner, grad, div, as_vector, Dx, sqrt, cross, bessel_I
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
    Return spatial derivative of transport equation in its weak form.

    :param sol_: TrialFunction,
    :param del_: TestFunction,
    :param beta: float, transport velocity
    :return: Form,
    """
    return inner(Dx(beta * sol_, 0), del_)

def F_spatial_visc(sol_, del_, beta, Nu):
    """
    Return spatial derivative of transport equation with diffusion in its weak form.

    :param sol_: TrialFunction,
    :param del_: TestFunction,
    :param beta: float, transport velocity
    :param Nu: Function,
    :return: Form,
    """
    return F_spatial(sol_, del_, beta) + Nu * Dx(sol_, 0) * Dx(del_, 0)

def get_max_local(function, degree_from, functionspace_from, functionspace_to, continuous=True, periodic=True):
    """
    Return maximum of function on cell ordered according to functionspace_to. Only suitable for 1D-Problems.

    :param function: Function,
    :param degree_from: int,
    :param functionspace_from: FunctionSpace,
    :param functionspace_to: FunctionSpace,
    :param continuous: boolean,
    :param periodic: boolean,
    :return: np.array
    """

    dof_arr_from = functionspace_from.tabulate_dof_coordinates().T
    dof_arr_to = functionspace_to.tabulate_dof_coordinates().T
    dof_args_from = np.argsort(dof_arr_from[0]) # Mapping Functionspace_from to ordered
    dof_args_to = np.argsort(np.argsort(dof_arr_to[0])) # Mapping ordered to Functionspace_from
    bool_continuous, bool_periodic = 0, 0
    if continuous:
        bool_continuous = 1
    if periodic:
        bool_periodic = 1

    locals_arr = function.vector().get_local()
    locals_arr_sorted = locals_arr[dof_args_from]
    iteration_step = degree_from + 1 - bool_continuous
    n_locals = (len(locals_arr) + bool_periodic - 2 + bool_continuous) // degree_from
    # FIXME: Probably inefficient; faster solution in Python/C++?
    max_local = np.fromiter(
        (max(abs(locals_arr_sorted[j * iteration_step: (j + 1) * iteration_step + 1])) for j in range(n_locals))
        , float
    )
    return max_local[dof_args_to]

def min_func(f1, f2):
    """
    Return minimum of two functions. Exact for degree=0.

    :param f1:
    :param f2:
    :return:
    """
    return (f1+f2-abs(f1-f2))/dlfn.Constant(2.)

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
    def __init__(self, degree, offset=0):
        self._offset = offset
        super().__init__(degree=degree)
    def eval(self, values, x):
        x_ = x[0] + self._offset
        if abs(2 * x_ - .3) <= 0.25 + dlfn.DOLFIN_EPS:
            values[0] = dlfn.exp(-300 * (2 * x_ - .3) ** 2)
        elif abs(2 * x_ - .9) <= 0.2 + dlfn.DOLFIN_EPS:
            values[0] = 1
        elif abs(2 * x_ - 1.6) <= 0.2:
            values[0] = dlfn.sqrt(1 - ((2 * x_ - 1.6) / .2) ** 2)
        else:
            values[0] = 0

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

def calculate_NormE(Jh):
    """
    Returns maximum value of entropy deviation ||E(J) - avg(E(J))||_max as FEA-approximation of infinity-norm on domain.

    :param Jh: Function, Inertia Tensor
    :return: float, maximum entropy deviation
    """
    fspace = Jh.function_space()
    n_dofs = fspace.dim()
    NormE = dlfn.project(
        dlfn.sqrt((entropy(Jh) - dlfn.project(entropy(Jh), fspace).vector().sum() / n_dofs) ** 2),
        fspace
    ).vector().max()

    return NormE

def calculate_NormD(D_norm, Jh, Jh_, Jh_n, dt, beta, degree):
    """
    Calculate cell-wise maximum of entropy residual as FEA-approximation of infinity-norm on the elements.

    :param D_norm: Function, assign result to this function
    :param Jh: Function,
    :param Jh_: Function,
    :param Jh_n: Function,
    :param dt: float, size of time-step
    :param beta: float, transport velocity
    :param degree: int, polynomial degree of J
    :return: None
    """
    fspace_J = Jh.function_space()
    fspace_D = D_norm.function_space()

    D = dlfn.project(
        (.5 * (3 * entropy(Jh) - 4 * entropy(Jh_) + entropy(Jh_n)) / dt
         + Dx(entropy_flux(Jh, beta), 0)),
        fspace_J
    )
    D_norm.vector().set_local(get_max_local(D, degree, fspace_J, fspace_D))
    return

def calculate_nu(Nu_h, D_norm, E_norm, Nu_max, c_e, h_k):
    """
    Calculate viscosity nu according to Entropy-Viscosity-Method.

    :param Nu_h: Function, assign result to this function
    :param D_norm: Function, cell-wise maximum of entropy residual
    :param E_norm: float, maximum value of entropy deviation
    :param Nu_max: float, maximum value of Nu_h
    :param c_e: float, tunable parameter in EV-Method
    :param h_k: float, spatial step size
    :return: None
    """
    fspace = D_norm.function_space()
    Nu_E = dlfn.project(c_e * h_k ** 2 * D_norm / E_norm, fspace)
    dlfn.assign(Nu_h, dlfn.project(min_func(Nu_E, Nu_max), fspace))
    return

def min_func(f1, f2):
    """
    Return minimum of two functions.

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

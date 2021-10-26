import os.path

import dolfin as dlfn
from entropy_viscosity_cpp import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters ==================================
# Polynomial Degree
p_deg = 3

# Domain
nx = 4	# discretization points in x-direction
l = 2.
h_k = l / nx

# == Mesh ========================================
mesh = dlfn.IntervalMesh(int(nx), 0, l)
space_dim = mesh.geometry().dim()
n_cells = mesh.num_cells()

# == Initial Condition ============================
class Initial_Condition(dlfn.UserExpression):
    def __init__(self, offset=0, degree=2):
        self._offset = offset
        super().__init__(degree=degree)
    def eval(self, values, x):
        if x[0] < 1. + dlfn.DOLFIN_EPS:
            values[0] = -4 * x[0] ** 2 + 5 * x[0]
        else:
            values[0] = 1

# == Element Formulation =========================
c = mesh.ufl_cell()



Wh = dlfn.FunctionSpace(mesh, "CG", p_deg)
Vh = dlfn.FunctionSpace(mesh, "DG", 0)
Ph = dlfn.FunctionSpace(mesh, "DG", p_deg)

n_dofs = Wh.dim()
J = dlfn.Function(Wh)
test_fun = dlfn.Function(Vh)


del_ = dlfn.TestFunction(Ph)
sol_ = dlfn.TrialFunction(Ph)
sol_h = dlfn.Function(Ph)


dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)
n = dlfn.FacetNormal(mesh)
#dA = dlfn.Measure("ds", domain=mesh, subdomain_data=facet_marker)

dlfn.assign(J, dlfn.interpolate(Initial_Condition(), Wh))
dlfn.assign(test_fun, dlfn.interpolate(Initial_Condition(), Vh))
nu = StabilizationParameterSD(J, test_fun)

F = (nu * del_ - sol_ * del_) * dx
dlfn.solve(dlfn.lhs(F)==dlfn.rhs(F), sol_h)

x_plot = np.linspace(0, l, 200)

y_plot_0 = np.fromiter((sol_h(x_i) for x_i in x_plot), float)
y_plot_1 = np.fromiter((J(x_i) for x_i in x_plot), float)
y_plot_2 = np.fromiter((test_fun(x_i) for x_i in x_plot), float)

plt.plot(x_plot, y_plot_1)
plt.plot(x_plot, y_plot_2)
plt.plot(x_plot, y_plot_1+y_plot_2)
plt.plot(x_plot, y_plot_0)

plt.legend(['$f_1$', '$f_2$', '$f_1+f_2$', '$max_K(f_1+f_2)$'])

plt.savefig('results/test_JIT.png')
#vtkfile = dlfn.File('results/min_max.pvd')
#vtkfile << dlfn.project(nu, Ph)

import os.path

import dolfin as dlfn
from dolfin import dot, inner, grad, div, as_vector, Dx, sqrt, cross, bessel_I, lhs, rhs
import numpy as np
import classes_EV
from entropy_viscosity_cpp import *

# Parameters ==================================
# Polynomial Degree
p_deg = 2

# Domain
nx = 100	# discretization points in x-direction
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
        x_ = x[0] + self._offset
        if x_ < 1. + dlfn.DOLFIN_EPS:
            values[0] = x_ ** 2
        else:
            values[0] = 1.

# == Element Formulation =========================
c = mesh.ufl_cell()



Wh = dlfn.FunctionSpace(mesh, "CG", p_deg)
Vh = dlfn.FunctionSpace(mesh, "DG", 0)

n_dofs = Wh.dim()
J = dlfn.Function(Wh)
test_fun = dlfn.Function(Vh)

dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)
n = dlfn.FacetNormal(mesh)
#dA = dlfn.Measure("ds", domain=mesh, subdomain_data=facet_marker)

dlfn.assign(J, dlfn.interpolate(Initial_Condition(), Wh))
dlfn.assign(test_fun, dlfn.interpolate(Initial_Condition(), Vh))

nu = StabilizationParameterSD(J, J)
#J_pc = J + delta*inner(dot(grad(u), u_), dot(grad(v), u_))*dx
vtkfile = dlfn.File('results/min_max.pvd')
vtkfile << dlfn.interpolate(nu, Vh)

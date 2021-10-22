import matplotlib.pyplot as plt
from classes_EV import *

# Parameters ==================================
# Polynomial Degree
p_deg = 3

# Domain
nx = 3
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
test_fun = dlfn.Function(Vh)

dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

def min_func(f1, f2):
    return (f1+f2-abs(f1-f2))/dlfn.Constant(2.)

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)
n = dlfn.FacetNormal(mesh)
#dA = dlfn.Measure("ds", domain=mesh, subdomain_data=facet_marker)

fun = Initial_Condition()
fun_proj = dlfn.interpolate(fun, Wh)
max_locals = get_max_local(fun_proj, p_deg, Wh, Vh)
test_fun.vector().set_local(min_func(max_locals, .7))

x = np.linspace(0, l, p_deg*nx)
y = np.fromiter((test_fun(x_i) for x_i in x), float)

dlfn.plot(test_fun)
plt.savefig('results/test_max_locals.png')


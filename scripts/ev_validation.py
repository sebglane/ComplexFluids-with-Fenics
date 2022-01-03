import dolfin as dlfn
from stabilization import *
import numpy as np
import matplotlib.pyplot as plt

dlfn.parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

def maximum(func, cell_i, dim):
    """
    Return maximum of function func on cell cell_i. Only for validation of stabilization.py, won't work in parallel.
    """
    V = func.function_space()

    dmap = np.reshape(cell_i.get_coordinate_dofs(), (-1, dim))
    coords = V.tabulate_dof_coordinates()

    idx = np.fromiter((np.sum(np.product(coord==dmap, axis=1), axis=0) for coord in coords), dtype=bool)
    return np.max(np.abs(func.vector().get_local()[idx]))

def ev(J_n, J_n_, J_n_n, v, dt, h_, mesh_):
    """
    Computes the continuous entropy residual for the special case
        * c_e = 1.
        * Nu_max -> infinity

    This is only suitable if E(J)=.5 J^2 is of the same polynomial degree as the function space of J.
    """
    nu = .25 / dt * (3 * dlfn.inner(J_n, J_n) - 4 * dlfn.inner(J_n_, J_n_) + dlfn.inner(J_n_n, J_n_n))\
         + dlfn.inner(J_n, dlfn.dot(dlfn.grad(J_n), v))
    return dlfn.project(h_ ** 2 * nu, dlfn.FunctionSpace(mesh_, 'CG', 2))

#mesh = dlfn.UnitCubeMesh(5, 10, 20)
x_0 = np.array([0., 0., 0.])
x_1 = np.array([1., 1., 1.])
nx, ny, nz = 5, 5, 5

meshes = [dlfn.IntervalMesh(nx, x_0[0], x_1[0]),
          dlfn.RectangleMesh(dlfn.Point(x_0[:2]), dlfn.Point(x_1[:2]), nx, ny),
          dlfn.BoxMesh(dlfn.Point(x_0), dlfn.Point(x_1), nx, ny, nz)]

x_diff = x_1 - x_0
h_min = [x_diff[0] / nx,
         min(x_diff[0] / nx, x_diff[1] / ny),
         min(x_diff[0] / nx, x_diff[1] / ny, x_diff[2] / nz)]

c_max = 10000000.

# Expressions
# Higher degree expressions for J lead to lower accuracy, probably due to approximation error in projection
J_expr = [dlfn.Expression((('x[0]',),), degree=2),
          dlfn.Expression((('x[0]', 'x[0] - x[1]'), ('x[0] - x[1]', 'x[1]')), degree=2),
          dlfn.Expression((('x[0]', 'x[0] - x[1]', 'x[2] + x[0]'),
                           ('x[0] - x[1]', 'x[1]', 'x[2] + x[1] + x[0]'),
                           ('x[2] + x[0]', 'x[2] + x[1] + x[0]', 'x[2]')), degree=2)]


beta_const_expr = [dlfn.Expression(('1.',), degree=1),
                   dlfn.Expression(('1.', '1.'), degree=1),
                   dlfn.Expression(('1.', '1.', '1.'), degree=1)]

beta_linear_expr = [dlfn.Expression(('2.',), degree=1),
                    dlfn.Expression(('x[0]', '-x[1]'), degree=1),
                    dlfn.Expression(('x[0]', 'x[1]', '-2. * x[2]'), degree=1)]

coords_expr = [dlfn.Expression((('x[0]',),), degree=1),
               dlfn.Expression((('x[0] + x[1]', 0.), (0., 0.)), degree=1),
               dlfn.Expression((('x[0] + x[1] + x[2]', 0., 0.), (0., 0., 0.), (0., 0., 0.)), degree=1)]

# EV-parameters
E_norm = dlfn.Constant(1.)  # Given normalized entropy variation
c_e = 1.
dt = dlfn.Constant(.25)  # time step size
time_step = dlfn.Constant(10)  # pass time_step > 1 to compute entropy-viscosity with second order backwards differences

degree = 2

# Relative error lists
median_arr = np.zeros((3, 4))
var_arr = np.zeros((3, 4))

for i, mesh in enumerate(meshes):
    dim = mesh.geometric_dimension()
    c = mesh.ufl_cell()


    Vh = dlfn.VectorFunctionSpace(mesh, 'DG', 1)

    Jh = dlfn.TensorFunctionSpace(mesh, 'CG', degree)

    Jh_test = dlfn.FunctionSpace(mesh, 'DG', 0)

    n_dofs = Jh.dim()

    # Zero and one tensor (e_i \otimes e_j)
    zero = dlfn.Constant([[dlfn.Constant(0.) for _ in range(dim)] for __ in range(dim)])

    zero_tensor = dlfn.interpolate(dlfn.as_tensor(zero), Jh)

    # Projections of Expressions
    beta_const = dlfn.project(beta_const_expr[i], Vh)
    beta_linear = dlfn.project(beta_const_expr[i], Vh)

    coords = dlfn.interpolate(coords_expr[i], Jh)

    J = dlfn.project(J_expr[i], Jh)
    J_old = dlfn.project(J_expr[i], Jh)
    J_old_old = dlfn.project(J_expr[i], Jh)

    # Analytical solutions
    Nu_projected = [ev(J, zero_tensor, zero_tensor, beta_const, dt, h_min[i], mesh),
                    ev(zero_tensor, J_old, zero_tensor, beta_const, dt, h_min[i], mesh),
                    ev(zero_tensor, zero_tensor, J_old_old, beta_const, dt, h_min[i], mesh),
                    ev(coords, zero_tensor, zero_tensor, beta_linear, dt, h_min[i], mesh)]

    # Nu cases
    Nu_computed = [StabilizationParameter(J, zero_tensor, zero_tensor, beta_const, c_e, c_max, dt, E_norm, time_step),
                   StabilizationParameter(zero_tensor, J_old, zero_tensor, beta_const, c_e, c_max, dt, E_norm, time_step),
                   StabilizationParameter(zero_tensor, zero_tensor, J_old_old, beta_const, c_e, c_max, dt, E_norm, time_step),
                   StabilizationParameter(coords, zero_tensor, zero_tensor, beta_linear, c_e, c_max, dt, E_norm, time_step)]

    # Projection of Nu cases
    Nu_computed_projected = [dlfn.project(nu, Jh_test) for nu in Nu_computed]

    J_list = []
    J_old_list = []
    J_old_old_list = []
    beta_list = []

    # Iterate over cells in given mesh: will not work in parallel and is only supposed for validating stabilization.py
    for cell in dlfn.cells(mesh):
        x_i = np.mean(np.reshape(cell.get_coordinate_dofs(), (-1, dim)), axis=0)
        vals = np.zeros(1)

        Nu_computed_projected[0].eval(vals, x_i)
        J_list.append(abs(vals[0]-maximum(Nu_projected[0], cell, dim))/(abs(vals[0])))
        assert abs(vals[0] - maximum(Nu_projected[0], cell, dim)) / abs(vals[0]) < .1

        Nu_computed_projected[1].eval(vals, x_i)
        J_old_list.append(abs(vals[0] - maximum(Nu_projected[1], cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_projected[1], cell, dim)) / abs(vals[0]) < .1

        Nu_computed_projected[2].eval(vals, x_i)
        J_old_old_list.append(abs(vals[0] - maximum(Nu_projected[2], cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_projected[2], cell, dim)) / abs(vals[0]) < .1

        Nu_computed_projected[3].eval(vals, x_i)
        beta_list.append(abs(vals[0] - maximum(Nu_projected[3], cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_projected[3], cell, dim)) / abs(vals[0]) < .1

    median_arr[i][0] = np.median(J_list)
    median_arr[i][1] = np.median(J_old_list)
    median_arr[i][2] = np.median(J_old_old_list)
    median_arr[i][3] = np.median(beta_list)

    var_arr[i][0] = np.var(J_list)
    var_arr[i][1] = np.var(J_old_list)
    var_arr[i][2] = np.var(J_old_old_list)
    var_arr[i][3] = np.var(beta_list)

    print(f"{dim}D-validation successful!")

print("Validation successful!")

# Plot relative error
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

im1 = ax[0].imshow(median_arr, cmap='inferno', interpolation='nearest')
im2 = ax[1].imshow(var_arr, cmap='inferno', interpolation='nearest')


ax[0].set_yticks(np.arange(3))
ax[0].set_yticklabels(["1D", "2D", "3D"])

ax[0].set_xticks(np.arange(4))
ax[0].set_xticklabels(["J", "J_old", "J_old_old", "beta"])

ax[0].title.set_text('Median')

ax[1].set_yticks(np.arange(3))
ax[1].set_yticklabels(["1D", "2D", "3D"])

ax[1].set_xticks(np.arange(4))
ax[1].set_xticklabels(["J", "J_old", "J_old_old", "beta"])

ax[1].title.set_text('Variance')

textcolors = ("black", "white")
threshold = (np.mean(median_arr), np.mean(var_arr))
for i in range(3):
    for j in range(4):
        text1 = ax[0].text(j, i, "{:.2e}".format(median_arr[i, j]),
                           ha="center", va="center", color=textcolors[median_arr[i, j] < threshold[0]])
        text2 = ax[1].text(j, i, "{:.2e}".format(var_arr[i, j]),
                           ha="center", va="center", color=textcolors[var_arr[i, j] < threshold[1]])

plt.savefig('results/rel_err.png')

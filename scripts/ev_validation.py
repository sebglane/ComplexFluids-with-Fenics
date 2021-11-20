import dolfin as dlfn
from stabilization import *
import numpy as np
import matplotlib.pyplot as plt

dlfn.parameters['form_compiler']['no-evaluate_basis_derivatives'] = False

def maximum(func, cell_i, dim):
    V = func.function_space()

    dmap = np.reshape(cell_i.get_coordinate_dofs(), (-1, dim))
    coords = V.tabulate_dof_coordinates()

    idx = np.fromiter((np.sum(np.product(coord==dmap, axis=1), axis=0) for coord in coords), dtype=bool)
    return np.max(np.abs(func.vector().get_local()[idx]))

def ev(J_n, J_n_, J_n_n, v, dt, h_, mesh_):
    nu = .25 / dt * (3 * J_n**2 - 4 * J_n_**2 + J_n_n**2) + dlfn.dot(dlfn.grad(.5 * J_n**2), v)
    return dlfn.project(h_ ** 2 * nu, dlfn.FunctionSpace(mesh_, 'CG', 2))

#mesh = dlfn.UnitCubeMesh(5, 10, 20)
x_0 = np.array([0., 0., 0.])
x_1 = np.array([.0125, .0125, .0125])
nx, ny, nz = 1, 1, 1

meshes = [dlfn.IntervalMesh(nx, x_0[0], x_1[0]),
        dlfn.RectangleMesh(dlfn.Point(x_0[:2]), dlfn.Point(x_1[:2]), nx, ny),
        dlfn.BoxMesh(dlfn.Point(x_0), dlfn.Point(x_1), nx, ny, nz)]

x_diff = x_1 - x_0
h_min = [x_diff[0] / nx,
         min(x_diff[0] / nx, x_diff[1] / ny),
         min(x_diff[0] / nx, x_diff[1] / ny, x_diff[2] / nz)]

Nu_max = 150.

# Expressions
# Higher degree expressions for J lead to lower accuracy, probably due to approximation error in projection
J_expr = [dlfn.Expression('x[0]', degree=2),
          dlfn.Expression('x[0] + x[1]', degree=2),
          dlfn.Expression('x[0] + x[1] + x[2]', degree=2)]

J_old_expr = [dlfn.Expression('x[0]', degree=1),
              dlfn.Expression('x[0] + x[1]', degree=1),
              dlfn.Expression('x[0] + x[1] + x[2]', degree=1)]

J_old_old_expr = [dlfn.Expression('-x[0]', degree=1),
                  dlfn.Expression('-x[1]', degree=1),
                  dlfn.Expression('-x[2]', degree=1)]

beta_const_expr = [dlfn.Expression(('1.',), degree=1),
                   dlfn.Expression(('1.', '1.'), degree=1),
                   dlfn.Expression(('1.', '1.', '1.'), degree=1)]

beta_linear_expr = [dlfn.Expression(('2.',), degree=1),
                    dlfn.Expression(('x[0]', '-x[1]'), degree=1),
                    dlfn.Expression(('x[0]', 'x[1]', '-2. * x[2]'), degree=1)]

coords_expr = [dlfn.Expression('x[0]', degree=1),
               dlfn.Expression('x[0] + x[1]', degree=1),
               dlfn.Expression('x[0] + x[1] + x[2]', degree=1)]

# EV-parameters
E_norm = dlfn.Constant(1.)
c_e = 1.
dt = dlfn.Constant(.25)
t_step = dlfn.Constant(10)

degree = 2

# Relative error lists
median_arr = np.zeros((3, 4))
var_arr = np.zeros((3, 4))

# TODO: Test for higher-rank J
for i, mesh in enumerate(meshes):
    dim = i + 1
    c = mesh.ufl_cell()


    Vh = dlfn.VectorFunctionSpace(mesh, 'DG', 1)

    Jh = dlfn.FunctionSpace(mesh, 'CG', degree)

    Jh_test = dlfn.FunctionSpace(mesh, 'DG', 0)

    n_dofs = Jh.dim()


    # Zero and one
    zero_scalar = dlfn.interpolate(dlfn.Constant(0.), Jh)
    one_scalar = dlfn.interpolate(dlfn.Constant(1.), Jh)

    # Projections of Expressions
    beta_const = dlfn.project(beta_const_expr[i], Vh)
    beta_linear = dlfn.project(beta_const_expr[i], Vh)

    coords = dlfn.project(coords_expr[i], Jh)

    J = dlfn.project(J_expr[i], Jh)
    J_old = dlfn.project(J_old_expr[i], Jh)
    J_old_old = dlfn.project(J_old_old_expr[i], Jh)


    # Analytical solutions
    Nu_J_projected = ev(J, zero_scalar, zero_scalar, beta_const, dt, h_min[i], mesh)
    Nu_J_old_projected = ev(zero_scalar, J_old, zero_scalar, beta_const, dt, h_min[i], mesh)
    Nu_J_old_old_projected = ev(zero_scalar, zero_scalar, J_old_old, beta_const, dt, h_min[i], mesh)

    Nu_beta_projected = ev(coords, zero_scalar, zero_scalar, beta_linear, dt, h_min[i], mesh)

    # Nu cases
    Nu_J = StabilizationParameter(J, zero_scalar, zero_scalar, beta_const, c_e, Nu_max, dt, E_norm, t_step)
    Nu_J_old = StabilizationParameter(zero_scalar, J_old, zero_scalar, beta_const, c_e, Nu_max, dt, E_norm, t_step)
    Nu_J_old_old = StabilizationParameter(zero_scalar, zero_scalar, J_old_old, beta_const, c_e, Nu_max, dt, E_norm, t_step)

    Nu_beta = StabilizationParameter(coords, zero_scalar, zero_scalar, beta_linear, c_e, Nu_max, dt, E_norm, t_step)

    # Projection of Nu cases
    Nu_Jh = dlfn.project(Nu_J, Jh_test)
    Nu_J_oldh = dlfn.project(Nu_J_old, Jh_test)
    Nu_J_old_oldh = dlfn.project(Nu_J_old_old, Jh_test)

    Nu_betah = dlfn.project(Nu_beta, Jh_test)

    J_list = []
    J_old_list = []
    J_old_old_list = []
    beta_list = []

    for cell in dlfn.cells(mesh):
        x_i = np.mean(np.reshape(cell.get_coordinate_dofs(), (-1, dim)), axis=0)
        vals = np.zeros(1)

        Nu_Jh.eval(vals, x_i)
        J_list.append(abs(vals[0]-maximum(Nu_J_projected, cell, dim))/(abs(vals[0])))
        assert abs(vals[0] - maximum(Nu_J_projected, cell, dim)) / abs(vals[0]) < .1

        Nu_J_oldh.eval(vals, x_i)
        J_old_list.append(abs(vals[0] - maximum(Nu_J_old_projected, cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_J_old_projected, cell, dim)) / abs(vals[0]) < .1

        Nu_J_old_oldh.eval(vals, x_i)
        J_old_old_list.append(abs(vals[0] - maximum(Nu_J_old_old_projected, cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_J_old_old_projected, cell, dim)) / abs(vals[0]) < .1

        Nu_betah.eval(vals, x_i)
        beta_list.append(abs(vals[0] - maximum(Nu_beta_projected, cell, dim)) / abs(vals[0]))
        assert abs(vals[0] - maximum(Nu_beta_projected, cell, dim)) / abs(vals[0]) < .1

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
fig, ax = plt.subplots()
im = ax.imshow(median_arr, cmap='inferno', interpolation='nearest')


ax.set_yticks(np.arange(3))
ax.set_yticklabels(["1D", "2D", "3D"])

ax.set_xticks(np.arange(4))
ax.set_xticklabels(["J", "J_old", "J_old_old", "beta"])

textcolors = ("black", "white")
threshold = np.median(median_arr)
for i in range(3):
    for j in range(4):
        text = ax.text(j, i, "{:.2e}".format(median_arr[i, j]),
                       ha="center", va="center", color=textcolors[median_arr[i, j] < threshold])

plt.savefig('results/rel_err.png')

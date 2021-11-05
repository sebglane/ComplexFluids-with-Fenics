import os.path
import dolfin as dlfn
from dolfin import dot, inner, grad, div, as_vector, Dx, sqrt, cross, bessel_I, lhs, rhs
import numpy as np
from stabilization import *
from classes_EV import *
import matplotlib.pyplot as plt

# == Parameters ==================================
# Polynomial Degree and DoFs
p_deg = 1
n_dofs = 200

# Domain
nx = n_dofs // p_deg	# discretization points in x-direction
l = 1.
h_k = l / nx

# Transport Velocity
u_0 = 1.

# Time-dependent Parameters
t_adaptive = 100
dt = .1 * l / n_dofs / abs(u_0)
dt_0 = dt * np.tanh(1 / t_adaptive)
dT = dlfn.Constant(dt_0) # Can be updated via dt.assign(dt_new) ==> adaptive time-stepping
E_norm = dlfn.Constant(0.0000001)
t_end = 1.


# E-V-Parameters
c_e = 2.
c_max = .02


#
c_e = c_e # Let c_e -> infty to find c_max
Nu_max = abs(u_0) * h_k * c_max

# Path
path = os.path.relpath('results/ev_p{p}_ce{ce:.3f}_cmax{cmax:.3f}'.format(p=p_deg, ce=c_e, cmax=c_max))
if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(os.path.join(path, 'img')):
    os.mkdir(os.path.join(path, 'img'))

# == Mesh ========================================
mesh = dlfn.IntervalMesh(int(nx), 0, l)
space_dim = mesh.geometry().dim()
n_cells = mesh.num_cells()
c = mesh.ufl_cell()

# == Element Formulation =========================
# instance for periodic boundary conditions
pbc = PeriodicBoundary(l)

# Function Spaces Wh and Vh
Wh = dlfn.FunctionSpace(mesh, "CG", p_deg, constrained_domain=pbc)
Vh = dlfn.FunctionSpace(mesh, "DG", 0)
#n_dofs = Wh.dim()

dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)

# == Test and Trial Functions ====================
del_J = dlfn.TestFunction(Wh)

sol_J = dlfn.TrialFunction(Wh)

ki = dlfn.TrialFunction(Wh)

# Solve for
sol_J_h = dlfn.Function(Wh)

k1_h = dlfn.Function(Wh)
k2_h = dlfn.Function(Wh)
k3_h = dlfn.Function(Wh)
k4_h = dlfn.Function(Wh)

# == Buffer Functions ====================
# Functions on Wh
D_func = dlfn.Function(Wh)
sol_J_ = dlfn.Function(Wh)
sol_J_n = dlfn.Function(Wh)
sol_J_n_ = dlfn.Function(Wh)


# == Initial J =========================
J_0_Expr = Initial_Condition(p_deg, l)
J_0_Expr_ = Initial_Condition(p_deg, l, -dT.values()[0] * u_0)
J_0_Expr_n = Initial_Condition(p_deg, l, -dT.values()[0] * u_0 * 2)

J_ana = dlfn.Function(Wh)

# == Initial Nu ========================
J_0 = dlfn.interpolate(J_0_Expr, Wh)
J_0_ = dlfn.interpolate(J_0_Expr_, Wh)
J_0_n = dlfn.interpolate(J_0_Expr_n, Wh)


# Assign initial condition to buffer functions
dlfn.assign(sol_J_h, J_0)
dlfn.assign(sol_J_, J_0)
dlfn.assign(sol_J_n, J_0_)
dlfn.assign(sol_J_n_, J_0_n)


F_t = (sol_J - sol_J_) * del_J * dx

F_1 = (ki * del_J + F_spatial_visc(sol_J_, del_J, u_0, Nu_max)) * dx
F_2 = (ki * del_J + F_spatial_visc(sol_J_ + .5 * dT * k1_h, del_J, u_0, Nu_max)) * dx
F_3 = (ki * del_J + F_spatial_visc(sol_J_ + .5 * dT * k2_h, del_J, u_0, Nu_max)) * dx
F_4 = (ki * del_J + F_spatial_visc(sol_J_ + dT * k3_h, del_J, u_0, Nu_max)) * dx

# TODO: Use dolfin-ERK4 class
Nu_h = StabilizationParameterSD(sol_J_h, sol_J_n, sol_J_n_, u_0, c_e, Nu_max, dT, E_norm)
F_visc_1 = (ki * del_J + F_spatial_visc(sol_J_, del_J, u_0, Nu_h)) * dx
F_visc_2 = (ki * del_J + F_spatial_visc(sol_J_ + .5 * dT * k1_h, del_J, u_0, Nu_h)) * dx
F_visc_3 = (ki * del_J + F_spatial_visc(sol_J_ + .5 * dT * k2_h, del_J, u_0, Nu_h)) * dx
F_visc_4 = (ki * del_J + F_spatial_visc(sol_J_ + dT * k3_h, del_J, u_0, Nu_h)) * dx

#dlfn.multistage.multistagescheme.ERK4(F_visc_1, , bcs=pbc)


F = F_t - dT / 3. * (.5 * k1_h + k2_h + k3_h + .5 * k4_h) * del_J * dx

E_norm.assign(calculate_NormE(sol_J_h, n_dofs))

vtkfile_J0 = dlfn.File(os.path.join(path, 'solution_0.pvd'))
vtkfile_J0 << sol_J_

vtkfile_J1 = dlfn.File(os.path.join(path, 'J_ana_0.pvd'))
vtkfile_J1 << sol_J_

vtkfile_J2 = dlfn.File(os.path.join(path, 'Nu_0.pvd'))
vtkfile_J2 << dlfn.project(Nu_h, Vh)

# == Integrate ===================================
t_i, i = 0., 0
while t_i < t_end:
    print("time step {i}, time {t_i}".format(i=i, t_i=t_i))
    # Adaptive time stepping and control parameters to overcome strong oscillations in the beginning
    dT.assign(dt * np.tanh((i + 1) / t_adaptive))
    #c_e.assign(c_e0 - (c_e0 - c_e_end) * np.tanh((i + 1) / t_adaptive))
    #Nu_max.assign(h_k * abs(u_0) * (c_max0 - (c_max0 - c_max_end) * np.tanh((i + 1) / t_adaptive)))
    t_i += dT.values()[0]
    print(t_i)

    #Nu_h = StabilizationParameterSD(sol_J_, sol_J_n, sol_J_n_, u_0, c_e, Nu_max, dT, E_norm)

    dlfn.assign(J_ana, dlfn.interpolate(Initial_Condition(p_deg, l, offset = t_i * u_0), Wh))


    # === Solve Problem ==========================
    # == RK4 ====================================
    # TODO: Calculate E_Norm for every recursion step
    E_norm.assign(calculate_NormE(sol_J_h, n_dofs))

    if i < 3:
        dlfn.solve(lhs(F_1) == rhs(F_1), k1_h)
        dlfn.solve(lhs(F_2) == rhs(F_2), k2_h)
        dlfn.solve(lhs(F_3) == rhs(F_3), k3_h)
        dlfn.solve(lhs(F_4) == rhs(F_4), k4_h)
    else:
        dlfn.solve(lhs(F_visc_1) == rhs(F_visc_1), k1_h)
        dlfn.solve(lhs(F_visc_2) == rhs(F_visc_2), k2_h)
        dlfn.solve(lhs(F_visc_3) == rhs(F_visc_3), k3_h)
        dlfn.solve(lhs(F_visc_4) == rhs(F_visc_4), k4_h)

    dlfn.solve(lhs(F) == rhs(F), sol_J_h)


    # === Assign to Buffer Functions =============
    dlfn.assign(sol_J_n_, sol_J_n)
    dlfn.assign(sol_J_n, sol_J_)
    dlfn.assign(sol_J_, sol_J_h)

    # Save as .vtk (J and Nu) and .png (only J)
    mod_i = 10
    if (i + 1) % mod_i == 0:
        vtkfile_1 = dlfn.File(os.path.join(path, 'solution_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_1 << sol_J_

        vtkfile_2 = dlfn.File(os.path.join(path, 'Nu_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_2 << dlfn.project(Nu_h, Vh)

        # TODO: Calculate Error-Norm of solution
        vtkfile_3 = dlfn.File(os.path.join(path, 'J_ana_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_3 << J_ana

        x_plot = np.linspace(0., 1., n_dofs)
        y_plot = np.fromiter((sol_J_(x_i) for x_i in x_plot), float)
        y_plot_ana = np.fromiter((J_ana(x_i) for x_i in x_plot), float)
        plt.plot(x_plot, y_plot)
        plt.grid()
        plt.plot(x_plot, y_plot_ana)
        plt.legend(['J', 'J_ana'])
        plt.savefig(os.path.join(path, 'img/sol_{}.png'.format((i+1) // mod_i)))
        plt.clf()

    i += 1

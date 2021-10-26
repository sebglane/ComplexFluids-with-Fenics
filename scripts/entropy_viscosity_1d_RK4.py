import os.path
import dolfin as dlfn
from dolfin import dot, inner, grad, div, as_vector, Dx, sqrt, cross, bessel_I, lhs, rhs
import numpy as np
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
dt = .5 * l / n_dofs / abs(u_0)
dT = dlfn.Constant(dt) # Can be updated via dt.assign(dt_new) ==> adaptive time-stepping
t_end = 2
num_steps = int(t_end // dt) # Only viable for constant dt

dt_0 = dt * np.tanh(1 / t_adaptive)

# E-V-Parameters
c_e = .25 # Let c_e -> infty to find c_max
c_max = .5

"""
Stable for
deg=1:
    c_e = .25, c_max = .5
deg=4:
    c_e = .25, c_max = .25
deg=5
    c_e = .12, c_max = .125
"""

# Path
path = os.path.relpath('results/ev_p{p}_ce{ce:.2f}_cmax{cmax:.2f}'.format(p=p_deg, ce=c_e, cmax=c_max))
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
n_dofs = Wh.dim()

dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)

# == Test and Trial Functions ====================
del_J = dlfn.TestFunction(Wh)

sol_J = dlfn.TrialFunction(Wh)

k1 = dlfn.TrialFunction(Wh)
k2 = dlfn.TrialFunction(Wh)
k3 = dlfn.TrialFunction(Wh)
k4 = dlfn.TrialFunction(Wh)

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

D_normalized = dlfn.Function(Vh)
Nu_h = dlfn.Function(Vh)
Nu_E = dlfn.Function(Vh)

# == Initial J =========================
J_0_Expr = Initial_Condition(p_deg, l)
J_0_Expr_ = Initial_Condition(p_deg, l, -dt_0 * u_0)
J_0_Expr_n = Initial_Condition(p_deg, l, -dt_0 * u_0 * 2)

J_ana = dlfn.Function(Wh)

# == Initial Nu ========================
J_0 = dlfn.interpolate(J_0_Expr, Wh)
J_0_ = dlfn.interpolate(J_0_Expr_, Wh)
J_0_n = dlfn.interpolate(J_0_Expr_n, Wh)

Nu_max = c_max * h_k * abs(u_0)

# Assign initial condition to buffer functions
dlfn.assign(sol_J_, J_0)
dlfn.assign(sol_J_n, J_0_)


F_t = (sol_J - sol_J_) * del_J * dx

F_1 = (k1 * del_J + F_spatial(sol_J_, del_J, u_0)) * dx
F_2 = (k2 * del_J + F_spatial(sol_J_ + .5 * dT * k1, del_J, u_0)) * dx
F_3 = (k3 * del_J + F_spatial(sol_J_ + .5 * dT * k2, del_J, u_0)) * dx
F_4 = (k4 * del_J + F_spatial(sol_J_ + dT * k3, del_J, u_0)) * dx

# TODO: Use dolfin-ERK4 class
F_visc_1 = (k1 * del_J + F_spatial_visc(sol_J_, del_J, u_0, Nu_h)) * dx
F_visc_2 = (k2 * del_J + F_spatial_visc(sol_J_ + .5 * dT * k1, del_J, u_0, Nu_h)) * dx
F_visc_3 = (k3 * del_J + F_spatial_visc(sol_J_ + .5 * dT * k2, del_J, u_0, Nu_h)) * dx
F_visc_4 = (k4 * del_J + F_spatial_visc(sol_J_ + dT * k3, del_J, u_0, Nu_h)) * dx

F = F_t - dT / 3. * (.5 * k1_h + k2_h + k3_h + .5 * k4_h) * del_J * dx

vtkfile_J0 = dlfn.File(os.path.join(path, 'solution_0.pvd'))
vtkfile_J0 << sol_J_

vtkfile_J1 = dlfn.File(os.path.join(path, 'J_ana_0.pvd'))
vtkfile_J1 << sol_J_

# == Integrate ===================================
t_i, i = 0., 0
while t_i < t_end:
    print("time step {i}, time {t_i}".format(i=i, t_i=t_i))
    dT.assign(dt * np.tanh((i + 1) / t_adaptive))
    t_i += dT.values()[0]
    print(t_i)

    dlfn.assign(J_ana, dlfn.interpolate(Initial_Condition(p_deg, l, offset = t_i * u_0), Wh))

    # Calc Nu
    E_normalized = calculate_NormE(sol_J_h)
    if i==0:
        E_normalized +=.00001
    calculate_NormD(D_normalized, sol_J_h, sol_J_, sol_J_n, dT, u_0, p_deg)
    calculate_nu(Nu_h, D_normalized, E_normalized, Nu_max, c_e, h_k)

    # === Solve Problem ==========================
    # == RK4 ====================================
    if i < 0:
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
    dlfn.assign(sol_J_n, sol_J_)
    dlfn.assign(sol_J_, sol_J_h)


    # Save as .vtk (J and Nu) and .png (only J)
    mod_i = 10
    if (i + 1) % mod_i == 0:
        vtkfile_1 = dlfn.File(os.path.join(path, 'solution_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_1 << sol_J_

        vtkfile_2 = dlfn.File(os.path.join(path, 'Nu_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_2 << Nu_h

        # TODO: Calculate Error-Norm of solution
        vtkfile_3 = dlfn.File(os.path.join(path, 'J_ana_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_3 << J_ana

        x_plot = np.linspace(0., 1., n_dofs)
        y_plot = np.fromiter((sol_J_(x_i) for x_i in x_plot), float)
        plt.plot(x_plot, y_plot)
        plt.savefig(os.path.join(path, 'img/sol_{}.png'.format((i+1) // mod_i)))
        plt.clf()

    i += 1

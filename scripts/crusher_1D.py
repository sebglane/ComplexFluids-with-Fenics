import os.path
from dolfin import lhs, rhs
from stabilization import *
from classes_EV import *
import matplotlib.pyplot as plt

# == Parameters ==================================
# Polynomial Degree and DoFs
p_deg = 2

# Domain
nx = 200	# discretization points in x-direction
h_k = 1. / nx

# Transport Velocity
u_0 = 1.0

# Time-dependent Parameters
t_adaptive = 100
dt = .5 / nx / p_deg / abs(u_0)
dT = dlfn.Constant(dt) # Can be updated via dt.assign(dt_new) ==> adaptive time-stepping
E_norm = dlfn.Constant(0.0000001)
t_end = 4.

dt_0 = dt

time_step = dlfn.Constant(0)

# E-V-Parameters
c_e = 2. # Let c_e -> infty to find c_max
c_max = .3

# crusher parameters
J_0 = 1.
rho_0 = 1.
J_crit = .25
alpha = 2.75

# Path
path = os.path.relpath('results/crusher')
if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(os.path.join(path, 'img')):
    os.mkdir(os.path.join(path, 'img'))


# == Mesh ========================================
x_0, x_1 = -1., 4.
delta = .25
mesh = dlfn.IntervalMesh(int(nx), x_0, x_1)
space_dim = mesh.geometry().dim()
n_cells = mesh.num_cells()
c = mesh.ufl_cell()


# == Element Formulation =========================
rho_elem = dlfn.FiniteElement("CG", c, p_deg)
J_elem = dlfn.FiniteElement("CG", c, p_deg)
mixed_elem = dlfn.MixedElement([rho_elem, J_elem])


# == Function Spaces =============================
Wh = dlfn.FunctionSpace(mesh, mixed_elem)
Rhoh, Jh = Wh.split()
Vh = dlfn.FunctionSpace(mesh, "DG", 0)
n_dofs = Wh.dim()

dlfn.info("Number of cells {0}, number of DoFs: {1}".format(n_cells, n_dofs))

# == Surface and Volume Element ==================
dx = dlfn.Measure("dx", domain=mesh)

# == Test and Trial Functions ====================
del_rho, del_J = dlfn.TestFunctions(Wh)

sol = dlfn.TrialFunction(Wh)
sol_rho, sol_J = dlfn.split(sol)

ki = dlfn.TrialFunction(Wh)
ki_rho, ki_J = dlfn.split(ki)

# Solve for
sol_h = dlfn.Function(Wh)
sol_rho_h, sol_J_h = dlfn.split(sol_h)

kh = []
kh_rho = [0] * 4
kh_J = [0] * 4

for k in range(4):
    kh.append(dlfn.Function(Wh))
    kh_rho[k], kh_J[k] = dlfn.split(kh[k])

u = dlfn.Function(Vh)
dlfn.assign(u, dlfn.project(dlfn.Expression('u_0', u_0=u_0, degree=2), Vh))

# == Buffer Functions ====================
# Functions on Wh
sol_ = dlfn.Function(Wh)
sol_n = dlfn.Function(Wh)
sol_n_ = dlfn.Function(Wh)

def F_spatial_visc_chi(sol_, del_, beta, Nu, chi_h):
    return F_spatial(sol_, del_, beta) + Nu * Dx(sol_, 0) * Dx(del_, 0) - chi_h * del_

# == Initial Condition ============================
class J_ana(dlfn.UserExpression):
    def __init__(self, delta, J_0, J_crit, alpha, beta, t, degree=2):
        self._delta = delta
        self._J_0 = J_0
        self._J_crit = J_crit
        self._alpha = alpha
        self._v_0 = beta
        self._t = t
        super().__init__(degree=degree)
    def eval(self, values, x):
        x_s = -self._delta + u_0 * self._t
        if self._t < 2 * self._delta / self._v_0:
            if x[0] < -self._delta - dlfn.DOLFIN_EPS:
                values[0] = self._J_0
            elif x[0] > x_s + dlfn.DOLFIN_EPS:
                values[0] = 0.
            else:
                values[0] = self._J_crit + (self._J_0 - self._J_crit) * dlfn.exp(-self._alpha / self._v_0 * (x[0]+self._delta))
        else:
            if x[0] < -self._delta - dlfn.DOLFIN_EPS:
                values[0] = self._J_0
            elif x[0] > x_s - dlfn.DOLFIN_EPS:
                values[0] = 0.
            elif x[0] < self._delta + dlfn.DOLFIN_EPS:
                values[0] = self._J_crit + (self._J_0 - self._J_crit) * dlfn.exp(-self._alpha / self._v_0 * (x[0]+self._delta))
            else:
                values[0] = self._J_crit + (self._J_0 - self._J_crit) * dlfn.exp(-2 * self._alpha / self._v_0 * (self._delta))


class Initial_J(dlfn.UserExpression):
    def __init__(self, delta, J_0, degree=0):
        self._delta = delta
        self._J_0 = J_0
        super().__init__(degree=degree)
    def eval(self, values, x):
        if x[0] < -self._delta + dlfn.DOLFIN_EPS:
            values[0] = self._J_0
        else:
            values[0] = 0.


class Initial_rho(dlfn.UserExpression):
    def __init__(self, delta, rho_0, degree=0):
        self._delta = delta
        self._rho_0 = rho_0
        super().__init__(degree=degree)
    def eval(self, values, x):
        if x[0] < -self._delta + dlfn.DOLFIN_EPS:
            values[0] = self._rho_0
        else:
            values[0] = 0.

class crusher_region(dlfn.UserExpression):
    def __init__(self, delta, degree=0):
        self._delta = delta
        super().__init__(degree=degree)
    def eval(self, values, x):
        if -self._delta - dlfn.DOLFIN_EPS < x[0] < self._delta + dlfn.DOLFIN_EPS:
            values[0] = 1.
        else:
            values[0] = 0.

chi_h = -crusher_region(delta) * dlfn.conditional(sol_h.sub(0) > .001, 1., 0.) * alpha * (dlfn.conditional(sol_h.sub(1) > J_crit, sol_h.sub(1) - J_crit, 0.))

# == Initial J and rho =================
rho_0_Expr = Initial_rho(delta, rho_0)
J_0_Expr = Initial_J(delta, J_0)

J_initial = dlfn.interpolate(J_0_Expr, Jh.collapse())
rho_initial = dlfn.interpolate(rho_0_Expr, Rhoh.collapse())

# Assign initial condition to buffer functions
dlfn.assign(sol_h.sub(0), rho_initial)
dlfn.assign(sol_h.sub(1), J_initial)

# == Stabilization =====================
Nu_max = c_max * h_k * abs(u_0)

E_rho_norm = dlfn.Constant(calculate_NormE(sol_h.sub(0), n_dofs))
E_J_norm = dlfn.Constant(calculate_NormE(sol_h.sub(1), n_dofs))

Nu_h_rho = StabilizationParameter(sol_h.sub(0), sol_n.sub(0), sol_n_.sub(0), u, c_e, Nu_max, dlfn.Constant(dt), E_rho_norm, time_step)
Nu_h_J = StabilizationParameter(sol_h.sub(1), sol_n.sub(1), sol_n_.sub(1), u, c_e, Nu_max, dlfn.Constant(dt), E_J_norm, time_step)

# == Problem in variational Form =======
F_t_rho = (sol_rho - sol_rho_h) * del_rho * dx
F_t_J = (sol_J - sol_J_h) * del_J * dx

F_rho_1 = (ki_rho * del_rho + F_spatial_visc(sol_h.sub(0), del_rho, u, Nu_h_rho)) * dx
F_J_1 = (ki_J * del_J + F_spatial_visc_chi(sol_h.sub(1), del_J, u, Nu_h_J, chi_h)) * dx
F_1 = F_rho_1 + F_J_1

F_rho_2 = (ki_rho * del_rho + F_spatial_visc(sol_h.sub(0) + .5 * dT * kh_rho[0], del_rho, u, Nu_h_rho)) * dx
F_J_2 = (ki_J * del_J + F_spatial_visc_chi(sol_h.sub(1) + .5 * dT * kh_J[0], del_J, u, Nu_h_J, chi_h)) * dx
F_2 = F_rho_2 + F_J_2

F_rho_3 = (ki_rho * del_rho + F_spatial_visc(sol_h.sub(0) + .5 * dT * kh_rho[1], del_rho, u, Nu_h_rho)) * dx
F_J_3 = (ki_J * del_J + F_spatial_visc_chi(sol_h.sub(1) + .5 * dT * kh_J[1], del_J, u, Nu_h_J, chi_h)) * dx
F_3 = F_rho_3 + F_J_3

F_rho_4 = (ki_rho * del_rho + F_spatial_visc(sol_h.sub(0) + dT * kh_rho[2], del_rho, u, Nu_h_rho)) * dx
F_J_4 = (ki_J * del_J + F_spatial_visc_chi(sol_h.sub(1) + dT * kh_J[2], del_J, u, Nu_h_J, chi_h)) * dx
F_4 = F_rho_4 + F_J_4

F_rho = F_t_rho - dT / 3. * (.5 * kh_rho[0] + kh_rho[1] + kh_rho[2] + .5 * kh_rho[3]) * del_rho * dx
F_J = F_t_J - dT / 3. * (.5 * kh_J[0] + kh_J[1] + kh_J[2] + .5 * kh_J[3]) * del_J * dx

F = F_rho + F_J

# boundary subdomains
gamma00 = dlfn.CompiledSubDomain("on_boundary")
gamma01 = dlfn.CompiledSubDomain("near(x[0], x_0) && on_boundary", x_0=x_0)
gamma02 = dlfn.CompiledSubDomain("near(x[0], x_1) && on_boundary", x_1=x_1)

facet_marker = dlfn.MeshFunction("size_t", mesh, space_dim - 1)
facet_marker.set_all(0)
gamma01.mark(facet_marker, 1)
gamma02.mark(facet_marker, 2)

# == Dirichlet Boundaries ==============
bcs = []
bcs.append(dlfn.DirichletBC(Rhoh, dlfn.Constant(rho_0), facet_marker, 1))
bcs.append(dlfn.DirichletBC(Rhoh, dlfn.Constant(0.), facet_marker, 2))
bcs.append(dlfn.DirichletBC(Jh, dlfn.Constant(J_0), facet_marker, 1))
bcs.append(dlfn.DirichletBC(Jh, dlfn.Constant(0.), facet_marker, 2))


# == Integrate ===================================
t_i, i = 0., 0
while t_i < t_end:
    print("time step {i}, time {t_i}".format(i=i, t_i=t_i))
    #dT.assign(dt * np.tanh((i + 1) / t_adaptive))
    t_i += dT.values()[0]
    print(t_i)
    time_step.assign(i)

    E_rho_norm.assign(calculate_NormE(sol_h.sub(0), n_dofs))
    E_J_norm.assign(calculate_NormE(sol_h.sub(1), n_dofs))



    dlfn.solve(lhs(F_1) == rhs(F_1), kh[0])
    dlfn.solve(lhs(F_2) == rhs(F_2), kh[1])
    dlfn.solve(lhs(F_3) == rhs(F_3), kh[2])
    dlfn.solve(lhs(F_4) == rhs(F_4), kh[3])

    dlfn.solve(lhs(F) == rhs(F), sol_h, bcs=bcs)


    # === Assign to Buffer Functions =============
    dlfn.assign(sol_n_.sub(0), sol_n.split()[0])
    dlfn.assign(sol_n_.sub(1), sol_n.split()[1])

    dlfn.assign(sol_n.sub(0), sol_.split()[0])
    dlfn.assign(sol_n.sub(1), sol_.split()[1])

    dlfn.assign(sol_.sub(0), sol_h.split()[0])
    dlfn.assign(sol_.sub(1), sol_h.split()[1])


    # Save as .vtk
    mod_i = 10
    if (i + 1) % mod_i == 0:
        sol_J_plot = sol_h.split()[1]

        vtkfile_1 = dlfn.File(os.path.join(path, 'rho_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_1 << sol_h.split()[0]

        vtkfile_2 = dlfn.File(os.path.join(path, 'J_{}.pvd'.format((i+1) // mod_i)))
        vtkfile_2 << sol_J_plot

        vtkfile_5 = dlfn.File(os.path.join(path, 'Nu_J{}.pvd'.format((i+1) // mod_i)))
        vtkfile_5 << dlfn.project(Nu_h_J, Vh)

        J_analytical = J_ana(delta, J_0, J_crit, alpha, u_0, t_i)

        vtkfile_6 = dlfn.File(os.path.join(path, 'J_ana{}.pvd'.format((i+1) // mod_i)))
        vtkfile_6 << dlfn.project(J_analytical, Jh.collapse())

        x_plot = np.linspace(x_0, x_1, n_dofs)
        y_plot = np.fromiter((sol_J_plot(x_i) for x_i in x_plot), float)
        y_plot_ana = np.fromiter((J_analytical(x_i) for x_i in x_plot), float)

        plt.figure(figsize=[10., 5.])
        plt.plot(x_plot, y_plot_ana)
        plt.scatter(x_plot, y_plot, s=10., c='red', marker='x')
        plt.grid()
        plt.legend(['J_ana', 'J'])
        plt.savefig(os.path.join(path, 'img/sol_{}.png'.format((i+1) // mod_i)))
        plt.clf()


    i += 1

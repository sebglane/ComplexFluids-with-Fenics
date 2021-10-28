#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import EquationCoefficientHandler
from auxiliary_methods import extract_all_boundary_markers
import dolfin as dlfn
from solver_base import VelocityBCType
from solver_base import PressureBCType
from solver_base import StationarySolverBase as StationarySolver
import numpy as np
import os
from os import path


class ProblemBase:
    _suffix = ".xdmf"

    def __init__(self, main_dir=None):
        # set write and read directory
        if main_dir is None:
            self._main_dir = os.getcwd()
        else:  # pragma: no cover
            assert isinstance(main_dir, str)
            assert path.exist(main_dir)
            self._main_dir = main_dir
        self._results_dir = path.join(self._main_dir, "results")

    def _add_to_field_output(self, field):
        """
        Add the field to a list containing additional fields which are written
        to the xdmf file.
        """
        if not hasattr(self, "_additional_field_output"):
            self._additional_field_output = []
        self._additional_field_output.append(field)

    def _create_xdmf_file(self):

        # get filename
        fname = self._get_filename()
        assert fname.endswith(".xdmf")

        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        self._xdmf_file = dlfn.XDMFFile(fname)
        self._xdmf_file.parameters["flush_output"] = True
        self._xdmf_file.parameters["functions_share_mesh"] = True
        self._xdmf_file.parameters["rewrite_function_mesh"] = False

    def _compute_relative_angular_velocity(self):
        """
        Compute the vorticity, i.e., the curl of the velocity, and project the
        field to a suitable function space.
        """
        velocity = self._get_velocity()
        omega = self._get_omega()

        for field in (velocity, omega):
            family = field.ufl_element().family()
            assert family == "Lagrange"
            degree = field.ufl_element().degree()
            assert degree > 0
        degree = min((velocity.ufl_element().degree(), omega.ufl_element().degree()))

        cell = self._mesh.ufl_cell()
        if self._space_dim == 2:
            elemOmegaRel = dlfn.FiniteElement("DG", cell, degree - 1)
            Wh = dlfn.FunctionSpace(self._mesh, elemOmegaRel)
            curl_v = -velocity[0].dx(1) + velocity[1].dx(0)
            expression = omega - dlfn.Constant(0.5) * curl_v
            relative_angular_velocity = dlfn.project(expression, Wh)
            relative_angular_velocity.rename("relative angular velocity", "")
            return relative_angular_velocity
        elif self._space_dim == 3:  # pragma: no cover
            elemOmegaRel = dlfn.VectorElement("DG", cell, degree - 1)
            Wh = dlfn.FunctionSpace(self._mesh, elemOmegaRel)
            expression = omega - dlfn.Constant(0.5) * dlfn.curl(velocity)
            relative_angular_velocity = dlfn.project(expression, Wh)
            relative_angular_velocity.rename("relative angular velocity", "")
            return relative_angular_velocity
        else:  # pragma: no cover
            raise RuntimeError()

    def _compute_vorticity(self):
        """
        Compute the vorticity, i.e., the curl of the velocity, and project the
        field to a suitable function space.
        """
        velocity = self._get_velocity()

        family = velocity.ufl_element().family()
        assert family == "Lagrange"
        degree = velocity.ufl_element().degree()
        assert degree > 0

        cell = self._mesh.ufl_cell()
        if self._space_dim == 2:
            elemOmega = dlfn.FiniteElement("DG", cell, degree - 1)
            Wh = dlfn.FunctionSpace(self._mesh, elemOmega)
            velocity_curl = dlfn.curl(velocity)
            vorticity = dlfn.project(velocity_curl, Wh)
            vorticity.rename("vorticity", "")
            return vorticity
        elif self._space_dim == 3:  # pragma: no cover
            elemOmega = dlfn.VectorElement("DG", cell, degree - 1)
            Wh = dlfn.FunctionSpace(self._mesh, elemOmega)
            vorticity = dlfn.project(velocity_curl, Wh)
            vorticity.rename("vorticity", "")
            return vorticity
        else:  # pragma: no cover
            raise RuntimeError()

    def _compute_pressure_gradient(self):
        """
        Returns the projection of the pressure gradient on a suitable function
        space of discontinuous Galerkin elements.
        """
        pressure = self._get_pressure()

        family = pressure.ufl_element().family()
        assert family == "Lagrange"
        degree = pressure.ufl_element().degree()
        assert degree >= 0

        cell = self._mesh.ufl_cell()
        elemGradP = dlfn.VectorElement("DG", cell, degree - 1)
        Wh = dlfn.FunctionSpace(self._mesh, elemGradP)
        pressure_gradient = dlfn.project(dlfn.grad(pressure), Wh)
        pressure_gradient.rename("pressure gradient", "")

        return pressure_gradient

    def _compute_stream_potential(self):
        """
        Computes the stream potential of the current velocity field. The stream
        potential or flow potential corresponds to the irrotional component of
        the velocity field. It is a projection of the actual velocity field
        onto the subspace of irrotional fields.
        """
        assert hasattr(self, "_mesh")
        assert hasattr(self, "_boundary_markers")

        velocity = self._get_velocity()

        # create suitable function space
        family = velocity.ufl_element().family()
        assert family == "Lagrange"
        degree = velocity.ufl_element().degree()
        assert degree > 0
        cell = self._mesh.ufl_cell()
        elemOmega = dlfn.FiniteElement("CG", cell, max(degree - 1, 1))
        Wh = dlfn.FunctionSpace(self._mesh, elemOmega)

        # test and trial functions
        phi = dlfn.TrialFunction(Wh)
        psi = dlfn.TestFunction(Wh)

        # volume element
        dV = dlfn.Measure("dx", domain=self._mesh)
        dA = dlfn.Measure("ds", domain=self._mesh, subdomain_data=self._boundary_markers)

        # TODO: Implement different boundary conditions
        # extract boundary conditions
        bc_map = self._get_boundary_conditions_map()
        assert VelocityBCType.no_slip in bc_map

        other_bndry_ids = extract_all_boundary_markers(self._mesh, self._boundary_markers)

        # apply homogeneous Dirichlet bcs on the potential where a no-slip
        # bc is applied on the velocity
        dirichlet_bcs = []
        for i in bc_map[VelocityBCType.no_slip]:
            bc_object = dlfn.DirichletBC(Wh, dlfn.Constant(0.0),
                                         self._boundary_markers, i)
            dirichlet_bcs.append(bc_object)
            # remove current boundary id from the list of all boundary ids
            other_bndry_ids.discard(i)

        # remove no-normal flux boundary ids from the list of all boundary ids
        if VelocityBCType.no_normal_flux in bc_map:
            for i in bc_map[VelocityBCType.no_normal_flux]:
                other_bndry_ids.discard(i)

        # normal vector
        normal = dlfn.FacetNormal(self._mesh)
        # weak forms
        lhs = dlfn.inner(dlfn.grad(phi), dlfn.grad(psi)) * dV
        rhs = dlfn.div(velocity) * psi * dV
        # apply Neumann boundary on the remaining part of the list of boundary
        # ids
        for i in other_bndry_ids:
            rhs += - dlfn.inner(normal, velocity) * psi * dA(i)

        # potential of the flow
        stream_potential = dlfn.Function(Wh)
        stream_potential.rename("velocity potential", "")

        # solve problem
        dlfn.solve(lhs == rhs, stream_potential, dirichlet_bcs)

        return stream_potential

    def _get_boundary_conditions_map(self, field="velocity"):
        """
        Returns a mapping relating the type of the boundary condition to the
        boundary identifiers where it is is applied.
        """
        assert isinstance(field, str)
        assert hasattr(self, "_bcs")

        if field == "velocity":
            BCType = VelocityBCType
        elif field.lower() == "pressure":
            BCType = PressureBCType
        else:  # pragma: no cover
            raise RuntimeError()

        bc_map = {}
        for bc_type, bc_bndry_id, _ in self._bcs:
            if bc_type not in BCType:
                continue
            if bc_type in bc_map:
                tmp = set(bc_map[bc_type])
                tmp.add(bc_bndry_id)
                bc_map[bc_type] = tuple(tmp)
            else:
                bc_map[bc_type] = (bc_bndry_id, )

        return bc_map

    def _get_filename(self):
        """
        Class method returning a filename for the given set of parameters.
        """
        # input check
        assert hasattr(self, "_problem_name")
        assert hasattr(self, "_coefficient_handler")

        problem_name = self._problem_name

        fname = problem_name
        fname += self._coefficient_handler.get_file_suffix()
        fname += self._suffix

        return path.join(self._results_dir, fname)

    def _get_pressure(self):
        """
        Returns the pressure field.
        """
        solver = self._get_solver()
        solution = solver.solution
        solution_components = solution.split()
        index = solver.field_association["pressure"]
        return solution_components[index]

    def _get_solver(self):  # pragma: no cover
        """
        Purely virtual method for getting the solver of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def _get_omega(self):
        """
        Returns the microrotation field.
        """
        solver = self._get_solver()
        solution = solver.solution
        solution_components = solution.split()
        index = solver.field_association["omega"]
        return solution_components[index]

    def _get_velocity(self):
        """
        Returns the velocity field.
        """
        solver = self._get_solver()
        solution = solver.solution
        solution_components = solution.split()
        index = solver.field_association["velocity"]
        return solution_components[index]

    def _write_xdmf_file(self, current_time=0.0):
        """
        Write the output to an xdmf file. The solution and additional fields
        are output to the file.
        """
        assert isinstance(current_time, float)
        # create xdmf file
        if not hasattr(self, "_xdmf_file"):
            self._create_xdmf_file()
        # get solution
        solver = self._get_solver()
        solution = solver.solution
        # serialize
        solution_components = solution.split()
        for index, name in solver.sub_space_association.items():
            solution_components[index].rename(name, "")
            self._xdmf_file.write(solution_components[index], current_time)

        if hasattr(self, "_additional_field_output"):
            for field in self._additional_field_output:
                self._xdmf_file.write(field, current_time)

    def postprocess_solution(self):  # pragma: no cover
        """
        Virtual method for additional post-processing.
        """
        pass

    def setup_mesh(self):  # pragma: no cover
        """
        Purely virtual method for setting up the mesh of the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_angular_velocity(self):  # pragma: no cover
        """
        Virtual method for specifying the angular velocity of the rotating system.
        """
        pass

    def set_boundary_conditions(self):  # pragma: no cover
        """
        Virtual method for specifying the boundary conditions of the
        problem.
        """
        pass

    def set_equation_coefficients(self):  # pragma: no cover
        """
        Sets up the dimensionless parameters of the model by creating an
        ``EquationCoefficientHandler`` object.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_internal_constraints(self):  # pragma: no cover
        """
        Virtual method for specifying constraints on internal degrees of freedom
        of the problem.
        """
        pass

    def set_periodic_boundary_conditions(self):  # pragma: no cover
        """
        Virtual method for specifying periodic boundary conditions of the
        problem.
        """
        pass

    def set_body_force(self):  # pragma: no cover
        """
        Virtual method for specifying the body force of the problem.
        """
        pass

    def set_volume_couple(self):  # pragma: no cover
        """
        Virtual method for specifying the volume couple of the problem.
        """
        pass

    def set_inertia_tensor(self):  # pragma: no cover
        """
        Virtual method for specifying the inertia tensor of the micropolar medium.
        """
        pass

    def solve_problem(self):  # pragma: no cover
        """
        Purely virtual method for solving the problem.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    @property
    def space_dim(self):
        assert hasattr(self, "_space_dim")
        return self._space_dim

    def write_boundary_markers(self):
        """
        Write the boundary markers specified by the MeshFunction
        `_boundary_markers` to a pvd-file.
        """
        assert hasattr(self, "_boundary_markers")
        assert hasattr(self, "_problem_name")

        # create results directory
        assert hasattr(self, "_results_dir")
        if not path.exists(self._results_dir):
            os.makedirs(self._results_dir)

        problem_name = self._problem_name
        suffix = ".pvd"
        fname = problem_name + "_BoundaryMarkers"
        fname += suffix
        fname = path.join(self._results_dir, fname)

        dlfn.File(fname) << self._boundary_markers


class StationaryProblem(ProblemBase):
    """
    Class to simulate stationary fluid flow using the `StationaryNavierStokesSolver`.
    Parameters
    ----------
    main_dir: str (optional)
        Directory to save the results.
    tol: float (optional)
        Final tolerance .
    maxiter: int (optional)
        Maximum number of iterations in total.
    tol_picard: float (optional)
        Tolerance for the Picard iteration.
    maxiter_picard: int (optional)
        Maximum number of Picard iterations.
    """
    def __init__(self, main_dir=None, form_convective_term="standard",
                 tol=1e-10, maxiter=50, tol_picard=1e-2, maxiter_picard=10):
        """
        Constructor of the class.
        """
        super().__init__(main_dir)

        # input check
        assert isinstance(form_convective_term, str)
        assert all(isinstance(i, int) and i > 0 for i in (maxiter, maxiter_picard))
        assert all(isinstance(i, float) and i > 0.0 for i in (tol_picard, tol_picard))
        self._form_convective_term = form_convective_term
        # set numerical tolerances
        self._tol_picard = tol_picard
        self._maxiter_picard = maxiter_picard
        self._tol = tol
        self._maxiter = maxiter

        # setting discretization parameters
        # polynomial degree
        self._p_deg = 1

    def _get_solver(self):
        assert hasattr(self, "_micropolar_solver")
        return self._micropolar_solver

    def solve_problem(self):
        """
        Solve the stationary problem.
        """
        # setup mesh
        self.setup_mesh()
        assert self._mesh is not None
        self._space_dim = self._mesh.geometry().dim()
        self._n_cells = self._mesh.num_cells()

        # setup periodic boundary conditions
        self.set_periodic_boundary_conditions()

        # setup internal constraints
        self.set_internal_constraints()

        # setup angular velocity
        self.set_angular_velocity()

        # setup boundary conditions
        self.set_boundary_conditions()

        # setup body forces
        self.set_body_force()

        # setup inertia tensor
        self.set_inertia_tensor()

        # setup has body force
        self.set_volume_couple()

        # set equation coefficients
        self.set_equation_coefficients()
        assert hasattr(self, "_coefficient_handler")
        assert isinstance(self._coefficient_handler, EquationCoefficientHandler)
        self._coefficient_handler.close()

        # check setup of boundary and initial conditions and constraints
        if not hasattr(self, "_bcs"):
            assert hasattr(self, "_periodic_bcs")
        if hasattr(self, "_internal_constraints"):
            assert hasattr(self, "_bcs")

        # create solver object
        if not hasattr(self, "_micropolar_solver"):
            self._micropolar_solver = \
                StationarySolver(self._mesh, self._boundary_markers,
                                 self._form_convective_term,
                                 self._tol, self._maxiter,
                                 self._tol_picard, self._maxiter_picard)

        # pass periodic boundary conditions
        if hasattr(self, "_periodic_bcs"):
            assert hasattr(self, "_periodic_boundary_ids")
            self._micropolar_solver.set_periodic_boundary_conditions(self._periodic_bcs,
                                                                     self._periodic_boundary_ids)

        if hasattr(self, "_angular_velocity"):
            self._micropolar_solver.set_angular_velocity(self._angular_velocity)

        # pass boundary conditions
        if hasattr(self, "_internal_constraints"):
            self._micropolar_solver.set_boundary_conditions(self._bcs,
                                                            self._internal_constraints)
        else:
            self._micropolar_solver.set_boundary_conditions(self._bcs)

        # pass equation coefficients
        self._micropolar_solver.set_equation_coefficients(self._coefficient_handler.equation_coefficients["momentum"],
                                                          self._coefficient_handler.equation_coefficients["spin"])

        # pass body force
        if hasattr(self, "_body_force"):
            self._micropolar_solver.set_body_force(self._body_force)

        try:
            dlfn.info("Solving problem")
            self._micropolar_solver.solve()
            # postprocess solution
            self.postprocess_solution()
            # write XDMF-files
            self._write_xdmf_file()
            return
        except RuntimeError:  # pragma: no cover
            pass
        except Exception as ex:  # pragma: no cover
            template = "An unexpected exception of type {0} occurred. " + \
                        "Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

        # parameter continuation
        dlfn.info("Solving problem with experimental parameter continuation...")  # pragma: no cover

        # mixed logarithmic-linear spacing
        finalRe = self._coefficient_handler.Re  # pragma: no cover
        assert finalRe is not None  # pragma: no cover
        logRange = np.logspace(np.log10(10.0), np.log10(finalRe),
                               num=8, endpoint=True)  # pragma: no cover
        linRange = np.linspace(logRange[-2], finalRe,
                               num=8, endpoint=True)  # pragma: no cover
        finalRange = np.concatenate((logRange[:-2], linRange))  # pragma: no cover
        for Re in finalRange:  # pragma: no cover
            # modify dimensionless numbers
            self._coefficient_handler.modify_dimensionless_number("Re", Re)
            self._micropolar_solver.set_equation_coefficients(self._coefficient_handler.equation_coefficients["momentum"],
                                                              self._coefficient_handler.equation_coefficients["spin"])
            # solve problem
            dlfn.info("Solving problem with Re = {0:.2f}".format(Re))
            self._micropolar_solver.solve()

        # postprocess solution
        self.postprocess_solution()  # pragma: no cover

        # write XDMF-files
        self._write_xdmf_file()  # pragma: no cover

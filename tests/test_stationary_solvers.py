#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import EquationCoefficientHandler
import dolfin as dlfn
from grid_generator import hyper_cube, HyperCubeBoundaryMarkers
from micropolar_problem import StationaryProblem
from solver_base import VelocityBCType, OmegaBCType

dlfn.set_log_level(20)


class PeriodicDomain(dlfn.SubDomain):
    def inside(self, x, on_boundary):
        """Return True if `x` is located on the master edge and False
        else.
        """
        return (dlfn.near(x[0], 0.0) and on_boundary)

    def map(self, x_slave, x_master):
        """Map the coordinates of the support points (nodes) of the degrees
        of freedom of the slave to the coordinates of the corresponding
        master edge.
        """
        x_master[0] = x_slave[0] - 1.0
        x_master[1] = x_slave[1]


class CavityProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir, form_convective_term="standard", tol=1e-10,
                         maxiter=50, tol_picard=1e-1, maxiter_picard=10)
        self._n_points = n_points
        self._problem_name = "Cavity"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        no_slip = VelocityBCType.no_slip
        constant = VelocityBCType.constant
        constant_spin = OmegaBCType.constant
        BoundaryMarkers = HyperCubeBoundaryMarkers
        self._bcs = ((no_slip, BoundaryMarkers.left.value, None),
                     (no_slip, BoundaryMarkers.right.value, None),
                     (no_slip, BoundaryMarkers.bottom.value, None),
                     (constant, BoundaryMarkers.top.value, (1.0, 0.0)),
                     (constant_spin, BoundaryMarkers.left.value, 0.0),
                     (constant_spin, BoundaryMarkers.right.value, 0.0),
                     (constant_spin, BoundaryMarkers.bottom.value, 0.0),
                     (constant_spin, BoundaryMarkers.top.value, 0.0))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0, N=0.25,
                                                               L=0.4, Th=1.0e-3)

    def postprocess_solution(self):
        _ = self._get_pressure()
        _ = self._get_velocity()
        _ = self._get_omega()
        _ = self.space_dim
        # add pressure gradient to the field output
        self._add_to_field_output(self._compute_pressure_gradient())
        # add vorticity to the field output
        self._add_to_field_output(self._compute_vorticity())
        # add stream potential to the field output
        self._add_to_field_output(self._compute_stream_potential())
        # add relative angular to the field output
        self._add_to_field_output(self._compute_relative_angular_velocity())
        

class CouetteProblem(StationaryProblem):
    """Couette flow problem with periodic boundary conditions in x-direction."""
    def __init__(self, n_points, main_dir=None):
        super().__init__(main_dir)

        self._n_points = n_points
        self._problem_name = "Couette"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_cube(2, self._n_points)

    def set_boundary_conditions(self):
        # boundary conditions
        no_slip = VelocityBCType.no_slip
        constant_component = VelocityBCType.constant_component
        no_normal_flux = VelocityBCType.no_normal_flux
        constant_spin = OmegaBCType.constant
        BoundaryMarkers = HyperCubeBoundaryMarkers
        self._bcs = ((no_slip, BoundaryMarkers.bottom.value, None),
                     (constant_component, BoundaryMarkers.top.value, 0, 1.0),
                     (no_normal_flux, BoundaryMarkers.top.value, None),
                     (constant_spin, BoundaryMarkers.top.value, None),
                     (constant_spin, BoundaryMarkers.bottom.value, None))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=1.0)

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary condition in x-direction."""
        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value)


def test_cavity():
    cavity_flow = CavityProblem(25)
    cavity_flow.solve_problem()


def test_couette_flow():
    couette_flow = CouetteProblem(10)
    couette_flow.solve_problem()


if __name__ == "__main__":
    test_cavity()
    test_couette_flow()
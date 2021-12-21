#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import EquationCoefficientHandler
import dolfin as dlfn
from grid_generator import hyper_cube
from grid_generator import hyper_rectangle
from grid_generator import HyperCubeBoundaryMarkers
from grid_generator import HyperRectangleBoundaryMarkers
from micropolar_problem import StationaryProblem
from solver_base import VelocityBCType, OmegaBCType, PressureBCType

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
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0, N=0.25,
                                                               L=0.4, Th=1.0e-3)

    def set_periodic_boundary_conditions(self):
        """Set periodic boundary condition in x-direction."""
        self._periodic_bcs = PeriodicDomain()
        self._periodic_boundary_ids = (HyperCubeBoundaryMarkers.left.value,
                                       HyperCubeBoundaryMarkers.right.value)


class ChannelFlowProblem(StationaryProblem):
    def __init__(self, n_points, main_dir=None, bc_type="inlet",
                 form_convective_term="standard"):
        super().__init__(main_dir, form_convective_term=form_convective_term,
                         tol_picard=1e6)

        assert isinstance(n_points, int)
        assert n_points > 0
        self._n_points = n_points

        assert isinstance(bc_type, str)
        assert bc_type in ("inlet", "pressure_gradient", "inlet_pressure", "inlet_component")
        self._bc_type = bc_type

        if self._bc_type == "inlet":
            self._problem_name = "ChannelFlowInlet"
        elif self._bc_type == "pressure_gradient":
            self._problem_name = "ChannelFlowPressureGradient"
        elif self._bc_type == "inlet_pressure":
            self._problem_name = "ChannelFlowInletPressure"
        elif self._bc_type == "inlet_component":
            self._problem_name = "ChannelFlowInletComponent"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers = hyper_rectangle((0.0, 0.0), (10.0, 1.0),
                                                             (10 * self._n_points, self._n_points))

    def set_boundary_conditions(self):
        # functions
        inlet_profile_str = "6.0*x[1]*(1.0-x[1])"
        inlet_velocity = dlfn.Expression((inlet_profile_str, "0.0"), degree=2)
        inlet_velocity_component = dlfn.Expression(inlet_profile_str, degree=2)
        inlet_omega = inlet_velocity_component
        outlet_pressure = dlfn.Expression("0.0", degree=0)
        # boundary markers
        Markers = HyperRectangleBoundaryMarkers

        # boundary conditions
        no_slip = VelocityBCType.no_slip
        function = VelocityBCType.function
        function_component = VelocityBCType.function_component
        constant_pressure = PressureBCType.constant
        function_pressure = PressureBCType.function
        constant_spin = OmegaBCType.constant
        function_spin = OmegaBCType.function
        
        self._bcs = [(function_spin, Markers.left.value, inlet_omega),
                     (constant_spin, Markers.top.value, 0.0),
                     (constant_spin, Markers.bottom.value, 0.0)]
        if self._bc_type == "inlet":
            # inlet velocity profile
            self._bcs.append((function, Markers.left.value, inlet_velocity))
            # no-slip on the walls
            self._bcs.append((no_slip, Markers.bottom.value, None))
            self._bcs.append((no_slip, Markers.top.value, None))
        elif self._bc_type == "inlet_pressure":
            # inlet velocity profile
            self._bcs.append((function, Markers.left.value, inlet_velocity))
            # no-slip on the walls
            self._bcs.append((no_slip, Markers.bottom.value, None))
            self._bcs.append((VelocityBCType.no_slip, Markers.top.value, None))
            # pressure at the outlet (as a function)
            self._bcs.append((function_pressure, Markers.right.value, outlet_pressure))
        elif self._bc_type == "inlet_component":
            # inlet velocity profile (component)
            self._bcs.append((function_component, Markers.left.value, 0, inlet_velocity_component))
            # no-slip on the walls
            self._bcs.append((no_slip, Markers.bottom.value, None))
            self._bcs.append((no_slip, Markers.top.value, None))
            # pressure at the outlet (as a constant)
            self._bcs.append((constant_pressure, Markers.right.value, 0.0))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=100.0, N=0.25,
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
        # add relative angular to the field output
        self._add_to_field_output(self._compute_relative_angular_velocity())


def test_cavity():
    cavity_flow = CavityProblem(25)
    cavity_flow.solve_problem()

def test_channel_flow():
    for bc_type in ("inlet", "inlet_pressure", "inlet_component"):
        print(bc_type)
        channel_flow = ChannelFlowProblem(10, bc_type=bc_type)
        channel_flow.solve_problem()

def test_couette_flow():
    couette_flow = CouetteProblem(10)
    couette_flow.solve_problem()


if __name__ == "__main__":
    test_cavity()
    test_channel_flow()
    test_couette_flow()
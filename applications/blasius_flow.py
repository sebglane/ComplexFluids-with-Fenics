#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dolfin as dlfn
from auxiliary_classes import EquationCoefficientHandler
from grid_generator import blasius_plate
from micropolar_problem import StationaryProblem
from solver_base import VelocityBCType, OmegaBCType

dlfn.set_log_level(20)


class MicroPolarBlasiusFlowProblem(StationaryProblem):
    def __init__(self, main_dir=None):
        super().__init__(main_dir)
        self._problem_name = "BlasiusFlow"

    def setup_mesh(self):
        # create mesh
        self._mesh, self._boundary_markers, self._boundary_marker_map = blasius_plate()

    def set_boundary_conditions(self):
        # velocity boundary conditions
        inlet_velocity = dlfn.Constant((1.0, 0.0))
        constant = VelocityBCType.constant
        no_flux = VelocityBCType.no_normal_flux
        constant_spin = OmegaBCType.constant
        self._bcs = ((constant, self._boundary_marker_map["inlet"], inlet_velocity),
                     (no_flux, self._boundary_marker_map["bottom"], None),
                     (no_flux, self._boundary_marker_map["top"], None),
                     (constant_spin, self._boundary_marker_map["inlet"], 0.0))

    def set_equation_coefficients(self):
        self._coefficient_handler = EquationCoefficientHandler(Re=200.0, N=0.25,
                                                               L=0.4, Th=1.0e-3)

    def set_internal_constraints(self):
        self._internal_constraints = ((VelocityBCType.no_slip, self._boundary_marker_map["plate"], None),
                                      (OmegaBCType.constant, self._boundary_marker_map["plate"], 0.0))

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
        # self._add_to_field_output(self._compute_stream_potential())
        # add relative angular to the field output
        self._add_to_field_output(self._compute_relative_angular_velocity())


if __name__ == "__main__":
    blasius_flow = MicroPolarBlasiusFlowProblem()
    blasius_flow.solve_problem()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dolfin import NonlinearProblem
from dolfin import SystemAssembler
from dolfin import Constant
import math


__all__ = ["AngularVelocityVector", "CustomNonlinearProblem",
           "EquationCoefficientHandler"]


class AngularVelocityVector:
    def __init__(self, space_dim=2, function=None):
        # input check
        assert isinstance(space_dim, int)
        assert space_dim in (2, 3)
        self._space_dim = space_dim
        self._current_time = 0.0
        # set value size
        if self._space_dim == 2:
            self._value_size = 1
        else:
            self._value_size = 3
        # set function
        if function is not None:
            self.set_angular_velocity_function(function)

    def _modify_time(self):
        assert hasattr(self, "_omega")
        omega_value = self._angular_velocity.value()
        self._omega.assign(Constant(omega_value))
        assert hasattr(self, "_alpha")
        if self._alpha is not None:
            alpha_value = self._angular_velocity.derivative()
            self._alpha.assign(Constant(alpha_value))

    def _setup_angular_acceleration(self):
        assert hasattr(self, "_angular_velocity")
        derivative_exists = True
        try:
            _ = self._angular_velocity.derivative()
        except RuntimeError:
            derivative_exists = False
        except Exception:  # pragma: no cover
            raise RuntimeError()
        if derivative_exists:
            derivative_value = self._angular_velocity.derivative()
            self._alpha = Constant(derivative_value)
        else:
            self._alpha = None

    def _setup_angular_velocity(self):
        assert hasattr(self, "_angular_velocity")
        value = self._angular_velocity.value()
        self._omega = Constant(value)

    def set_angular_velocity_function(self, function):
        assert isinstance(function, FunctionTime)
        if self._space_dim == 2:
            assert function.value_size == 1
        else:
            assert function.value_size == 3
        self._angular_velocity = function
        self._setup_angular_velocity()
        self._setup_angular_acceleration()

    def set_time(self, current_time):
        assert isinstance(current_time, float)
        assert current_time >= self._current_time
        self._current_time = current_time
        self._angular_velocity.set_time(self._current_time)
        self._modify_time()

    @property
    def derivative(self):
        assert hasattr(self, "_alpha")
        return self._alpha

    @property
    def space_dim(self):
        return self._space_dim

    @property
    def value(self):
        assert hasattr(self, "_omega")
        return self._omega


class FunctionTime:
    def __init__(self, value_size, current_time=0.0):
        # input check
        assert isinstance(value_size, int)
        assert value_size > 0
        self._value_size = value_size
        assert isinstance(current_time, float)
        self._current_time = 0.0

    def derivative(self):  # pragma: no cover
        """
        Purely virtual method returning the time derivative of the function.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    def set_time(self, current_time):
        assert isinstance(current_time, float)
        assert current_time >= self._current_time
        self._current_time = current_time

    def value(self):  # pragma: no cover
        """
        Purely virtual method returning the value of the function.
        """
        raise NotImplementedError("You are calling a purely virtual method.")

    @property
    def value_size(self):
        return self._value_size


class CustomNonlinearProblem(NonlinearProblem):
    """Class for interfacing with not only :py:class:`NewtonSolver`."""
    def __init__(self, F, bcs, J):
        """Return subclass of :py:class:`dolfin.NonlinearProblem` suitable
        for :py:class:`NewtonSolver` based on
        :py:class:`field_split.BlockPETScKrylovSolver` and PCD
        preconditioners from :py:class:`preconditioners.PCDPreconditioner`.

        *Arguments*
            F (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Linear form representing the equation.
            bcs (:py:class:`list` of :py:class:`dolfin.DirichletBC`)
                Boundary conditions applied to ``F``, ``J``, and ``J_pc``.
            J (:py:class:`dolfin.Form` or :py:class:`ufl.Form`)
                Bilinear form representing system Jacobian for the iteration.

        All the arguments should be given on the common mixed function space.
        """
        super(CustomNonlinearProblem, self).__init__()

        # assembler for Newton/Picard system
        self.assembler = SystemAssembler(J, F, bcs)

        # store forms/bcs for later
        self.forms = {
            "F": F,
            "J": J
        }
        self._bcs = bcs

    def get_form(self, key):
        form = self.forms.get(key)
        if form is None:  # pragma: no cover
            raise AttributeError("Form '%s' requested by NonlinearProblem not "
                                 "available" % key)
        return form

    def function_space(self):
        return self.forms["F"].arguments()[0].function_space()

    def F(self, b, x):
        self.assembler.assemble(b, x)

    def J(self, A, x):
        self.assembler.assemble(A)


class EquationCoefficientHandler():
    def __init__(self, **kwargs):
        self._dimensionless_numbers = dict()
        self._read_dimensionless_number(kwargs, "Re", "Reynolds")
        self._read_dimensionless_number(kwargs, "Fr", "Froude")
        self._read_dimensionless_number(kwargs, "Ro", "Rossby")
        self._read_dimensionless_number(kwargs, "Ek", "Ekman")
        self._read_dimensionless_number(kwargs, "L", None)
        self._read_dimensionless_number(kwargs, "M", None)
        self._read_dimensionless_number(kwargs, "N", "Coupling number")
        self._read_dimensionless_number(kwargs, "Th", "Inertia number")
        self._closed = False

    def __str__(self):
        assert hasattr(self, "_dimensionless_numbers")

        def add_row(string, *args):
            width = 65
            widths = (32, 32)
            if len(args) == 0:
                string += "+" + widths[0] * "-" + "+" + widths[1] * "-" + "+\n"
            elif len(args) == 1:
                string += "|" + f" {args[0]:<{width-1}}" + "|\n"
            elif len(args) == 2:
                string += "|" + f"{args[0]:^{widths[0]}}" + "|" + f"{args[1]:^{widths[1]}}" + "|\n"
            else:  # pragma: no cover
                raise RuntimeError()
            return string

        string = "+" + 65 * "-" + "+\n"

        string = add_row(string, "dimensionless numbers")
        string = add_row(string)
        string = add_row(string, "name", "value")
        string = add_row(string)

        for key, value in self._dimensionless_numbers.items():
            if value is not None:
                string = add_row(string, key, value)
            else:
                string = add_row(string, key, "None")
        string = add_row(string)

        if hasattr(self, "_equation_coefficients"):
            string = add_row(string, "equation coefficients")
            string = add_row(string)

            for super_key, coefficients in self._equation_coefficients.items():
                string = add_row(string, super_key + " equation")
                string = add_row(string, "term", "value")
                string = add_row(string)
                for key, value in coefficients.items():
                    name = key.rstrip("term").replace("_", " ").strip()
                    if value is not None:
                        string = add_row(string, key, value)
                    else:
                        string = add_row(string, key, "None")
                string = add_row(string)

            string = add_row(string, "coefficient expressions")
            string = add_row(string, "term", "value")
            string = add_row(string)

            coefficient_expressions = dict()

            if not any(x in self._dimensionless_numbers for x in ("L", "M", "N", "Th")):
                if ("Ro" not in self._dimensionless_numbers) and \
                        ("Ek" not in self._dimensionless_numbers):
                    coefficient_expressions["pressure"] = "1"
                    coefficient_expressions["viscous"] = "1 / Re"
                else:
                    if "Ro" in self._dimensionless_numbers and "Re" in self._dimensionless_numbers:
                        rotation_coefficient = "1/ Ro"
                        viscous_coefficient = "1 / Re"
                    elif "Ro" in self._dimensionless_numbers and "Ek" in self._dimensionless_numbers:
                        rotation_coefficient = "1 / Ro"
                        viscous_coefficient = "Ek / Ro"
                    elif "Ek" in self._dimensionless_numbers and "Re" in self._dimensionless_numbers:
                        rotation_coefficient = "1/ (Ek * Re)"
                        viscous_coefficient = "1 / Re"
                    elif "Ek" in self._dimensionless_numbers:
                        assert "Re" not in self._dimensionless_numbers
                        assert "Ro" not in self._dimensionless_numbers
                        rotation_coefficient = "1"
                        viscous_coefficient = "Ek"
                    elif "Ro" in self._dimensionless_numbers:
                        assert "Re" not in self._dimensionless_numbers
                        assert "Ek" not in self._dimensionless_numbers
                        rotation_coefficient = "1 / Ro"
                        viscous_coefficient = "1"
                    else:  # pragma: no cover
                        raise RuntimeError()
                    coefficient_expressions["coriolis"] = rotation_coefficient
                    coefficient_expressions["euler"] = rotation_coefficient
                    coefficient_expressions["pressure"] = "1"
                    coefficient_expressions["viscous"] = viscous_coefficient
                if ("Fr" in self._dimensionless_numbers):
                    coefficient_expressions["body force"] = "1 / Fr^2"
                else:
                    coefficient_expressions["body force"] = "--"
            else:
                assert all(x not in self._dimensionless_numbers for x in ("Ek", "Ro"))
                coefficient_expressions["pressure"] = "1"
                coefficient_expressions["viscous"] = "1 / (1 -N^2) / Re"
                coefficient_expressions["coupling"] = "N^2 / (1 - N^2)"
                coefficient_expressions["viscous (spin)"] = "1 / L^2 / Re / Th"
                coefficient_expressions["vol. viscous (spin)"] = "1 / M^2 / Re / Th"
                coefficient_expressions["coupling (spin)"] = "N^2 / (1 - N^2)"
                if ("Fr" in self._dimensionless_numbers):
                    coefficient_expressions["body force"] = "1 / Fr^2"
                else:
                    coefficient_expressions["body force"] = "--"

            for name, expression in coefficient_expressions.items():
                string = add_row(string, name, expression)
            string = add_row(string)
        return string

    def _compute_equation_coefficients(self):
        assert hasattr(self, "_dimensionless_numbers")

        # hydrodynamic case
        if not any(x in self._dimensionless_numbers for x in ("L", "M", "N", "Th")):

            if not hasattr(self, "_equation_coefficients"):
                self._equation_coefficients = dict()
            self._equation_coefficients["momentum"] = dict()
            coefficients = self._equation_coefficients["momentum"]

            coefficients["convective_term"] = 1.0

            if "Ro" not in self._dimensionless_numbers and \
                    "Ek" not in self._dimensionless_numbers:
                coefficients["coriolis_term"] = None
                coefficients["euler_term"] = None
                coefficients["pressure_term"] = 1.0
                if "Re" in self._dimensionless_numbers:
                    coefficients["viscous_term"] = 1.0 / self._dimensionless_numbers["Re"]
                else:  # pragma: no cover
                    raise RuntimeError()
            else:
                assert not all(x in self._dimensionless_numbers for x in ("Ek", "Re", "Ro"))

                if "Ro" in self._dimensionless_numbers and "Re" in self._dimensionless_numbers:
                    rotation_coefficient = 1.0 / self._dimensionless_numbers["Ro"]
                    viscous_coefficient = 1.0 / self._dimensionless_numbers["Re"]
                elif "Ro" in self._dimensionless_numbers and "Ek" in self._dimensionless_numbers:
                    rotation_coefficient = 1.0 / self._dimensionless_numbers["Ro"]
                    viscous_coefficient = self._dimensionless_numbers["Ek"] / self._dimensionless_numbers["Ro"]
                elif "Ek" in self._dimensionless_numbers and "Re" in self._dimensionless_numbers:
                    rotation_coefficient = 1.0 / (self._dimensionless_numbers["Ek"] *
                                                  self._dimensionless_numbers["Re"])
                    viscous_coefficient = 1.0 / self._dimensionless_numbers["Re"]
                elif "Ek" in self._dimensionless_numbers:
                    assert "Re" not in self._dimensionless_numbers
                    assert "Ro" not in self._dimensionless_numbers
                    rotation_coefficient = 1.0
                    viscous_coefficient = self._dimensionless_numbers["Ek"]
                elif "Ro" in self._dimensionless_numbers:
                    assert "Re" not in self._dimensionless_numbers
                    assert "Ek" not in self._dimensionless_numbers
                    rotation_coefficient = 1.0 / self._dimensionless_numbers["Ro"]
                    viscous_coefficient = 1.0
                else:  # pragma: no cover
                    raise RuntimeError()
                coefficients["coriolis_term"] = rotation_coefficient
                coefficients["euler_term"] = rotation_coefficient
                coefficients["pressure_term"] = 1.0
                coefficients["viscous_term"] = viscous_coefficient
            if "Fr" in self._dimensionless_numbers:
                coefficients["body_force_term"] = 1.0 / self._dimensionless_numbers["Fr"]**2
            else:
                coefficients["body_force_term"] = None
        # micropolar case
        else:
            assert all(x not in self._dimensionless_numbers for x in ("Ek", "Ro"))

            if not hasattr(self, "_equation_coefficients"):
                self._equation_coefficients = dict()

            self._equation_coefficients["momentum"] = dict()
            self._equation_coefficients["spin"] = dict()

            momentum_coefficients = self._equation_coefficients["momentum"]
            momentum_coefficients["convective_term"] = 1.0
            momentum_coefficients["coriolis_term"] = None
            momentum_coefficients["euler_term"] = None
            momentum_coefficients["pressure_term"] = 1.0
            if "Re" in self._dimensionless_numbers:
                momentum_coefficients["viscous_term"] = 1.0 / (1.0 - self._dimensionless_numbers["N"]**2) / \
                                                        self._dimensionless_numbers["Re"]
            else:  # pragma: no cover
                raise RuntimeError()
            if self._dimensionless_numbers["N"] > 0.0:
                momentum_coefficients["coupling_term"] = self._dimensionless_numbers["N"]**2 / \
                    (1.0 - self._dimensionless_numbers["N"]**2)

            spin_coefficients = self._equation_coefficients["spin"]
            spin_coefficients["convective_term"] = 1.0
            if all(x in self._dimensionless_numbers for x in ("Re", "L", "Th")):
                spin_coefficients["viscous_term"] = 1.0 / self._dimensionless_numbers["L"]**2 / \
                                                    self._dimensionless_numbers["Re"] / \
                                                    self._dimensionless_numbers["Th"]
            else:  # pragma: no cover
                raise RuntimeError()
            if all(x in self._dimensionless_numbers for x in ("Re", "M", "Th")):
                spin_coefficients["vol_viscous_term"] = 1.0 / self._dimensionless_numbers["M"]**2 / \
                                                        self._dimensionless_numbers["Re"] / \
                                                        self._dimensionless_numbers["Th"]
            if "coupling_term" in momentum_coefficients:
                spin_coefficients["coupling_term"] = momentum_coefficients["coupling_term"]

    def _read_dimensionless_number(self, d, key, alternative_key):
        assert hasattr(self, "_dimensionless_numbers")
        assert isinstance(d, dict)
        assert isinstance(key, str)
        if alternative_key is not None:
            assert isinstance(alternative_key, str)
        value = None
        if key in d:  # pragma: no cover
            assert alternative_key not in d
            value = d[key]
        if alternative_key in d:
            assert key not in d
            value = d[alternative_key]
        if value is not None:
            assert math.isfinite(value)
            if key not in ("N"):
                assert value > 0.0
            else:
                assert value >= 0.0
            self._dimensionless_numbers[key] = value

    def _set_dimensionless_number(self, key, value):
        assert self._closed is False
        assert hasattr(self, "_dimensionless_numbers")
        assert isinstance(key, str)
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value > 0.0
        self._dimensionless_numbers[key] = value

    def clear(self):
        self._closed = False
        self._equation_coefficients.clear()
        self._dimensionless_numbers.clear()

    def close(self):
        self._closed = True

    @property
    def equation_coefficients(self):
        self._compute_equation_coefficients()
        return self._equation_coefficients

    def get_file_suffix(self):
        assert hasattr(self, "_dimensionless_numbers")
        assert len(self._dimensionless_numbers) > 0
        suffix = ""
        for key, value in self._dimensionless_numbers.items():
            suffix += "_" + key + "{:1.3e}".format(value)
        return suffix

    def modify_dimensionless_number(self, key, value):
        assert key in self._dimensionless_numbers
        is_closed = self._closed
        self._closed = False
        self._set_dimensionless_number(key, value)
        self._closed = is_closed
        self._compute_equation_coefficients

    @property
    def Re(self):
        return self._dimensionless_numbers.get("Re")

    @Re.setter
    def Re(self, value):
        assert self._closed is False
        if "Ek" in self._dimensionless_numbers and \
                "Ro" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Overconstrained parameter set.")
        self._set_dimensionless_number("Re", value)

    @property
    def Fr(self):
        return self._dimensionless_numbers.get("Fr")

    @Fr.setter
    def Fr(self, value):
        assert self._closed is False
        self._set_dimensionless_number("Fr", value)

    @property
    def Ek(self):
        return self._dimensionless_numbers.get("Ek")

    @Ek.setter
    def Ek(self, value):
        assert self._closed is False
        if "Re" in self._dimensionless_numbers and \
                "Ro" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Overconstrained parameter set.")
        self._set_dimensionless_number("Ek", value)

    @property
    def Ro(self):
        return self._dimensionless_numbers.get("Ro")

    @Ro.setter
    def Ro(self, value):
        assert self._closed is False
        if "Re" in self._dimensionless_numbers and \
                "Ek" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Overconstrained parameter set.")
        self._set_dimensionless_number("Ro", value)

    @property
    def L(self):
        return self._dimensionless_numbers.get("L")

    @L.setter
    def L(self, value):
        assert self._closed is False
        if "Ro" in self._dimensionless_numbers or \
                "Ek" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Unknown non-dimenional form.")
        self._set_dimensionless_number("L", value)

    @property
    def M(self):
        return self._dimensionless_numbers.get("M")

    @M.setter
    def M(self, value):
        assert self._closed is False
        if "Ro" in self._dimensionless_numbers or \
                "Ek" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Unknown non-dimenional form.")
        self._set_dimensionless_number("M", value)

    @property
    def N(self):
        return self._dimensionless_numbers.get("N")

    @N.setter
    def N(self, value):
        assert self._closed is False
        if "Ro" in self._dimensionless_numbers or \
                "Ek" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Unknown non-dimenional form.")
        assert value < 1.0, "Parameter N must be less than unity."
        self._set_dimensionless_number("N", value)

    @property
    def Th(self):
        return self._dimensionless_numbers.get("Th")

    @Th.setter
    def Th(self, value):
        assert self._closed is False
        if "Ro" in self._dimensionless_numbers or \
                "Ek" in self._dimensionless_numbers:  # pragma: no cover
            raise RuntimeError("Unknown non-dimenional form.")
        self._set_dimensionless_number("Th", value)

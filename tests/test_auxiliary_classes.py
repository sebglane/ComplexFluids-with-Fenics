#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from auxiliary_classes import EquationCoefficientHandler

def test_equation_coefficients():
    eq01 = EquationCoefficientHandler(Reynolds=50.0)
    print(eq01)
    eq01.Fr = 10.0
    _ = eq01.equation_coefficients
    print(eq01)
    eq01.Ek = 25.0
    _ = eq01.equation_coefficients
    print(eq01)
    _ = eq01.Fr
    _ = eq01.Ek
    _ = eq01.Re

    eq02 = EquationCoefficientHandler()
    print(eq02)
    eq02.Ro = 1.0
    _ = eq02.equation_coefficients
    print(eq02)
    eq02.Ek = 25.0
    _ = eq02.equation_coefficients
    print(eq02)
    eq02.Fr = 10.0
    _ = eq02.equation_coefficients
    print(eq02)
    _ = eq01.Fr
    _ = eq01.Ek
    _ = eq01.Ro
    eq02.clear()

    eq03 = EquationCoefficientHandler()
    print(eq02)
    eq03.Ek = 1.0
    _ = eq03.equation_coefficients
    print(eq03)
    eq03.Re = 25.0
    _ = eq03.equation_coefficients
    print(eq03)
    eq03.modify_dimensionless_number("Re", 10.0)
    print(eq03.get_file_suffix())    


if __name__ == "__main__":
    test_equation_coefficients()

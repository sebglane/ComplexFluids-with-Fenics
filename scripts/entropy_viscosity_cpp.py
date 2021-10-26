from dolfin import Constant, CompiledExpression, FiniteElement, compile_cpp_code

__all__ = ['StabilizationParameterSD']


_streamline_diffusion_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>
using namespace dolfin;
class StabilizationParameterSD : public Expression
{
public:
  std::shared_ptr<GenericFunction> f_1, f_2;
  // constructor
  StabilizationParameterSD() : Expression() { }
  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    // Get dolfin cell
    // FIXME: Avoid dynamic allocation
    dolfin_assert(f_1->function_space());

    const std::shared_ptr<const Mesh> mesh = f_1->function_space()->mesh();
    const std::shared_ptr<const FiniteElement> element = f_1->function_space()->element();
    
    const Cell cell(*mesh, c.index);
    double h = cell.h();
        
    Array<double> vals_1(f_1->value_size());
    Array<double> vals_2(f_2->value_size());
    Array<double> x_(3);
    
    std::vector<double> coordinate_dofs;
    boost::multi_array<double, 2> coords;
    
    
    cell.get_coordinate_dofs(coordinate_dofs);
    
    
    std::vector<double> vertex_coords;
    cell.get_vertex_coordinates(vertex_coords);
    
    element->tabulate_dof_coordinates(coords, vertex_coords, cell);

    double max_val = 0;
    double sum_vals = 0;
    
    // Different for higher dimension, since all coordinates need to be updated
    for(auto i = coords.origin(); i < (coords.origin() + coords.num_elements()); ++i)  
    {
        x_[0] = i[0];
        
        f_1->eval(vals_1, x_, c);
        f_2->eval(vals_2, x_, c);
        sum_vals = vals_1[0] + vals_2[0];
        if (sum_vals > max_val)
        {
            max_val = sum_vals;
        }
    }
    values[0] = max_val;
  }
};

PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<StabilizationParameterSD,
             std::shared_ptr<StabilizationParameterSD>,
             Expression>
    (m, "StabilizationParameterSD")
    .def(pybind11::init<>())
    .def_readwrite("f_1", &StabilizationParameterSD::f_1)
    .def_readwrite("f_2", &StabilizationParameterSD::f_2);
}
"""

_expr = compile_cpp_code(_streamline_diffusion_cpp).StabilizationParameterSD


def StabilizationParameterSD(f_1, f_2):
    """
    Returns a subclass of :py:class:`dolfin.Expression` evaluating the cell-wise maximum of f_1+f_2.
    Represents an example for JIT-compiled expression: Could be used to efficiently compute local viscosity according to
    the entropy-viscosity-method.

    :param f_1: Function
    :param f_2: Function
    :return: Expression, cell-wise maximum
    """
    # TODO: Pass gradient of a function to/ compute gradient in CompiledExpression.
    mesh = f_1.function_space().mesh()
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    delta_sd = CompiledExpression(_expr(), element=element, domain=mesh)
    delta_sd.f_1 = f_1._cpp_object
    delta_sd.f_2 = f_2._cpp_object

    return delta_sd

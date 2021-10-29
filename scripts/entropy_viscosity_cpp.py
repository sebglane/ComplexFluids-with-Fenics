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
  // constructor
  StabilizationParameterSD() : Expression() { }

  // function pointers    
  std::shared_ptr<GenericFunction> function_01;
  
  std::shared_ptr<GenericFunction> function_02;
  

  // evaluate expression at a point in a cell
  void eval(Array<double>       &values,
            const Array<double> &x,
            const ufc::cell     &ufl_cell) const
  {
    const unsigned int dim = x.size();
    
    dolfin_assert(function_01->function_space());
    dolfin_assert(function_02->function_space());

    // get mesh
    const std::shared_ptr<const Mesh> mesh = function_01->function_space()->mesh();
    
    // get finite element
    const std::shared_ptr<const FiniteElement> element = function_01->function_space()->element();
    
    const Cell cell(*mesh, ufl_cell.index);
    const double h = cell.h();
        
    // get cell vertex coordinates (not coordinate dofs)
    std::vector<double> vertex_coords;
    cell.get_vertex_coordinates(vertex_coords);

    // get cell coordinate dofs not vertex coordinates
    std::vector<double> coordinate_dofs;
    cell.get_coordinate_dofs(coordinate_dofs);    
    
    // tabulate the coordinates of all dofs on an element
    boost::multi_array<double, 2> coords;
    element->tabulate_dof_coordinates(coords, vertex_coords, cell);


    dolfin_assert(function_01->value_size() == values.size());
    dolfin_assert(function_01->value_size() == function_02->value_size());
    const unsigned int n = function_values_01->value_size();
    
    Array<double> function_values_01(n);
    Array<double> function_values_02(n);
    Array<double> max_values(n);
    Array<double> x_(dim);
    
    // loop over all dof coordinates
    for (auto i = coords.origin(); i < (coords.origin() + coords.num_elements()); ++i)  
    {
     // assign current dof coordinate
     for (unsigned int d=0; d<dim; ++d)
       x_[d] = i[d];
     
     // evaluate functions at current dof coordinate
     function_01->eval(function_values_01, x_, ufl_cell);
     function_01->eval(function_values_02, x_, ufl_cell);
     
     // compute maximum for all components
     for (unsigned int j=0; j<n; ++j)
       max_val[j] = std::max(max_val[j], vals_1[j] + vals_2[j]);
    }
    
    // assign return value
    for (unsigned int j=0; j<n; ++j)
      values[j] = max_val[j];

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

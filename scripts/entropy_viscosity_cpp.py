from dolfin import Constant, CompiledExpression, FiniteElement, compile_cpp_code

__all__ = ['StabilizationParameterSD']


_streamline_diffusion_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/FunctionSpace.h>
using namespace dolfin;
class StabilizationParameterSD : public Expression
{
public:
  std::shared_ptr<GenericFunction> J;
  // constructor
  StabilizationParameterSD() : Expression() { }
  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
  {
    // Get dolfin cell and its diameter
    // FIXME: Avoid dynamic allocation
    dolfin_assert(J->function_space());
    const std::shared_ptr<const Mesh> mesh = J->function_space()->mesh();
    const Cell cell(*mesh, c.index);
    double h = cell.h();
    
    Array<double> nu(J->value_size());
    J->eval(nu, x, c);
    printf("%lu\\n", cell.num_vertices());
    printf("%d\\n", nu.size());
    values[0] = nu[0];
  }
};
PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<StabilizationParameterSD,
             std::shared_ptr<StabilizationParameterSD>,
             Expression>
    (m, "StabilizationParameterSD")
    .def(pybind11::init<>())
    .def_readwrite("J", &StabilizationParameterSD::J);
}
"""

_expr = compile_cpp_code(_streamline_diffusion_cpp).StabilizationParameterSD


def StabilizationParameterSD(J):
    """Returns a subclass of :py:class:`dolfin.Expression` representing
    streamline diffusion stabilization parameter.
    This kind of stabilization is convenient when a multigrid method is used
    for the convection term in the Navier-Stokes equation. The idea of the
    stabilization involves adding an additional term of the form::
      delta_sd*inner(dot(grad(u), w), dot(grad(v), w))*dx
    into the Navier-Stokes equation. Here ``u`` is a trial function, ``v`` is a
    test function and ``w`` defines so-called "wind" which is a known vector
    function.  Regularization parameter ``delta_sd`` is determined by the local
    mesh Peclet number (PE), see the implementation below.
    *Arguments*
        wind (:py:class:`dolfin.GenericFunction`)
            A vector field determining convective velocity.
        viscosity (:py:class:`dolfin.GenericFunction`)
            A scalar field determining dynamic viscosity.
    """
    mesh = J.function_space().mesh()
    element = FiniteElement("DG", mesh.ufl_cell(), 0)
    delta_sd = CompiledExpression(_expr(), element=element, domain=mesh)
    delta_sd.J = J._cpp_object
    return delta_sd

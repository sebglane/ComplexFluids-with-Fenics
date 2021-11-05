from dolfin import Constant, CompiledExpression, FiniteElement, compile_cpp_code, project, Dx, sqrt
from classes_EV import entropy, entropy_flux

__all__ = ['StabilizationParameterSD']

_streamline_diffusion_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>
#include <cmath>
using namespace dolfin;
class StabilizationParameterSD : public Expression
{
public:
  // constructor
  StabilizationParameterSD() : Expression() { }

  // function pointers
  std::shared_ptr<GenericFunction> J;
  
  std::shared_ptr<GenericFunction> J_;
  
  std::shared_ptr<GenericFunction> J_n;
  
  std::shared_ptr<GenericFunction> E_norm;

  std::shared_ptr<GenericFunction> dt;

  //std::shared_ptr<GenericFunction> c_e;

  //std::shared_ptr<GenericFunction> nu_max;

  
  // declare passed parameters
  double beta, c_e, nu_max;


  // evaluate expression at a point in a cell
  void eval(Array<double>		&values,
  			const Array<double>	&x,
            const ufc::cell		&c) const
  {
	const unsigned int dim = x.size();
	
    dolfin_assert(J->function_space());
    dolfin_assert(J_->function_space());
    dolfin_assert(J_n->function_space());
    dolfin_assert(E_norm->function_space());
    dolfin_assert(c_e->function_space());
    dolfin_assert(nu_max->function_space());
    
    
	// get mesh
    const std::shared_ptr<const Mesh> mesh = J->function_space()->mesh();
    
    
    // get finite element
    const std::shared_ptr<const FiniteElement> element = J->function_space()->element();

    const Cell cell(*mesh, c.index);
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

	
    dolfin_assert(J->value_size() == J_->value_size());
    dolfin_assert(J_->value_size() == J_n->value_size());
	
	//
    const unsigned int num_nodes = coords.num_elements();
    const unsigned int n = J->value_size();
	
	
    std::vector<double> basis_derivatives(dim * num_nodes * num_nodes);
    std::vector<double> derivatives(num_nodes);
    
    double *basis_derivatives_pointer;
    
    // buffer arrays for function values
    Array<double> vals_J(n);
    Array<double> vals_J_(n);
    Array<double> vals_J_n(n);
    
    Array<double> vals_Enorm(1);
    Array<double> val_dt(1);
    //Array<double> val_c_e(1);
    //Array<double> val_nu_max(1);
    
    // aggregated function values
    Array<double> vals_J_all(n * num_nodes);
    Array<double> vals_J__all(n * num_nodes);
    Array<double> vals_J_n_all(n * num_nodes);

	// evaluate normalized entropy variation
	E_norm->eval(vals_Enorm, x, c);
	dt->eval(val_dt, x, c);
	//c_e->eval(val_c_e, x, c);
	//nu_max->eval(val_nu_max, x, c);


    double Dh = 0;
    double max_Dh = 0;
    
   	// 
    std::vector<double> x_(dim);
    Array<double> x_function(dim);

	// pointers to (vertex) coordinates for evaluation of basis derivatives
    double* x_basis = &x_[0];
    const double *vertex_coords_basis = &vertex_coords[0];
    
    // index
    int j = 0;

    // Different for higher dimension, since all coordinates need to be updated
    for(auto i = coords.origin(); i < (coords.origin() + coords.num_elements()); ++i)  
    {
    	for (unsigned int d=0; d<dim; ++d) {
	    	x_[d] = i[d];
    	    x_function[d] = i[d];
        }
        
        // evaluate functions at current dof coordinate
        J->eval(vals_J, x_function, c);
        J_->eval(vals_J_, x_function, c);
        J_n->eval(vals_J_n, x_function, c);
        
        // aggregate values
        for (unsigned int e=0; e<dim; ++e) {
			vals_J_all[j * dim + e] = vals_J[e];
			vals_J__all[j * dim + e] = vals_J_[e];
			vals_J_n_all[j * dim + e] = vals_J_n[e];        	
        }
        
        // pointer to current basis derivatives and 
        basis_derivatives_pointer = &basis_derivatives[num_nodes * j];
		element->evaluate_basis_derivatives_all(1, basis_derivatives_pointer, x_basis, vertex_coords_basis, (int) cell.orientation());

        j++;
    }
    
     for(int k=0; k < num_nodes * num_nodes; ++k) {
		derivatives[k / num_nodes] = derivatives[k / num_nodes] + 0.5 * basis_derivatives[k] * pow(vals_J_all[k % num_nodes], 2);
		
		if ((k % num_nodes) == num_nodes - 1) {
			Dh = 0.25 * (3 * pow(vals_J_all[k / num_nodes], 2) - 4 * pow(vals_J__all[k / num_nodes], 2) + pow(vals_J_n_all[k / num_nodes], 2)) / val_dt[0] + beta * derivatives[k / num_nodes];
			max_Dh = std::max(abs(Dh), max_Dh);
		}
	}

    double e_norm = (vals_Enorm[0] == 0) ? 0.000001 : vals_Enorm[0];
    double nu_e = c_e * pow(h, 2) * max_Dh / e_norm;
    values[0] = std::min(nu_e, nu_max);
  }
};

PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<StabilizationParameterSD,
             std::shared_ptr<StabilizationParameterSD>,
             Expression>
    (m, "StabilizationParameterSD")
    .def(pybind11::init<>())
    .def_readwrite("J", &StabilizationParameterSD::J)
    .def_readwrite("J_", &StabilizationParameterSD::J_)
    .def_readwrite("J_n", &StabilizationParameterSD::J_n)
    .def_readwrite("beta", &StabilizationParameterSD::beta)
    .def_readwrite("dt", &StabilizationParameterSD::dt)
    .def_readwrite("c_e", &StabilizationParameterSD::c_e)
    .def_readwrite("nu_max", &StabilizationParameterSD::nu_max)
    .def_readwrite("E_norm", &StabilizationParameterSD::E_norm);
}
"""

_expr = compile_cpp_code(_streamline_diffusion_cpp).StabilizationParameterSD


def StabilizationParameterSD(J, J_, J_n, beta, c_e, nu_max, dt, E_norm):
	# TODO: Pass gradient of a function to/ compute gradient in CompiledExpression.
	f_space = J.function_space()
	mesh = f_space.mesh()
	element = FiniteElement("DG", mesh.ufl_cell(), 0)

	delta_sd = CompiledExpression(_expr(), element=element, domain=mesh)
	delta_sd.J = J._cpp_object
	delta_sd.J_ = J_._cpp_object
	delta_sd.J_n = J_n._cpp_object

	delta_sd.E_norm = E_norm._cpp_object
	delta_sd.dt = dt._cpp_object

	delta_sd.beta = beta
	delta_sd.c_e = c_e#._cpp_object
	delta_sd.nu_max = nu_max#._cpp_object

	return delta_sd

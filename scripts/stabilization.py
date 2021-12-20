from dolfin import CompiledExpression, FiniteElement, compile_cpp_code

__all__ = ['StabilizationParameter']

_entropy_viscosity_cpp = """
#include <pybind11/pybind11.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Edge.h>
using namespace dolfin;
class StabilizationParameter : public Expression
{
public:
  // constructor
  StabilizationParameter() : Expression() { }

  // function pointers
  std::shared_ptr<GenericFunction> J;
  
  std::shared_ptr<GenericFunction> J_old;
  
  std::shared_ptr<GenericFunction> J_old_old;
  
  std::shared_ptr<GenericFunction> E_norm;

  std::shared_ptr<GenericFunction> dt;
  
  std::shared_ptr<GenericFunction> beta;
  
  std::shared_ptr<GenericFunction> time_step;


  
  // declare passed parameters
  double c_e, c_max;
  
  // evaluate expression at a point in a cell
  void eval(Array<double>		&values,
            const Array<double> &x,
            const ufc::cell     &c) const
  {
    const unsigned int dim = x.size();

    dolfin_assert(J->function_space());
    dolfin_assert(J_old->function_space());
    dolfin_assert(J_old_old->function_space());
    dolfin_assert(E_norm->function_space());
    dolfin_assert(beta->function_space());
    
    
    // get mesh
    const std::shared_ptr<const Mesh> mesh = J->function_space()->mesh();
    
    // get dimension of J
    const unsigned int n = J->value_size();	
    
    
    // get finite element
    std::shared_ptr<const FiniteElement> element;
    if (n > 1) {
        element = J->function_space()->sub(0)->element();
    } else {
        element = J->function_space()->element();
    }

    const Cell cell(*mesh, c.index);
    double h = cell.h();

    // get cell vertex coordinates (not coordinate dofs)
    std::vector<double> vertex_coords;
    cell.get_vertex_coordinates(vertex_coords);
    
    // get cell coordinate dofs (not vertex coordinates)
    std::vector<double> coordinate_dofs;
    cell.get_coordinate_dofs(coordinate_dofs);

    // tabulate the coordinates of all dofs on an element
    boost::multi_array<double, 2> coords;
    element->tabulate_dof_coordinates(coords, vertex_coords, cell);


    dolfin_assert(J->value_size() == J_old->value_size());
    dolfin_assert(J_old->value_size() == J_old_old->value_size());
    dolfin_assert(dim == beta->value_size());

    // number of nodes on element
    const unsigned int num_nodes = coords.num_elements() / dim;

    // multidimensional array for v_k dJ_{ij} / dx_k on every node
    std::vector<std::vector<double>> basis_derivatives_matrix(num_nodes, std::vector<double>(dim * num_nodes));
    std::vector<std::vector<double>> inertia_flux_divergence(num_nodes, std::vector<double>(n));
    
    // buffer arrays for function values
    Array<double> vals_J(n);
    Array<double> vals_J_old(n);
    Array<double> vals_J_old_old(n);

    Array<double> vals_beta(dim);
    
    Array<double> vals_Enorm(1);
    Array<double> val_dt(1);
    Array<double> val_time_step(1);
    
    // aggregated function values
    std::vector<std::vector<double>> vals_J_all(num_nodes, std::vector<double>(n));
    std::vector<std::vector<double>> vals_J_old_all(num_nodes, std::vector<double>(n));
    std::vector<std::vector<double>> vals_J_old_old_all(num_nodes, std::vector<double>(n));

    std::vector<std::vector<double>> vals_beta_all(num_nodes, std::vector<double>(dim));

    // evaluate normalized entropy variation
    E_norm->eval(vals_Enorm, x, c);
    dt->eval(val_dt, x, c);
    time_step->eval(val_time_step, x, c);
    
    double Dh_t;
    double Dh_x;
    double beta_max = 0;
    double beta_abs;
    double max_Dh = 0;
    
    // find smallest edge on cell
    if (dim > 1) {
        for(EdgeIterator edge(cell); !edge.end(); ++edge) {
            h = std::min(h, edge->length());
        }
    }
    
    // coordinate vector (for function evaluation) and array (for basis derivative evaluation)
    std::vector<double> x_(dim);
    Array<double> x_function(dim);

    // Iterate over all nodes
    for(unsigned int i = 0; i < num_nodes; ++i)  
    {
        for (unsigned int d=0; d<dim; ++d) {
            x_[d] = coords[i][d];
            x_function[d] = x_[d];
        }
        
        // evaluate functions at current dof coordinate
        J->eval(vals_J, x_function, c);
        J_old->eval(vals_J_old, x_function, c);
        J_old_old->eval(vals_J_old_old, x_function, c);

        beta->eval(vals_beta, x_function, c);
        
        // aggregate beta
        beta_abs = 0;
        for (unsigned int d=0; d<dim; ++d) {
            vals_beta_all[i][d] = vals_beta[d];
            beta_abs += pow(vals_beta[d], 2);
        }
        beta_max = std::max(beta_max, sqrt(beta_abs));
        
        // aggregate J
        for (unsigned int f=0; f<n; ++f) {
            vals_J_all[i][f] = vals_J[f];
            vals_J_old_all[i][f] = vals_J_old[f];
            vals_J_old_old_all[i][f] = vals_J_old_old[f];
        }
        
        // get basis derivatives at node
        element->evaluate_basis_derivatives_all(1,
                                                &basis_derivatives_matrix[i][0],
                                                &x_[0],
                                                &vertex_coords[0],
                                                (int)cell.orientation());
    }
    
    // Compute partial/partial t ( 0.5 * J_ij J_ij ) + div( 0.5 * J_ij J_ij * beta ) for an incompressible flow, i.e. div(beta)=0
    // d/dt is approximated by the second-order backward finite difference scheme if time_step > 1
    // Use of chain rule: div( 0.5 * J_ij J_ij * beta ) = beta_k * J_ij * d/dx_k (J_ij)
    // vals_J_all[i][f] denotes J_f at the i-th node with f being the index of J rewritten as six-dimensional vector
    // inertia_flux_divergence[i][f] denotes beta_k * d/dx_k (J_f)
    
    // iterate over nodes
    for(unsigned int i=0; i<num_nodes; ++i) {
        Dh_t = 0;
        Dh_x = 0;

        // iterate over basis derivatives at node
        for(unsigned int k=0; k<num_nodes; ++k) {
            
            // iterate over J
            for(unsigned int f=0; f<n; ++f) {
                
                // iterate over spatial dimensions
                for(unsigned int d=0; d<dim; ++d) {
                    // compute (beta_d * dJ_f / x_d)| at x_k (k-th node on element)
                    inertia_flux_divergence[i][f] +=\
                        vals_beta_all[i][d] * basis_derivatives_matrix[i][k * dim + d] * vals_J_all[k][f];
                }
                
                // if derivative at node x_k is computed, add
                if (k == num_nodes - 1) {
                    if (val_time_step[0]==0) {
                        continue;
                    } else if (val_time_step[0]==1) {
                        Dh_t += 0.5 * (pow(vals_J_all[i][f], 2) - pow(vals_J_old_all[i][f], 2)) / val_dt[0];
                        Dh_x += vals_J_all[i][f] * inertia_flux_divergence[i][f];
                    } else {
                        Dh_t += 0.25 * (3. * pow(vals_J_all[i][f], 2) - 4. * pow(vals_J_old_all[i][f], 2)\
                            + pow(vals_J_old_old_all[i][f], 2)) / val_dt[0];
                        Dh_x += vals_J_all[i][f] * inertia_flux_divergence[i][f];                   
                    }
                }
            }
        }
        
        // Evaluate maximum entropy flux divergence of first i+1 nodes
        max_Dh = std::max(abs(Dh_t + Dh_x), max_Dh);
    }
    
    double e_norm, nu_e;
    double nu_max = h * c_max * beta_max;
    
    if (val_time_step[0]==0) {
        values[0] = nu_max;
    } else {
        e_norm = (vals_Enorm[0] == 0) ? 0.000001 : vals_Enorm[0];
        nu_e = c_e * pow(h, 2) * max_Dh / e_norm;
        values[0] = std::min(nu_e, nu_max);
    }
  }
};

PYBIND11_MODULE(SIGNATURE, m)
{
  pybind11::class_<StabilizationParameter,
             std::shared_ptr<StabilizationParameter>,
             Expression>
    (m, "StabilizationParameter")
    .def(pybind11::init<>())
    .def_readwrite("J", &StabilizationParameter::J)
    .def_readwrite("J_old", &StabilizationParameter::J_old)
    .def_readwrite("J_old_old", &StabilizationParameter::J_old_old)
    .def_readwrite("beta", &StabilizationParameter::beta)
    .def_readwrite("dt", &StabilizationParameter::dt)
    .def_readwrite("c_e", &StabilizationParameter::c_e)
    .def_readwrite("c_max", &StabilizationParameter::c_max)
    .def_readwrite("time_step", &StabilizationParameter::time_step)
    .def_readwrite("E_norm", &StabilizationParameter::E_norm);
}
"""

_expr = compile_cpp_code(_entropy_viscosity_cpp).StabilizationParameter


def StabilizationParameter(J, J_, J_n, beta, c_e, c_max, dt, E_norm, time_step):
	"""
	Computes the cellwise maximum of the entropy-viscosity-residual for the entropy pair E(J)=J**J, F(J)=(J**J) beta:

	R = ||E_t + div(F)||_(inf,K),

	the absolute transport velocity:

	nu_max = c_max * h_k * ||beta||_(inf,K),

	and the resulting entropy-viscosity:

	nu_h = min(nu_max, ce * h_k^2 * R / E_norm).

	Some variables are passed as dolfin Constants to enable changing them globally.

	:param J: dolfin Function, inertia tensor at given time_step
	:param J_: dolfin Function, inertia tensor at  time_step - 1
	:param J_n: dolfin Function, inertia tensor at  time_step - 2
	:param beta: dolfin Function, velocity vector at given time_step
	:param c_e: float, control parameter
	:param c_max: float, control parameter
	:param dt: dolfin constant, time step size
	:param E_norm: dolfin constant, normalized entropy variation
	:param time_step:
	:return: dolfin expression
	"""
	# TODO: Check for symmetry of J and adjust the code accordingly
	f_space = J.function_space()
	mesh = f_space.mesh()
	element = FiniteElement("DG", mesh.ufl_cell(), 0)

	delta_sd = CompiledExpression(_expr(), element=element, domain=mesh)
	delta_sd.J = J._cpp_object
	delta_sd.J_old = J_._cpp_object
	delta_sd.J_old_old = J_n._cpp_object

	delta_sd.E_norm = E_norm._cpp_object
	delta_sd.dt = dt._cpp_object
	delta_sd.time_step = time_step._cpp_object

	delta_sd.beta = beta._cpp_object

	delta_sd.c_e = c_e
	delta_sd.c_max = c_max

	return delta_sd

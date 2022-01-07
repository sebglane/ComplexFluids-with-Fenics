/* ---------------------------------------------------------------------
 *
 * Program solving example 3.1.3 of Guermond et al., JCP 230 (2011)
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Sebastian Glane, 2022
 *
 */


#include <deal.II/base/discrete_time.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <deal.II/base/time_stepping.h>


namespace Guermond
{

using namespace dealii;


template <int dim>
class InitialCondition: public Function<dim>
{
public:

  InitialCondition();

  virtual double value(const Point<dim> &point,
                       unsigned int component) const;

  virtual void value_list(const std::vector<Point<dim>> &point_list,
                          std::vector<double>           &value_list,
                          const unsigned int component) const;

};



template <int dim>
InitialCondition<dim>::InitialCondition()
:
Function<dim>()
{}


template <>
double InitialCondition<1>::value
(const Point<1> &point,
 unsigned int /* component */) const
{
  if (std::abs(2.0 * point[0] - 0.3) <= 0.25)
    return std::exp(-300.0 * std::pow(2.0 * point[0] - 0.3, 2));
  else if (std::abs(2.0 * point[0] - 0.9) <= 0.2)
    return 1.0;
  else if (std::abs(2.0 * point[0] - 1.6) <= 0.2)
    return std::sqrt(1.0 - std::pow((2.0 * point[0] - 1.6)/ 0.2, 2));
  else
    return 0.0;
}


template <>
void InitialCondition<1>::value_list
(const std::vector<Point<1>> &point_list,
 std::vector<double>         &value_list,
 const unsigned int /* component */) const
 {

  AssertDimension(point_list.size(), value_list.size());

  for (unsigned int i=0; i<point_list.size(); ++i)
  {
    const double x{point_list[i][0]};

    if (std::abs(2.0 * x - 0.3) <= 0.25)
      value_list[i] = std::exp(-300.0 * std::pow(2.0 * x - 0.3, 2));
    else if (std::abs(2.0 * x - 0.9) <= 0.2)
      value_list[i] = 1.0;
    else if (std::abs(2.0 * x - 1.6) <= 0.2)
      value_list[i] = std::sqrt(1.0 - std::pow((2.0 * x - 1.6)/ 0.2, 2));
    else
      value_list[i] = 0.0;
  }

 }



enum class ResidualComputation
{
  gradient_based,
  gradient_based_mean_value,
  standard
};



struct RunTimeParameters
{
  RunTimeParameters();

  RunTimeParameters(const std::string &parameter_filename);

  static void declare_parameters(ParameterHandler &prm);

  void parse_parameters(ParameterHandler &prm);

  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const RunTimeParameters &prm);

  unsigned int  fe_degree;

  unsigned int  problem_size;

  double        initial_time_step;

  double        entropy_stabilization;

  double        standard_stabilization;

  double        start_time;

  double        final_time;

  ResidualComputation residual_computation;

  bool          verbose;
};



RunTimeParameters::RunTimeParameters()
:
fe_degree(1),
problem_size(200),
initial_time_step(1e-3),
entropy_stabilization(0.25),
standard_stabilization(0.5),
start_time(0.0),
final_time(1.0),
residual_computation(ResidualComputation::gradient_based),
verbose(false)
{}



RunTimeParameters::RunTimeParameters(const std::string &parameter_filename)
:
RunTimeParameters()
{
  ParameterHandler prm;
  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;
    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());
    prm.print_parameters(parameter_out,
                         ParameterHandler::OutputStyle::Text);

    AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}



void RunTimeParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("FE degree",
                    "1",
                    Patterns::Integer(0));

  prm.declare_entry("Problem size",
                    "200",
                    Patterns::Integer(0));

  prm.declare_entry("Initial time step",
                    "1e-3",
                    Patterns::Double(0.0));

  prm.declare_entry("Entropy stabilization coefficient",
                    "0.25",
                    Patterns::Double(0.0));

  prm.declare_entry("Standard stabilization coefficient",
                    "0.5",
                    Patterns::Double(0.0));

  prm.declare_entry("Start time",
                    "0.0",
                    Patterns::Double(0.0));

  prm.declare_entry("Final time",
                    "1.0",
                    Patterns::Double(0.0));

  prm.declare_entry("Entropy residual computation",
                    "Standard",
                    Patterns::Selection("Gradient|GradientMeanValue|Standard|"
                                        "gradient|gradient_mean_value|standard"));

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());
}

void RunTimeParameters::parse_parameters(ParameterHandler &prm)
{
  fe_degree = prm.get_integer("FE degree");
  Assert(fe_degree > 0,
         ExcLowerRangeType<unsigned int>(fe_degree, 0));

  problem_size = prm.get_integer("Problem size");
  Assert(problem_size > 0,
         ExcLowerRangeType<unsigned int>(problem_size, 0));

  initial_time_step = prm.get_double("Initial time step");
  Assert(initial_time_step > 0,
         ExcLowerRangeType<double>(initial_time_step, 0));


  entropy_stabilization = prm.get_double("Entropy stabilization coefficient");
  Assert(entropy_stabilization > 0,
         ExcLowerRangeType<double>(entropy_stabilization, 0));

  standard_stabilization = prm.get_double("Standard stabilization coefficient");
  Assert(standard_stabilization > 0,
         ExcLowerRangeType<double>(standard_stabilization, 0));

  start_time = prm.get_double("Start time");
  Assert(start_time >= 0.0, ExcLowerRangeType<double>(start_time, 0.0));

  final_time = prm.get_double("Final time");
  Assert(final_time > 0.0, ExcLowerRangeType<double>(final_time, 0.0));
  Assert(final_time > start_time, ExcLowerRangeType<double>(final_time, start_time));
  Assert(initial_time_step <= final_time,
         ExcLowerRangeType<double>(initial_time_step, final_time));

  std::string residual_computation_type;
  residual_computation_type = prm.get("Entropy residual computation");

  std::transform(residual_computation_type.begin(), residual_computation_type.end(),
                 residual_computation_type.begin(), [](unsigned char c){ return std::tolower(c); });

  if (residual_computation_type == "standard")
    residual_computation = ResidualComputation::standard;
  else if (residual_computation_type == "gradient")
    residual_computation = ResidualComputation::gradient_based;
  else if (residual_computation_type == "gradientmeanvalue" ||
           residual_computation_type == "gradient_mean_value")
    residual_computation = ResidualComputation::gradient_based_mean_value;
  else
    AssertThrow(false,
                ExcMessage("Unexpected string for type of entropy resiudal computation."));

  verbose = prm.get_bool("Verbose");

}

template<typename Stream>
Stream& operator<<(Stream &stream, const RunTimeParameters &prm)
{
  const size_t column_width[2] ={ 40, 20 };

  constexpr size_t line_width = 63;

  const char header[] = "+------------------------------------------+"
                        "----------------------+";

  auto add_line = [&](const char first_column[],
                      const auto second_column)
  {
    stream << "| "
           << std::setw(column_width[0]) << first_column
           << " | "
           << std::setw(column_width[1]) << second_column
           << " |"
           << std::endl;
  };

  stream << std::left << header << std::endl;

  stream << "| "
         << std::setw(line_width)
         << "Run time parameters"
         << " |"
         << std::endl;

  stream << header << std::endl;

  add_line("FE degree", prm.fe_degree);
  add_line("Problem size", prm.problem_size);
  add_line("Initial time step", prm.initial_time_step);
  add_line("Start time", prm.start_time);
  add_line("Final time", prm.final_time);

  std::string residual_computation_type;
  switch (prm.residual_computation)
  {
    case ResidualComputation::standard:
      residual_computation_type = "Standard";
      break;
    case ResidualComputation::gradient_based:
      residual_computation_type = "Gradient";
      break;
    case ResidualComputation::gradient_based_mean_value:
      residual_computation_type = "Gradient mean value";
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Given ResidualComputation is not known or cannot be "
                             "interpreted."));
    break;
  }
  add_line("Entropy residual computation type", residual_computation_type);

  add_line("Verbose", (prm.verbose? "true": "false"));

  stream << header;

  return (stream);
}



template <int dim>
class EntropyViscosity
{
public:
  EntropyViscosity(const RunTimeParameters &parameters);

  void run();

private:
  void make_grid();

  void setup_dofs();

  void setup_system();

  void setup_vectors();

  void assemble_system();

  void apply_initial_condition();

  double get_entropy_variation() const;

  double get_cfl_number(const double time_step) const;

  double compute_viscosity(const std::vector<double>         &solution_values,
                           const std::vector<double>         &old_solution_values,
                           const std::vector<double>         &old_old_solution_values,
                           const std::vector<Tensor<1,dim>>  &solution_gradients,
                           const std::vector<Tensor<1,dim>>  &velocity_values,
                           const double                       average_solution_value,
                           const double                       global_entropy_variation,
                           const double                       cell_diameter,
                           const unsigned int                 step_number) const;

  Vector<double> evaluate_rhs(const double          time,
                              const Vector<double> &old_solution,
                              const double          average_solution_value,
                              const double          global_entropy_variation,
                              const unsigned int    step_number) const;

  void output_results(const double                     time,
                      const unsigned int               time_step,
                      TimeStepping::runge_kutta_method method) const;

  void explicit_method(const TimeStepping::runge_kutta_method method,
                       const double                           initial_time,
                       const double                           final_time,
                       const double                           time_step);

  unsigned int
  embedded_explicit_method(const TimeStepping::runge_kutta_method method,
                           const double       initial_time,
                           const double       final_time,
                           const double       initial_time_step);

  const RunTimeParameters &prm;

  const ConstantTensorFunction<1, dim>  velocity_function;

  Triangulation<dim> triangulation;

  const FE_Q<dim> fe;

  DoFHandler<dim> dof_handler;

  AffineConstraints<double> constraint_matrix;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> mass_matrix;

  SparseDirectUMFPACK inverse_mass_matrix;

  Vector<double>  solution;

  Vector<double>  old_solution;

  Vector<double>  old_old_solution;

  double old_time_step;

  double old_old_time_step;

};


template <>
EntropyViscosity<1>::EntropyViscosity
(const RunTimeParameters &parameters)
:
prm(parameters),
velocity_function(Tensor<1, 1>({1.0})),
fe(prm.fe_degree),
dof_handler(triangulation)
{}



template <>
EntropyViscosity<2>::EntropyViscosity
(const RunTimeParameters &parameters)
:
prm(parameters),
velocity_function(Tensor<1, 2>({1.0, 0.0})),
fe(prm.fe_degree),
dof_handler(triangulation)
{}



template <>
EntropyViscosity<3>::EntropyViscosity
(const RunTimeParameters &parameters)
:
prm(parameters),
velocity_function(Tensor<1, 3>({1.0, 0.0, 0.0})),
fe(prm.fe_degree),
dof_handler(triangulation)
{}



template <int dim>
void EntropyViscosity<dim>::make_grid()
{
  const unsigned int repetitions{prm.problem_size / (dim * fe.get_degree())};

  GridGenerator::subdivided_hyper_cube(triangulation,
                                       repetitions,
                                       0.0,
                                       1.0,
                                       true);

  const types::boundary_id left_bndry_id{0};
  const types::boundary_id right_bndry_id{1};

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator >>
  periodicity_vector;

  GridTools::collect_periodic_faces(triangulation,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);
  triangulation.add_periodicity(periodicity_vector);

  GridTools::distort_random(0.05, triangulation);

  const double minimum_cell_diameter{GridTools::minimal_cell_diameter(triangulation)};
  const double maximum_cell_diameter{GridTools::maximal_cell_diameter(triangulation)};

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
  std::cout << "Ratio of maximum to minimum cell diameter: "
            << maximum_cell_diameter / minimum_cell_diameter
            << std::endl;
}


template <int dim>
void EntropyViscosity<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "FE degree: "
            << fe.get_degree()
            << std::endl;
  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  constraint_matrix.clear();
  const types::boundary_id left_bndry_id{0};
  const types::boundary_id right_bndry_id{1};

  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator >>
  periodicity_vector;
  GridTools::collect_periodic_faces(dof_handler,
                                    left_bndry_id,
                                    right_bndry_id,
                                    0,
                                    periodicity_vector);

  DoFTools::make_periodicity_constraints<dim, dim>(periodicity_vector,
                                                   constraint_matrix);
  constraint_matrix.close();
}



template <int dim>
void EntropyViscosity<dim>::setup_system()
{
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint_matrix);
  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);
}



template <int dim>
void EntropyViscosity<dim>::setup_vectors()
{
  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  old_old_solution.reinit(dof_handler.n_dofs());
}


template <int dim>
void EntropyViscosity<dim>::assemble_system()
{
  mass_matrix = 0.;

  const QGauss<dim> quadrature_formula(fe.get_degree() + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values|update_gradients|update_JxW_values);

  FullMatrix<double> cell_mass_matrix(fe_values.dofs_per_cell,
                                      fe_values.dofs_per_cell);

  std::vector<double> phi_values(fe_values.dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(fe_values.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_mass_matrix = 0.;

    fe_values.reinit(cell);

    for (const auto q: fe_values.quadrature_point_indices())
    {
      const double JxW_value{fe_values.JxW(q)};

      for (const auto i: fe_values.dof_indices())
        phi_values[i] = fe_values.shape_value(i, q);

      for (const auto i: fe_values.dof_indices())
        for (const auto j: fe_values.dof_indices())
          cell_mass_matrix(i, j) += phi_values[i] * phi_values[j] * JxW_value;
    }

    cell->get_dof_indices(local_dof_indices);

    constraint_matrix.distribute_local_to_global(cell_mass_matrix,
                                                 local_dof_indices,
                                                 mass_matrix);
  }

  inverse_mass_matrix.initialize(mass_matrix);
}



template <int dim>
void EntropyViscosity<dim>::apply_initial_condition()
{
  InitialCondition<dim> initial_condition;

  VectorTools::interpolate(dof_handler,
                           initial_condition,
                           solution);
}


template <int dim>
double EntropyViscosity<dim>::compute_viscosity
(const std::vector<double>         &solution_values,
 const std::vector<double>         &old_solution_values,
 const std::vector<double>         &old_old_solution_values,
 const std::vector<Tensor<1,dim>>  &solution_gradients,
 const std::vector<Tensor<1,dim>>  &velocity_values,
 const double                       average_solution_value,
 const double                       global_entropy_variation,
 const double                       cell_diameter,
 const unsigned int                 step_number) const
{
  AssertDimension(solution_values.size(), old_solution_values.size());
  AssertDimension(solution_values.size(), old_solution_values.size());
  AssertDimension(solution_values.size(), old_old_solution_values.size());
  AssertDimension(solution_values.size(), solution_gradients.size());
  AssertDimension(solution_values.size(), velocity_values.size());

  AssertIsFinite(global_entropy_variation);
  AssertIsFinite(cell_diameter);
  AssertIsFinite(old_time_step);
  AssertIsFinite(old_old_time_step);

  Assert(global_entropy_variation >= 0, ExcLowerRangeType(0.0, global_entropy_variation));
  Assert(cell_diameter > 0, ExcLowerRangeType(0.0, cell_diameter));

  std::vector<double> alpha(3);
  if (step_number > 1)
  {
    const double omega{old_time_step / old_old_time_step};

    alpha[0] = (1.0 + 2.0 * omega) / (1.0 + omega);
    alpha[1] = -(1.0 + omega);
    alpha[2] = omega * omega / (1.0 + omega);
  }
  else
  {
    alpha[0] = 1.0;
    alpha[1] = -1.0;
    alpha[2] = 0.0;
  }

  auto entropy_function = [](const double x){ return (0.5 * x * x); };

  double max_entropy_residual = 0;
  double max_velocity = 0;

  double residual;

  for (unsigned int q = 0; q < solution_values.size(); ++q)
  {
    switch (prm.residual_computation)
    {
      case ResidualComputation::standard:
      {
        double entropy_residual = (alpha[0] * entropy_function(solution_values[q]) +
                                   alpha[1] * entropy_function(old_solution_values[q]) +
                                   alpha[2] * entropy_function(old_old_solution_values[q])) /
                                  old_time_step;
        entropy_residual += velocity_values[q] * solution_gradients[q] * solution_values[q];

        residual = std::abs(entropy_residual);

        break;
      }
      case ResidualComputation::gradient_based:
      {
        double gradient_residual = (alpha[0] * solution_values[q] +
                                    alpha[1] * old_solution_values[q] +
                                    alpha[2] * old_old_solution_values[q]) /
                                   old_time_step;
        gradient_residual += velocity_values[q] * solution_gradients[q] * solution_values[q];
        residual = std::abs(gradient_residual) * std::abs(solution_values[q]);
        break;
      }
      case ResidualComputation::gradient_based_mean_value:
      {
        double gradient_residual = (alpha[0] * solution_values[q] +
                                    alpha[1] * old_solution_values[q] +
                                    alpha[2] * old_old_solution_values[q]) /
                                   old_time_step;
        gradient_residual += velocity_values[q] * solution_gradients[q] * solution_values[q];

        residual = std::abs(gradient_residual) * std::abs(solution_values[q] - average_solution_value);
        break;
      }
      default:
        AssertThrow(false,
                    ExcMessage("Given ResidualComputation is not known or cannot be "
                               "interpreted."));
        break;
    }

    max_entropy_residual = std::max(max_entropy_residual, residual);
    max_velocity = std::max(max_velocity, velocity_values[q].norm());
  }

  const double max_viscosity{prm.standard_stabilization * max_velocity * cell_diameter};

  if (step_number == 0)
    return (max_viscosity);
  else
  {
    const double entropy_viscosity{prm.entropy_stabilization * std::pow(cell_diameter, 2) *
                                   max_entropy_residual / global_entropy_variation};

    return (std::min(max_viscosity, entropy_viscosity));
  }
}



template <int dim>
double EntropyViscosity<dim>::get_entropy_variation() const
{
  const QGauss<dim>  quadrature_formula(fe.get_degree() + 1);
  FEValues<dim>      fe_values(fe,
                               quadrature_formula,
                               update_values|update_JxW_values);

  std::vector<double> solution_values(fe_values.n_quadrature_points);

  double  min_entropy = std::numeric_limits<double>::max();
  double  max_entropy = -std::numeric_limits<double>::max();
  double  volume = 0.0;
  double  entropy_integral = 0.0;

  for (const auto &cell: dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    fe_values.get_function_values(solution,
                                  solution_values);

    for (const auto q: fe_values.quadrature_point_indices())
    {
      const double entropy = 0.5 * solution_values[q] * solution_values[q];

      min_entropy = std::min(min_entropy, entropy);
      max_entropy = std::max(max_entropy, entropy);

      volume += fe_values.JxW(q);
      entropy_integral += entropy * fe_values.JxW(q) ;
    }
  }
  AssertIsFinite(min_entropy);
  AssertIsFinite(max_entropy);
  AssertIsFinite(volume);
  AssertIsFinite(entropy_integral);

  Assert(min_entropy >= 0, ExcLowerRangeType(0.0, min_entropy));
  Assert(max_entropy > 0, ExcLowerRangeType(0.0, max_entropy));
  Assert(volume > 0, ExcLowerRangeType(0.0, volume));
  Assert(entropy_integral > 0, ExcLowerRangeType(0.0, entropy_integral));

  const double average_entropy{entropy_integral / volume};
  AssertIsFinite(average_entropy);
  Assert(average_entropy > 0, ExcLowerRangeType(0.0, average_entropy));

  Assert(min_entropy <= average_entropy, ExcLowerRangeType(min_entropy, average_entropy));
  Assert(average_entropy <= max_entropy, ExcLowerRangeType(average_entropy, max_entropy));

  const double entropy_variation = std::max(max_entropy - average_entropy,
                                            average_entropy - min_entropy);

  return (entropy_variation);
}



template <int dim>
double EntropyViscosity<dim>::get_cfl_number(const double time_step) const
{
  const QIterated<dim> quadrature_formula(QTrapezoid<1>(),
                                          fe.get_degree());

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_quadrature_points);

  std::vector<Tensor<1, dim>> velocity_values(fe_values.n_quadrature_points);

  double max_cfl = 0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);

    velocity_function.value_list(fe_values.get_quadrature_points(),
                                 velocity_values);

    double max_cell_velocity = 0;

    for (const auto q: fe_values.quadrature_point_indices())
      max_cell_velocity = std::max(max_cell_velocity, velocity_values[q].norm());

    max_cfl = std::max(max_cfl, max_cell_velocity / cell->diameter());
  }

  max_cfl *= time_step * double(fe.get_degree());

  return (max_cfl);
}



template <int dim>
Vector<double> EntropyViscosity<dim>::evaluate_rhs
(const double           /* time */,
 const Vector<double>  &evaluation_point,
 const double           average_solution_value,
 const double           global_entropy_variation,
 const unsigned int     step_number) const
{
  Vector<double> rhs_vector(dof_handler.n_dofs());
  rhs_vector = 0;

  const QGauss<dim> quadrature_formula(fe.get_degree() + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values|update_gradients|
                          update_quadrature_points|update_JxW_values);

  Vector<double> cell_rhs(fe_values.dofs_per_cell);

  std::vector<double>         phi_values(fe_values.dofs_per_cell);
  std::vector<Tensor<1,dim>>  phi_gradients(fe_values.dofs_per_cell);

  std::vector<double>         solution_values(fe_values.n_quadrature_points);
  std::vector<double>         old_solution_values(fe_values.n_quadrature_points);
  std::vector<double>         old_old_solution_values(fe_values.n_quadrature_points);

  std::vector<Tensor<1,dim>>  solution_gradients(fe_values.n_quadrature_points);

  std::vector<Tensor<1,dim>>  velocity_values(fe_values.n_quadrature_points);

  std::vector<types::global_dof_index> local_dof_indices(fe_values.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell_rhs = 0;

    fe_values.reinit(cell);

    // compute values of previous solutions (required for cell viscosity)
    fe_values.get_function_values(solution,
                                  solution_values);
    fe_values.get_function_values(old_solution,
                                  old_solution_values);
    fe_values.get_function_values(old_old_solution,
                                  old_old_solution_values);

    fe_values.get_function_gradients(solution,
                                     solution_gradients);

    // compute values of velocity
    velocity_function.value_list(fe_values.get_quadrature_points(),
                                 velocity_values);

    // compute cell viscosity
    const double cell_viscosity = compute_viscosity(solution_values,
                                                    old_solution_values,
                                                    old_old_solution_values,
                                                    solution_gradients,
                                                    velocity_values,
                                                    average_solution_value,
                                                    global_entropy_variation,
                                                    cell->diameter(),
                                                    step_number);
    AssertIsFinite(cell_viscosity);
    Assert(cell_viscosity > 0, ExcLowerRangeType(0.0, cell_viscosity));

    // compute values of evaluation point
    fe_values.get_function_values(evaluation_point,
                                  solution_values);
    fe_values.get_function_gradients(evaluation_point,
                                     solution_gradients);

    for (const auto q: fe_values.quadrature_point_indices())
    {
      const double JxW_value{fe_values.JxW(q)};

      for (const auto i: fe_values.dof_indices())
      {
        phi_values[i] = fe_values.shape_value(i, q);
        phi_gradients[i] = fe_values.shape_grad(i, q);
      }

      for (const auto i: fe_values.dof_indices())
        cell_rhs(i) -= (solution_gradients[q] *
                        velocity_values[q] *
                        phi_values[i] +
                        cell_viscosity *
                        solution_gradients[q] *
                        phi_gradients[i]) *
                       JxW_value;
    }
    cell->get_dof_indices(local_dof_indices);
    constraint_matrix.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 rhs_vector);
  }

  Vector<double> value(dof_handler.n_dofs());
  inverse_mass_matrix.vmult(value, rhs_vector);

  return value;
}



template <int dim>
void EntropyViscosity<dim>::output_results
(const double                     time,
 const unsigned int               time_step,
 TimeStepping::runge_kutta_method method) const
{
  std::string method_name;

  switch (method)
  {
    case TimeStepping::FORWARD_EULER:
    {
      method_name = "forward_euler";
      break;
    }
    case TimeStepping::RK_THIRD_ORDER:
    {
      method_name = "rk3";
      break;
    }
    case TimeStepping::RK_CLASSIC_FOURTH_ORDER:
    {
      method_name = "rk4";
      break;
    }
    case TimeStepping::BACKWARD_EULER:
    {
      method_name = "backward_euler";
      break;
    }
    case TimeStepping::IMPLICIT_MIDPOINT:
    {
      method_name = "implicit_midpoint";
      break;
    }
    case TimeStepping::SDIRK_TWO_STAGES:
    {
      method_name = "sdirk";
      break;
    }
    case TimeStepping::HEUN_EULER:
    {
      method_name = "heun_euler";
      break;
    }
    case TimeStepping::BOGACKI_SHAMPINE:
    {
      method_name = "bogacki_shampine";
      break;
    }
    case TimeStepping::DOPRI:
    {
      method_name = "dopri";
      break;
    }
    case TimeStepping::FEHLBERG:
    {
      method_name = "fehlberg";
      break;
    }
    case TimeStepping::CASH_KARP:
    {
      method_name = "cash_karp";
      break;
    }
    default:
    {
      break;
    }
  }

  DataOut<dim>  data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches(fe.get_degree());

  {
    data_out.set_flags(DataOutBase::VtkFlags(time, time_step));

    const std::string filename = "solution_" + method_name + "-" +
                                 Utilities::int_to_string(time_step, 3) +
                                 ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    static std::vector<std::pair<double, std::string>> times_and_names;

    static std::string method_name_prev = "";
    static std::string pvd_filename;
    if (method_name_prev != method_name)
    {
      times_and_names.clear();
      method_name_prev = method_name;
      pvd_filename     = "solution_" + method_name + ".pvd";
    }
    times_and_names.emplace_back(time, filename);

    std::ofstream pvd_output(pvd_filename);
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
  {
    data_out.set_flags(DataOutBase::GnuplotFlags());

    const std::string filename = "solution_" + method_name + "-" +
                                 Utilities::int_to_string(time_step, 3) +
                                 ".gpl";

    std::ofstream output(filename);
    data_out.write_gnuplot(output);
  }

}


template <int dim>
unsigned int EntropyViscosity<dim>::embedded_explicit_method
(const TimeStepping::runge_kutta_method method,
 const double       initial_time,
 const double       final_time,
 const double       initial_time_step)
{
  const double coarsen_param = 2.0;
  const double refine_param  = 0.5;
  const double minimum_time_step = 1.0e-3 * initial_time_step;
  const double maximum_time_step = 1.0e6 * initial_time_step;
  const double refine_tol    = 1e-1;
  const double coarsen_tol   = 1e-3;

  apply_initial_condition();

  Vector<double>  aux_solution(solution);
  constraint_matrix.distribute(aux_solution);

  TimeStepping::EmbeddedExplicitRungeKutta<Vector<double>>
  embedded_explicit_runge_kutta(method,
                                coarsen_param,
                                refine_param,
                                minimum_time_step,
                                maximum_time_step,
                                refine_tol,
                                coarsen_tol);

  output_results(initial_time, 0, method);

  DiscreteTime time(initial_time, final_time, initial_time_step);

  double average_solution_value;
  double global_entropy_variation;
  unsigned int step_number;

  auto worker =
  [this, &average_solution_value, &global_entropy_variation, &step_number]
   (const double time, const Vector<double> &y)
   {
      return this->evaluate_rhs(time,
                                y,
                                average_solution_value,
                                global_entropy_variation,
                                step_number);
   };

  old_time_step = initial_time_step;
  old_old_time_step = initial_time_step;

  while (time.is_at_end() == false)
  {
    old_old_time_step = old_time_step;
    old_time_step = time.get_previous_step_size();

    average_solution_value = VectorTools::compute_mean_value(dof_handler,
                                                             QGauss<dim>(fe.get_degree() + 1),
                                                             solution,
                                                             0);
    global_entropy_variation = get_entropy_variation();
    step_number = time.get_step_number();

    if (prm.verbose)
      std::cout << "Step number: " << step_number << ", "
                << "Current time: " << time.get_current_time() << ", "
                << "Next time step: " << time.get_next_step_size()
                << std::endl;

    const double new_time =
        embedded_explicit_runge_kutta.evolve_one_time_step(worker,
                                                           time.get_current_time(),
                                                           time.get_next_step_size(),
                                                           aux_solution);
    constraint_matrix.distribute(aux_solution);

    old_old_solution = old_solution;
    old_solution = solution;
    solution = aux_solution;

    time.set_next_step_size(new_time - time.get_current_time());
    time.advance_time();

    constraint_matrix.distribute(solution);

    if (time.get_step_number() % 100 == 0 || time.is_at_end())
      output_results(time.get_current_time(),
                     time.get_step_number(),
                     method);

    time.set_desired_next_step_size(embedded_explicit_runge_kutta.get_status().delta_t_guess);
  }

  return time.get_step_number();
}



template <int dim>
void EntropyViscosity<dim>::explicit_method
(const TimeStepping::runge_kutta_method method,
 const double                           initial_time,
 const double                           final_time,
 const double                           time_step)
{
  apply_initial_condition();

  Vector<double>  aux_solution(solution);
  constraint_matrix.distribute(aux_solution);

  TimeStepping::ExplicitRungeKutta<Vector<double>> explicit_runge_kutta(method);

  output_results(initial_time, 0, method);

  DiscreteTime time(initial_time, final_time, time_step);

  double average_solution_value;
  double global_entropy_variation;
  unsigned int step_number;

  auto worker =
  [this, &average_solution_value, &global_entropy_variation, &step_number]
   (const double time, const Vector<double> &y)
   {
      return this->evaluate_rhs(time,
                                y,
                                average_solution_value,
                                global_entropy_variation,
                                step_number);
   };

  old_time_step = time_step;
  old_old_time_step = time_step;

  while (time.is_at_end() == false)
  {
    average_solution_value = VectorTools::compute_mean_value(dof_handler,
                                                             QGauss<dim>(fe.get_degree() + 1),
                                                             solution,
                                                             0);
    global_entropy_variation = get_entropy_variation();
    step_number = time.get_step_number();

    if (prm.verbose)
      std::cout << "Step number: " << step_number << ", "
                << "Current time: " << time.get_current_time() << ", "
                << "Next time step: " << time.get_next_step_size()
                << std::endl;

    explicit_runge_kutta.evolve_one_time_step(worker,
                                              time.get_current_time(),
                                              time.get_next_step_size(),
                                              aux_solution);
    constraint_matrix.distribute(aux_solution);

    old_old_solution = old_solution;
    old_solution = solution;
    solution = aux_solution;

    time.advance_time();

    if (time.get_step_number() % 100 == 0 || time.is_at_end())
      output_results(time.get_current_time(),
                     time.get_step_number(),
                     method);
  }
}



template <int dim>
void EntropyViscosity<dim>::run()
{
  make_grid();

  setup_dofs();

  setup_system();

  setup_vectors();

  assemble_system();

  const double initial_time{prm.start_time};
  const double final_time{prm.final_time};
  const double time_step{prm.initial_time_step};
  const double cfl{get_cfl_number(time_step)};

  std::cout << "CFL number (based on initial time step): " << cfl << std::endl;


  std::cout << "Explicit methods:" << std::endl;
//  explicit_method(TimeStepping::FORWARD_EULER,
//                  initial_time,
//                  final_time,
//                  time_step);
//  std::cout << "   Forward Euler:            error= "
//            << solution.l2_norm()
//            << std::endl;
//
  explicit_method(TimeStepping::RK_THIRD_ORDER,
                  initial_time,
                  final_time,
                  time_step);
  std::cout << "   Third order Runge-Kutta:  error = "
            << solution.l2_norm()
            << std::endl;

  explicit_method(TimeStepping::RK_CLASSIC_FOURTH_ORDER,
                  initial_time,
                  final_time,
                  time_step);
  std::cout << "   Fourth order Runge-Kutta: error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << std::endl;

  unsigned int n_steps;
  std::cout << "Embedded explicit methods:" << std::endl;
  n_steps = embedded_explicit_method(TimeStepping::HEUN_EULER,
                                     initial_time,
                                     final_time,
                                     time_step);
  std::cout << "   Heun-Euler:               error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << "                   steps performed = "
            << n_steps
            << std::endl;

  n_steps = embedded_explicit_method(TimeStepping::BOGACKI_SHAMPINE,
                                     initial_time,
                                     final_time,
                                     time_step);
  std::cout << "   Bogacki-Shampine:         error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << "                   steps performed = "
            << n_steps
            << std::endl;

  n_steps = embedded_explicit_method(TimeStepping::DOPRI,
                                     initial_time,
                                     final_time,
                                     time_step);
  std::cout << "   Dopri:                    error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << "                   steps performed = "
            << n_steps
            << std::endl;

  n_steps = embedded_explicit_method(TimeStepping::FEHLBERG,
                                     initial_time,
                                     final_time,
                                     time_step);
  std::cout << "   Fehlberg:                 error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << "                   steps performed = "
            << n_steps
            << std::endl;

  n_steps = embedded_explicit_method(TimeStepping::CASH_KARP,
                                     initial_time,
                                     final_time,
                                     time_step);
  std::cout << "   Cash-Karp:                error = "
            << solution.l2_norm()
            << std::endl;
  std::cout << "                   steps performed = "
            << n_steps
            << std::endl;
}

} // namespace Step52



int main(int argc, char *argv[])
{
  try
  {
    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "entropy_viscosity.prm";

    Guermond::RunTimeParameters parameter_set(parameter_filename);

    Guermond::EntropyViscosity<1> diffusion(parameter_set);
    diffusion.run();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
    std::cerr << "Exception on processing: " << std::endl
        << exc.what() << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
        << std::endl
        << "----------------------------------------------------"
        << std::endl;
    std::cerr << "Unknown exception!" << std::endl
        << "Aborting!" << std::endl
        << "----------------------------------------------------"
        << std::endl;
    return 1;
  };

  return 0;
}

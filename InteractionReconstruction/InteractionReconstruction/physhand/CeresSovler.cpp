#include"CeresSolver.h"

#include<vector>
#include<time.h>
#include<stdlib.h>

using ceres::DynamicAutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

/************************************************************************/
/*CostFunctorForce                                                      */
/************************************************************************/

CostFunctorForce::CostFunctorForce(
	const Eigen::Matrix3Xd& var_to_net_force,
	const Eigen::Vector3d& gravity_force,
	const Eigen::Vector3d& tar_force,
	double weight /* = 1.0 */) :
	m_var_to_net_force(var_to_net_force),
	m_gravity_force(gravity_force),
	m_tar_force(tar_force),
	m_weight(weight)
{ }

template<typename T>
bool CostFunctorForce::operator()(T const* const* x, T* e) const
{
	int var_num = m_var_to_net_force.cols();
	
	for (int r = 0; r < 3; r++)
	{
		e[r] = T(0);
		for (int c = 0; c < var_num; c++)
		{
			e[r] += T(m_var_to_net_force(r, c)) * x[0][c];
		}
		e[r] += T(m_gravity_force[r]) - T(m_tar_force[r]);
		e[r] *= T(m_weight);
	}
	return true;
}

/************************************************************************/
/*CostFunctorMoment                                                     */
/************************************************************************/

CostFunctorMoment::CostFunctorMoment(
	const Eigen::Matrix3Xd& var_to_net_moment, const Eigen::Vector3d& tar_moment, double weight /* = 1.0 */) :
	m_var_to_net_moment(var_to_net_moment),
	m_tar_moment(tar_moment),
	m_weight(weight)
{ }

template<typename T>
bool CostFunctorMoment::operator()(T const* const* x, T* e) const
{
	int var_num = m_var_to_net_moment.cols();

	for (int r = 0; r < 3; r++)
	{
		e[r] = T(0);
		for (int c = 0; c < var_num; c++)
		{
			e[r] += T(m_var_to_net_moment(r, c)) * x[0][c];
		}
		e[r] -= T(m_tar_moment[r]);
		e[r] *= T(m_weight);
	}
	return true;
}

/************************************************************************/
/*CostFunctorForceVal                                                   */
/************************************************************************/

CostFunctorForceVal::CostFunctorForceVal(int dim, double weight /* = 1 */) 
	: m_dim(dim), m_weight(weight)
{ }

template<typename T>
bool CostFunctorForceVal::operator()(T const* const* x, T* e) const
{
	for (int i = 0; i < m_dim; i++)
		e[i] = T(m_weight) * x[0][i];
	return true;
}

/************************************************************************/
/*CostFunctorForceAvg                                                   */
/************************************************************************/

CostFunctorForceAvg::CostFunctorForceAvg(int dim, double weight /* = 1 */)
	: m_dim(dim), m_weight(weight)
{ }

template<typename T>
bool CostFunctorForceAvg::operator()(T const* const* x, T* e) const
{
	T avg = T(0);
	for (int i = 0; i < m_dim; i++)
		avg += x[0][i];
	avg /= T(m_dim);
	for (int i = 0; i < m_dim; i++)
		e[i] = T(m_weight) * (x[0][i] - avg);
	return true;
}

/************************************************************************/
/*CostFuncorDist														*/
/************************************************************************/

CostFuncorDist::CostFuncorDist(const std::vector<double>& dist_arr, double weight /* = 1.0 */)
	: m_dist_arr(dist_arr), m_weight(weight)
{ }

template<typename T>
bool CostFuncorDist::operator()(T const* const* x, T* e) const
{
	for (int i = 0; i < m_dist_arr.size(); i++)
		e[i] = T(m_weight) * (T(m_dist_arr[i]) - x[0][i]);
	return true;
}

/************************************************************************/
/*CostFunctorNoContactNoforce											*/
/************************************************************************/

CostFunctorNoContactNoforce::CostFunctorNoContactNoforce(int dim, double weight /* = 1.0 */) 
	: m_dim(dim), m_weight(weight)
{ }

template<typename T>
bool CostFunctorNoContactNoforce::operator()(T const* const* x, T* e) const
{
	for (int d = 0; d < m_dim; d++)
	{
		T force_sum = T(0);
		for (int i = 0; i < 4; i++)
			force_sum += x[0][4 * d + i];
		e[d] = T(m_weight) * force_sum * x[1][d];
	}
	return true;
}



double solveForceWithCeres(
	const Eigen::Matrix3Xd& var_to_net_force,
	const Eigen::Matrix3Xd& var_to_net_moment,
	const Eigen::Vector3d& gravity_force,
	const std::vector<double>& dists,
	const double force_weight,
	const double moment_weight,
	const double force_val_weight,
	const double contact_dist_weight,
	const double no_contact_no_force_weight,
	Eigen::VectorXd& vars_ret,
	std::vector<double>& dist_ret,
	const Eigen::Vector3d& tar_force,
	const Eigen::Vector3d& tar_moment
)
{
	int var_num = var_to_net_force.cols();
	int dist_num = dists.size();

	assert(var_num / 4 == dist_num);

	std::vector<double> ceres_var_force(var_num, 0);
	std::vector<double> ceres_var_dist(dists);

	srand(time(0));
	for (int i = 0; i < var_num; i++)
	{
		double num = ((double)(rand()) / RAND_MAX - 0.5) * 2 * gravity_force.norm();
		ceres_var_force[i] = num;
	}

	DynamicAutoDiffCostFunction<CostFunctorForce, 4>* cost_func_force =
		new DynamicAutoDiffCostFunction<CostFunctorForce, 4>(
			new CostFunctorForce(var_to_net_force, gravity_force, tar_force, force_weight));
	cost_func_force->AddParameterBlock(var_num);
	cost_func_force->SetNumResiduals(3);

	DynamicAutoDiffCostFunction<CostFunctorMoment, 4>* cost_func_moment =
		new DynamicAutoDiffCostFunction<CostFunctorMoment, 4>(
			new CostFunctorMoment(var_to_net_moment, tar_moment, moment_weight));
	cost_func_moment->AddParameterBlock(var_num);
	cost_func_moment->SetNumResiduals(3);

	DynamicAutoDiffCostFunction<CostFunctorForceVal, 4>* cost_func_val =
		new DynamicAutoDiffCostFunction<CostFunctorForceVal, 4>(
			new CostFunctorForceVal(var_num, force_val_weight));
	cost_func_val->AddParameterBlock(var_num);
	cost_func_val->SetNumResiduals(var_num);

	DynamicAutoDiffCostFunction<CostFuncorDist, 4>* cost_func_dist =
		new DynamicAutoDiffCostFunction<CostFuncorDist, 4>(
			new CostFuncorDist(dists, contact_dist_weight));
	cost_func_dist->AddParameterBlock(dist_num);
	cost_func_dist->SetNumResiduals(dist_num);

	DynamicAutoDiffCostFunction<CostFunctorNoContactNoforce, 4>* cost_func_ncnf =
		new DynamicAutoDiffCostFunction<CostFunctorNoContactNoforce, 4>(
			new CostFunctorNoContactNoforce(dist_num, no_contact_no_force_weight));
	cost_func_ncnf->AddParameterBlock(var_num);
	cost_func_ncnf->AddParameterBlock(dist_num);
	cost_func_ncnf->SetNumResiduals(dist_num);

	Problem problem;
	problem.AddResidualBlock(cost_func_force, nullptr, &ceres_var_force[0]);
	problem.AddResidualBlock(cost_func_moment, nullptr, &ceres_var_force[0]);
	problem.AddResidualBlock(cost_func_val, nullptr, &ceres_var_force[0]);
	// problem.AddResidualBlock(cost_func_avg, nullptr, &ceres_var_force[0]);
	problem.AddResidualBlock(cost_func_dist, nullptr, &ceres_var_dist[0]);
	problem.AddResidualBlock(cost_func_ncnf, nullptr, &ceres_var_force[0], &ceres_var_dist[0]);

	for (int i = 0; i < var_num; i++)
	{
		problem.SetParameterLowerBound(&ceres_var_force[0], i, 0);
	}
	for (int i = 0; i < dist_num; i++)
	{
		problem.SetParameterLowerBound(&ceres_var_dist[0], i, 0);
	}

	// Run the solver!
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = false;
	options.max_num_iterations = 100;
	Solver::Summary summary;
	Solve(options, &problem, &summary);

	// std::cout << summary.BriefReport() << std::endl;

	for (int i = 0; i < var_num; i++)
		vars_ret(i) = ceres_var_force[i];
	for (int i = 0; i < dist_num; i++)
		dist_ret[i] = ceres_var_dist[i];

	return summary.final_cost;
}
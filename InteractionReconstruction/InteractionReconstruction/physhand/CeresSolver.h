#pragma once
#include<ceres/ceres.h>
#include<Eigen/dense>

class CostFunctorForce {
private:
	Eigen::Matrix3Xd m_var_to_net_force;
	Eigen::Vector3d m_gravity_force;
	Eigen::Vector3d m_tar_force;
	double m_weight;

public:
	CostFunctorForce(
		const Eigen::Matrix3Xd& var_to_net_force,
		const Eigen::Vector3d& gravity_force,
		const Eigen::Vector3d& tar_force,
		const double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

class CostFunctorMoment {
private:
	Eigen::Matrix3Xd m_var_to_net_moment;
	Eigen::Vector3d m_tar_moment;
	double m_weight;

public:
	CostFunctorMoment(const Eigen::Matrix3Xd& var_to_net_moment, const Eigen::Vector3d& tar_moment, const double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

class CostFunctorForceVal {
private:
	int m_dim;
	double m_weight;
public:
	CostFunctorForceVal(int dim, const double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

class CostFunctorForceAvg {
private:
	int m_dim;
	double m_weight;
public:
	CostFunctorForceAvg(int dim, const double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

class CostFuncorDist {
private:
	std::vector<double> m_dist_arr;
	double m_weight;
public:
	CostFuncorDist(const std::vector<double>& dist_arr, double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

class CostFunctorNoContactNoforce {
private:
	double m_weight;
	int m_dim;
public:
	CostFunctorNoContactNoforce(int dim, double weight = 1.0);

	template<typename T>
	bool operator()(T const* const* x, T* e) const;
};

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
);
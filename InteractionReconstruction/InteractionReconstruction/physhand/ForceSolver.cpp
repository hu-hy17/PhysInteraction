#include<iostream>
#include<fstream>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<thread>

#include"ForceSolver.h"
#include"Utils.h"
#include"CeresSolver.h"

using std::cerr;
using std::cout;
using std::endl;

ForceSolver::ForceSolver()
{
	m_contacts_pos = std::vector<Eigen::Vector3d>();
	m_contacts_norm = std::vector<Eigen::Vector3d>();
	m_contact_forces = std::vector <Eigen::Vector3d>();
	m_contact_frictions = std::vector<Eigen::Vector3d>();
	srand(time(0));
}

double ForceSolver::_calTotalEnergy_(const Eigen::VectorXd& vars, const std::vector<double>& dists, std::vector<double>& energy_arr)
{
	assert(energy_arr.size() >= ENERGY_NUM);
	assert(vars.size() % 4 == 0 && vars.size() / 4 == m_contacts_pos.size());
	assert(vars.size() / 4 == dists.size());

	energy_arr[1] = (m_var_to_net_force * vars + m_mass * m_gravity - m_target_force).norm();
	energy_arr[1] = energy_arr[1] * energy_arr[1];

	energy_arr[2] = (m_var_to_net_moment * vars - m_target_moment).norm();
	energy_arr[2] = energy_arr[2] * energy_arr[2];

	energy_arr[3] = 0;
	energy_arr[4] = 0;

	int contact_num = m_contacts_pos.size();
	for (int i = 0; i < contact_num; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (vars[4 * i + j] < 0)
				energy_arr[3] += vars[4 * i + j] * vars[4 * i + j];
			energy_arr[4] += vars[4 * i + j] * vars[4 * i + j];
		}
	}

	energy_arr[5] = 0;
	energy_arr[6] = 0;
	for (int i = 0; i < dists.size(); i++)
	{
		energy_arr[5] += (dists[i] - m_contacts_dist[i]) * (dists[i] - m_contacts_dist[i]);
		energy_arr[6] += abs(dists[i]) * vars(3 * i);
	}

	energy_arr[0] = m_force_weight * energy_arr[1] + m_moment_weight * energy_arr[2] +
		m_friction_cone_weight * energy_arr[3] + m_force_value_weight * energy_arr[4] +
		m_contact_dist_weight * energy_arr[5] + m_no_contact_no_force_weight * energy_arr[6];

	return energy_arr[0];
}

Eigen::VectorXd ForceSolver::_calGradient_(const Eigen::VectorXd& vars)
{
	int contact_num = m_contacts_pos.size();
	int var_num = vars.size();
	assert(var_num == 3 * contact_num);

	auto grad_force = 2 * m_var_to_net_force.transpose() * (m_var_to_net_force * vars + m_mass * m_gravity - m_target_force);
	auto grad_moment = 2 * m_var_to_net_moment.transpose() * (m_var_to_net_moment * vars - m_target_moment);
	
	Eigen::VectorXd grad_friction_cone = Eigen::VectorXd::Zero(var_num);
	Eigen::VectorXd grad_force_value = Eigen::VectorXd::Zero(var_num);
	Eigen::VectorXd grad_force_dir = Eigen::VectorXd::Zero(var_num);

	const double sqrt2 = 1 / sqrt(2);

	for (int i = 0; i < contact_num; i++)
	{
		bool wrong_dir = vars(3 * i) >= 0 ? false : true;

		if (abs(vars(3 * i + 1)) > sqrt2 * m_friction_factor * vars(3 * i))
		{
			grad_friction_cone(3 * i + 1) = vars(3 * i + 1) >= 0 ? 1 : -1;
			grad_friction_cone(3 * i) -= sqrt2 * m_friction_factor;
		}

		if (abs(vars(3 * i + 2)) > sqrt2 * m_friction_factor * vars(3 * i))
		{
			grad_friction_cone(3 * i + 2) = vars(3 * i + 2) >= 0 ? 1 : -1;
			grad_friction_cone(3 * i) -= sqrt2 * m_friction_factor;
		}

		for (int d = 0; d < 3; d++)
		{
			grad_force_value(3 * i + d) = 2 * vars(3 * i + d);
		}

		if (wrong_dir)
		{
			grad_force_dir(3 * i) = -1;
		}
	}

	return m_force_weight * grad_force + m_moment_weight * grad_moment
		+ m_friction_cone_weight * grad_friction_cone + m_force_value_weight * grad_force_value;
}

void ForceSolver::_generateForceMat_()
{
	int contact_num = m_contacts_pos.size();
	Eigen::Matrix3Xd ret(3, contact_num * 4);

	for (int i = 0; i < contact_num; i++)
	{	
		Eigen::Vector3d fdir1, fdir2;
		getOrthognalBase(m_contacts_norm[i], fdir1, fdir2);
		Eigen::Vector3d base1 = (m_contacts_norm[i] + m_friction_factor * fdir1).normalized();
		Eigen::Vector3d base2 = (m_contacts_norm[i] - m_friction_factor * fdir1).normalized();
		Eigen::Vector3d base3 = (m_contacts_norm[i] + m_friction_factor * fdir2).normalized();
		Eigen::Vector3d base4 = (m_contacts_norm[i] - m_friction_factor * fdir2).normalized();

		for (int j = 0; j < 3; j++)
		{
			ret(j, 4 * i) = base1(j);
			ret(j, 4 * i + 1) = base2(j);
			ret(j, 4 * i + 2) = base3(j);
			ret(j, 4 * i + 3) = base4(j);
		}
	}

	m_var_to_net_force = ret;
}

void ForceSolver::_generateMomentMat_()
{
	int contact_num = m_contacts_pos.size();
	Eigen::Matrix3Xd ret(3, contact_num * 4);

	for (int i = 0; i < contact_num; i++)
	{
		Eigen::Vector3d r = m_contacts_pos[i] - m_mass_center;	// Á¦±Û
		for (int j = 0; j < 4; j++)
		{
			Eigen::Vector3d force_dir(m_var_to_net_force.block(0, 4 * i + j, 3, 1));
			Eigen::Vector3d moment = r.cross(force_dir);
			for (int r = 0; r < 3; r++)
			{
				ret(r, 4 * i + j) = moment(r);
			}
		}
	}

	m_var_to_net_moment = ret;
}

void ForceSolver::_solveForce_(const Eigen::Vector3d& target_force,
	const Eigen::Vector3d& target_moment)
{
	if (!m_set_phys_data || !m_set_contact_data || !m_set_weight_data)
	{
		cerr << m_class_name << "Error: you have not set enough data for ForceSolver!" << endl;
		system("pause");
		exit(-1);
	}

	int var_num = 4 * m_contacts_pos.size();
	if (var_num == 0)
	{
		m_force_energy = std::pow((target_force - m_mass * m_gravity).norm(), 2);
		m_moment_energy = std::pow(target_moment.norm(), 2);
		m_force_value_energy = m_friction_cone_energy = 0;
		m_energy = m_force_weight * m_force_energy + m_moment_weight * m_moment_energy;
		return;
	}

	_generateForceMat_();
	_generateMomentMat_();

	// cout << var_to_net_force << endl;
	// cout << var_to_net_moment << endl;

	double g_force = 2 * m_mass * 9.8;
	Eigen::VectorXd vars(var_num);
	std::vector<double> var_dist(m_contacts_dist);
	std::vector<double> min_energy(ENERGY_NUM, 0);

	if (m_solver == M_GD_)
	{
		for (int i = 0; i < m_contacts_pos.size(); i++)
		{
			vars[3 * i] = ((double)(rand()) / RAND_MAX) * g_force;
			vars[3 * i + 1] = ((double)(rand()) / RAND_MAX - 0.5) * (2 * m_friction_factor / sqrt(2)) * vars[3 * i];
			vars[3 * i + 2] = ((double)(rand()) / RAND_MAX - 0.5) * (2 * m_friction_factor / sqrt(2)) * vars[3 * i];
		}

		std::vector<double> energy_record(6, 0);

		double start_step = 0.01;
		double step = start_step;
		double eps = 1e-4;
		double old_val = _calTotalEnergy_(vars, min_energy, m_contacts_dist);

		while (1)
		{
			// printf("%f\n", old_val);
			auto grad = _calGradient_(vars);
			// cout << grad.transpose() << endl;
			auto new_vars = vars - step * grad;
			double new_val = _calTotalEnergy_(new_vars, energy_record, m_contacts_dist);
			if (new_val < old_val)
			{
				vars = new_vars;
				old_val = new_val;
				min_energy.assign(energy_record.begin(), energy_record.end());
				step = start_step;
			}
			else
			{
				if (abs(new_val - old_val) < eps)
					break;
				step = step / 2;
			}
		}
	}
	else if (m_solver == M_SA_)
	{
		int sim_times = 10;
		std::vector<std::vector<double>> thread_energy(sim_times);
		std::vector<Eigen::VectorXd> thread_vars(sim_times);
		std::vector<std::thread> threads(sim_times);

		for (int i = 0; i < sim_times; i++)
		{
			thread_energy[i].assign(min_energy.begin(), min_energy.end());
			thread_vars[i] = vars;
			// _multiThreadSA_(std::ref(thread_energy[i]), std::ref(thread_vars[i]));
			threads[i] = std::thread(&ForceSolver::_multiThreadSA_, this, std::ref(thread_energy[i]), std::ref(thread_vars[i]), rand());
		}

		for (int i = 0; i < sim_times; i++)
		{
			threads[i].join();
		}

		int min_idx = -1;
		double min_energy_val = 1e16;
		for (int i = 0; i < sim_times; i++)
		{
			if (thread_energy[i][0] < min_energy_val)
			{
				min_idx = i;
				min_energy_val = thread_energy[i][0];
			}
		}

		assert(min_idx >= 0);
		min_energy = thread_energy[min_idx];
		vars = thread_vars[min_idx];
	}
	else if (m_solver == M_CERES_)
	{
		_solveForceWithCeres_(vars, var_dist, min_energy, target_force, target_moment);
	}

	m_energy = min_energy[0];
	m_force_energy = min_energy[1];
	m_moment_energy = min_energy[2];
	m_friction_cone_energy = min_energy[3];
	m_force_value_energy = min_energy[4];
	m_contact_dist_energy = min_energy[5];
	m_no_contact_no_force_energy = min_energy[6];

	m_contact_result_dists = var_dist;

	for (int i = 0; i < m_contacts_pos.size(); i++)
	{
		Eigen::Vector3d net_force = Eigen::Vector3d::Zero();
		for (int j = 0; j < 4; j++)
		{
			net_force += vars(4 * i + j) * m_var_to_net_force.block(0, 4 * i + j, 3, 1);
		}
		Eigen::Vector3d pressure_force = (net_force.dot(m_contacts_norm[i])) * m_contacts_norm[i];
		Eigen::Vector3d friction_force = net_force - pressure_force;
		m_contact_forces.push_back(pressure_force);
		m_contact_frictions.push_back(friction_force);
	}
}

void  ForceSolver::_multiThreadSA_(std::vector<double>& energy_result,
	Eigen::VectorXd& vars,
	unsigned int seed)
{
	double max_temp = 1e5;
	double temp = max_temp;
	double min_temp = 1e-5;
	double alpha = 0.98;
	double r = 1000;

	int var_num = vars.size();
	double g_force = 2 * m_mass * 9.8;
	std::vector<double> tmp_energy(ENERGY_NUM, 0);

	srand(seed);
	for (int i = 0; i < var_num; i++)
	{
		double num = ((double)(rand()) / RAND_MAX - 0.5) * g_force;
		vars[i] = num;
	}

	double old_val = _calTotalEnergy_(vars, m_contacts_dist, energy_result);
	temp = max_temp;

	while (temp > min_temp)
	{
		Eigen::VectorXd delta_var(var_num);
		for (int i = 0; i < var_num; i++)
		{
			delta_var(i) = ((double)(rand()) / RAND_MAX - 0.5) * (temp / max_temp) * g_force;
		}
		Eigen::VectorXd new_vars = vars + delta_var;
		double new_val = _calTotalEnergy_(new_vars, m_contacts_dist, tmp_energy);
		if (new_val < old_val)
		{
			old_val = new_val;
			vars = new_vars;
			temp *= alpha;
			energy_result = tmp_energy;
		}
		else
		{
			double ra = ((double)rand()) / RAND_MAX;
			double re = exp(-r * (new_val - old_val) / temp);
			if (ra < re)
			{
				old_val = new_val;
				vars = new_vars;
				energy_result = tmp_energy;
			}
		}
	}
}

void ForceSolver::_solveForceWithCeres_(Eigen::VectorXd& vars, std::vector<double>& var_dist, std::vector<double>& energy_arr,
	const Eigen::Vector3d& target_force, const Eigen::Vector3d& target_moment)
{
	assert(energy_arr.size() >= ENERGY_NUM);
	assert(vars.size() % 4 == 0 && vars.size() / 4 == m_contacts_pos.size());

	double tot_energy = solveForceWithCeres(m_var_to_net_force, m_var_to_net_moment, m_mass * m_gravity, m_contacts_dist,
		sqrt(m_force_weight), sqrt(m_moment_weight), sqrt(m_force_value_weight),
		sqrt(m_contact_dist_weight), sqrt(m_no_contact_no_force_weight),
		vars, var_dist, target_force, target_moment);

	energy_arr[1] = std::pow((m_var_to_net_force * vars + m_mass * m_gravity).norm(), 2);
	energy_arr[2] = std::pow((m_var_to_net_moment * vars).norm(), 2);
	energy_arr[3] = 0;
	energy_arr[4] = 0;
	for (int i = 0; i < vars.size(); i++)
	{
		if (vars[i] < 0)
			energy_arr[3] += std::pow(vars[i], 2);
		energy_arr[4] += std::pow(vars[i], 2);
	}

	energy_arr[5] = 0;
	energy_arr[6] = 0;
	for (int i = 0; i < var_dist.size(); i++)
	{
		energy_arr[5] += std::pow(var_dist[i] - m_contacts_dist[i], 2);
		double force_sum = 0;
		for (int j = 0; j < 4; j++)
			force_sum += vars(4 * i + j);
		energy_arr[6] += std::pow(force_sum * var_dist[i], 2);
	}

	energy_arr[0] = m_force_weight * energy_arr[1] +
		m_moment_weight * energy_arr[2] + m_friction_cone_weight * energy_arr[3] +
		m_force_value_weight * energy_arr[4] + m_contact_dist_weight * energy_arr[5]
		+ m_no_contact_no_force_weight * energy_arr[6];

	// output force solution
	// cout << "Force solution: " << vars.transpose() << endl;

	//cout << "Energy by Ceres : " << tot_energy * 2 << endl;
	//cout << "Energy cal by me : ";
	//for (auto en : energy_arr)
	//	cout << ' ' << en;
	//cout << endl;

	std::ofstream ofs("../../../../result/energy_debug.txt", std::ios::app);
	ofs << tot_energy * 2 << endl;
	ofs.close();
}

void ForceSolver::reset(bool clear_phys_data, bool clear_weight_data)
{
	m_set_tar_force_and_moment = false;
	m_set_pos = false;
	m_set_contact_data = false;
	
	clearContactData();

	if (clear_phys_data)
		m_set_phys_data = false;
	if (clear_weight_data)
		m_set_weight_data = false;
}

void ForceSolver::clearContactData()
{
	m_set_contact_data = false;

	m_contacts_pos.clear();
	m_contacts_norm.clear();
	m_contacts_dist.clear();
	m_contact_forces.clear();
	m_contact_frictions.clear();
	m_contact_result_dists.clear();

	m_contacts_pos.swap(std::vector<Eigen::Vector3d>());
	m_contacts_norm.swap(std::vector<Eigen::Vector3d>());
	m_contacts_dist.swap(std::vector<double>());
	m_contact_forces.swap(std::vector<Eigen::Vector3d>());
	m_contact_frictions.swap(std::vector<Eigen::Vector3d>());
	m_contact_result_dists.swap(std::vector<double>());
}

void ForceSolver::setCurPos(const Eigen::Vector3d& cur_pos)
{
	m_mass_center = cur_pos;
	m_set_pos = true;
}

void ForceSolver::setTarForceAndTorque(const Eigen::Vector3d& tar_force, const Eigen::Vector3d& tar_moment)
{
	m_target_force = tar_force;
	m_target_moment = tar_moment;
	m_set_tar_force_and_moment = true;
}

void ForceSolver::setPhysData(const double friction_factor,       const double mass,
							  const Eigen::Vector3d& gravity,	  const double delta_t)
{
	if (m_set_phys_data)
	{
		cout << m_class_name << "Warning: you have not cleared the current physical data, it will be overwritten." << endl;
	}

	m_friction_factor = friction_factor;
	m_mass = mass;
	m_gravity = gravity;
	m_delta_t = delta_t;

	m_set_phys_data = true;
}

void ForceSolver::setContactData(const std::vector<Eigen::Vector3d>& contacts_pos,
								 const std::vector<Eigen::Vector3d>& contacts_norm,
								 const std::vector<double>& contacts_dist)
{
	if (m_set_contact_data)
	{
		cerr << m_class_name << "Error: you have not cleared the current contact points data!" << endl;
		system("pause");
		exit(-1);
	}

	assert(contacts_pos.size() == contacts_norm.size());
	assert(contacts_norm.size() == contacts_dist.size());

	m_contacts_pos.insert(m_contacts_pos.end(), contacts_pos.begin(), contacts_pos.end());
	m_contacts_norm.insert(m_contacts_norm.end(), contacts_norm.begin(), contacts_norm.end());
	m_contacts_dist.insert(m_contacts_dist.end(), contacts_dist.begin(), contacts_dist.end());

	m_set_contact_data = true;
}

void ForceSolver::setWeightData(const double force_weight,
				                const double moment_weight,
				                const double friction_cone_weight,
				                const double force_value_weight,
								const double contact_dist_weight,
								const double no_contact_no_force_weight)
{
	if (m_set_weight_data)
	{
		cout << m_class_name << "Warning: you have not cleared the current weight data, it will be overwritten." << endl;
	}

	m_force_weight = force_weight;
	m_moment_weight = moment_weight;
	m_friction_cone_weight = friction_cone_weight;
	m_force_value_weight = force_value_weight;
	m_contact_dist_weight = contact_dist_weight;
	m_no_contact_no_force_weight = no_contact_no_force_weight;
	
	m_set_weight_data = true;
}

void ForceSolver::solve(bool static_balance /*  = true */)
{
	Eigen::Vector3d target_force(0, 0, 0);
	Eigen::Vector3d target_moment(0, 0, 0);
	
	if (!m_set_tar_force_and_moment || static_balance)
	{
		_solveForce_(target_force, target_moment);
	}
	else
	{
		_solveForce_(m_target_force, m_target_moment);
	}
}

Eigen::Vector3d ForceSolver::getTargetForce(bool with_gravity /* = false */)
{
	return with_gravity ? m_target_force - m_mass * m_gravity : m_target_force;
}

std::vector<Eigen::Vector3f> ForceSolver::getContactPoints()
{
	std::vector<Eigen::Vector3f> ret;
	for (auto point : m_contacts_pos) {
		ret.push_back(1e3 * Eigen::Vector3f(point.x(), point.y(), -point.z()));
	}
	return ret;
}

std::vector<Eigen::Vector3f> ForceSolver::getContactForces()
{
	std::vector<Eigen::Vector3f> ret;
	for (int i = 0; i < m_contact_forces.size(); i++) {
		auto f = m_contact_forces[i] + m_contact_frictions[i];
		ret.push_back(Eigen::Vector3f(f.x(), f.y(), -f.z()));
	}
	return ret;
}

Eigen::Vector3f ForceSolver::get_tar_force()
{
	return Eigen::Vector3f(m_target_force.x(), m_target_force.y(), -m_target_force.z());
}

Eigen::Vector3f ForceSolver::get_tar_moment()
{
	return Eigen::Vector3f(m_target_moment.x(), m_target_moment.y(), -m_target_moment.z());
}

Eigen::Vector3f ForceSolver::get_cur_pos()
{
	return 1e3 * Eigen::Vector3f(m_mass_center.x(), m_mass_center.y(), -m_mass_center.z());
}
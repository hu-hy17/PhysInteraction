#pragma once
#include<vector>
#include<string>

#include<Eigen/Dense>

class ForceSolver
{
private:
	std::string m_class_name = "(ForceSolver)";

	Eigen::Vector3d m_cur_pos;
    Eigen::Vector3d m_cur_vel;		

	Eigen::Vector3d m_cur_ang;
	Eigen::Vector3d m_cur_ang_vel;

	bool m_set_tar_force_and_moment = false;

	bool m_set_pos = false;

	double m_friction_factor;			
	double m_mass;						
	Eigen::Vector3d m_mass_center;		
	Eigen::Vector3d m_gravity;			
	double m_delta_t;					
    
	bool m_set_phys_data = false;

	std::vector<Eigen::Vector3d> m_contacts_pos;
	std::vector<Eigen::Vector3d> m_contacts_norm;
	std::vector<double> m_contacts_dist;

	bool m_set_contact_data = false;

	double m_force_weight;				
	double m_moment_weight;				
	double m_friction_cone_weight;		
	double m_force_value_weight;		
	double m_contact_dist_weight;			
	double m_no_contact_no_force_weight;	

	bool m_set_weight_data = false;

	Eigen::Matrix3Xd m_var_to_net_force;
	Eigen::Matrix3Xd m_var_to_net_moment;

public:
	const static int M_GD_ = 0;
	const static int M_SA_ = 1;
	const static int M_CERES_ = 2;
	int m_solver = M_CERES_;

	// 力求解的结果
	std::vector<Eigen::Vector3d> m_contact_forces;		
	std::vector<Eigen::Vector3d> m_contact_frictions;	
	std::vector<double> m_contact_result_dists;			

	// 能量项信息
	const static int ENERGY_NUM = 6 + 1;				
	double m_energy;									
	double m_force_energy;
	double m_moment_energy;
	double m_friction_cone_energy;
	double m_force_value_energy;
	double m_contact_dist_energy;
	double m_no_contact_no_force_energy;

	// 目标力与力矩
	Eigen::Vector3d m_target_force;
	Eigen::Vector3d m_target_moment;
	
private:
	double _calTotalEnergy_(const Eigen::VectorXd& vars, const std::vector<double>& dists, std::vector<double>& energy_arr);

	Eigen::VectorXd _calGradient_(const Eigen::VectorXd& vars);

	void _generateForceMat_();

	void _generateMomentMat_();

	void _solveForce_(const Eigen::Vector3d& target_force,
					  const Eigen::Vector3d& target_moment);


	void _multiThreadSA_(std::vector<double>& energy_result,
						 Eigen::VectorXd& vars,
						 unsigned int seed);

	void _solveForceWithCeres_(Eigen::VectorXd& vars, std::vector<double>& var_dist, std::vector<double>& energy_arr,
		const Eigen::Vector3d& target_force, const Eigen::Vector3d& target_moment);

public:
    ForceSolver();

	void reset(bool clear_phys_data = false, bool clear_weight_data = false);

	void clearContactData();

	void setCurPos(const Eigen::Vector3d& cur_pos);

	void setTarForceAndTorque(const Eigen::Vector3d& tar_force, const Eigen::Vector3d& tar_moment);

	void setPhysData(const double friction_factor,       const double mass,
					 const Eigen::Vector3d& gravity,     const double delta_t);

	void setContactData(const std::vector<Eigen::Vector3d>& contacts_pos,
						const std::vector<Eigen::Vector3d>& contacts_norm,
						const std::vector<double>& dists);

	void setWeightData(const double force_weight,
					   const double moment_weight,
					   const double friction_cone_weight,
					   const double force_value_weight,
					   const double contact_dist_weight,
					   const double no_contact_no_force_weight);

	void solve(bool static_balance = true);

	Eigen::Vector3d getTargetForce(bool with_gravity = false);

	std::vector<Eigen::Vector3f> getContactPoints();

	std::vector<Eigen::Vector3f> getContactForces();

	Eigen::Vector3f get_tar_force();

	Eigen::Vector3f get_tar_moment();

	Eigen::Vector3f get_cur_pos();
};
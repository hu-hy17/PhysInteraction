#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Eigen>
#include <fstream>
#include "vector_operations.hpp"
#include "manomodel.h"
#include <windows.h>

//Eigen::Affine3d TransformationSolver_CoarseAlign(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out);
//
//mat34 TransformationSolver_FineAlign(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &data_points_Cube_face1, const std::vector<float3> &corspd_points_Cube_face1, const std::vector<float3> &data_points_Cube_face2, const std::vector<float3> &corspd_points_Cube_face2, float &mean_dist);
//
//mat34 TransformationSolver_FineAlign_CornerConstraint(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &data_points_Corner, const std::vector<float3> &corspd_points_Corner, float &mean_dist, float &mean_corner_dist);
//
//mat34 TransformationSolver_FineAlign_Point2Plane(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &corspd_norms_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &corspd_norms_Cube, float &mean_dist);
//
//mat34 TransformationSolver_FineAlign_Point2Plane_CornerConstraint(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &corspd_norms_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &corspd_norms_Cube, const std::vector<float3> &data_points_Corner, const std::vector<float3> &corspd_points_Corner, const std::vector<float3> &corspd_norms_Corner, float &mean_dist, float &mean_corner_dist);
//
//float CalculateEnergy(const std::vector<float3> &data_points, const std::vector<float3> &corspd_points);
//
//float3 SolveSpherePosition(float3 sphere_pos, const std::vector<float3> data_points, const float R_ball, float &mean_dist);

class GaussNewtonSolver
{
public:
	double DERIV_STEP = 1e-4;
	double MSE_THRESHOLD = 1e-8;
	int max_itr = 1;
	int regular_num = POSE_PCA_NUM;

	/*std::ofstream JTJ_file;

	GaussNewtonSolver() { JTJ_file.open("../data/JTJ.txt"); }
	~GaussNewtonSolver() { JTJ_file.close(); }*/

	//Eigen::Matrix<float, 118, 61> Jacobian;

	Eigen::VectorXd calculate_r(const Eigen::MatrixX3d &posed_joint,const Eigen::VectorXd &para);
	Eigen::VectorXd calculate_derive(MANOModel &hand_model, const Eigen::VectorXd &para, int col_id);
	Eigen::VectorXd Solve(MANOModel &hand_model, Eigen::VectorXd &para, const Eigen::VectorXd &given_joints_xyz);

};
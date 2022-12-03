#pragma once
#include "Mesh.h"
#include "Solver.h"
#include "manomodel.h"

#include <memory>


class ModelConvertor
{
private:
	mesh m_handmesh;
	std::unique_ptr<MANOModel> m_mano;

	Eigen::VectorXd m_pose_pca;
	Eigen::VectorXd m_shape;
	Eigen::Vector3d m_global_R;
	Eigen::Vector3d m_global_T;
	Eigen::MatrixX3f m_model_f;

	GaussNewtonSolver m_GN_solver;
	Eigen::VectorXd m_para;

	int m_frame_id;
public:
	void init();
	void convert(std::vector<float> keypoints, int frame_id, int max_iter = 10);
	mesh* getMesh();
};
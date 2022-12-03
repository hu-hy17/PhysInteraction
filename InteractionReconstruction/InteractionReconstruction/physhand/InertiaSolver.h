#pragma once

#include<vector>
#include<string>

#include<Eigen/Dense>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<igl/AABB.h>

class InertiaSolver
{
private:
	std::string m_class_name = "(InertiaSolver)";

	pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
	bool m_set_data = false;

public:
	Eigen::Vector3f m_mass_center_by_obb_box;
	Eigen::Vector3d m_mass_center_by_obb_box_d;

	Eigen::Matrix3f m_inertia_tensor;

	Eigen::Matrix3f m_obj_rotation;
	Eigen::Matrix3d m_obj_rotation_d;

	Eigen::Matrix4f m_obj_to_cano;
	Eigen::Matrix4d m_obj_to_cano_d;

	Eigen::MatrixXf m_vertices;
	Eigen::MatrixXf m_normals;
	Eigen::MatrixXi m_indices;
	std::unique_ptr<igl::AABB<Eigen::MatrixXf, 3>> m_obj_tree;

	int m_length = 58;

public:
	InertiaSolver();

	void loadPointCloud(const std::vector<Eigen::Vector3f>& points,
						const std::vector<Eigen::Vector3f>& normals);

	void reset();

	void solve();

	Eigen::Vector3f getMassCenter() const;

	Eigen::Matrix3f getObjRotation() const;

	Eigen::Vector3f getTriangleNormal(int idx) const;

private:
	void _buildAABBTree_();
};
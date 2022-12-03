#include"InertiaSolver.h"

#include<iostream>
#include<fstream>

#include <pcl/features/moment_of_inertia_estimation.h>

using std::cerr;
using std::cout;
using std::endl;

InertiaSolver::InertiaSolver()
{
	m_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
}

void InertiaSolver::loadPointCloud(const std::vector<Eigen::Vector3f>& points,
								   const std::vector<Eigen::Vector3f>& normals)
{
	if (m_set_data)
	{
		m_cloud->clear();
		cout << m_class_name << "Warning : You have not clear the current point cloud data and it will be overwritten!" << endl;
	}

	if (points.size() != normals.size())
	{
		cout << m_class_name << "Error : Points number does not match normals number!" << endl;
		goto fail;
	}

	if (points.size() % 3 != 0)
	{
		cout << m_class_name << "Error : Points number must be exact division of 3!" << endl;
		goto fail;
	}

	int vertices_num = points.size();
	m_vertices.resize(vertices_num, 3);
	m_indices.resize(vertices_num / 3, 3);
	m_normals.resize(vertices_num, 3);

	for (int i = 0; i < vertices_num; i++)
	{
		m_cloud->push_back(pcl::PointXYZ(points[i].x(), points[i].y(), points[i].z()));

		m_vertices(i, 0) = points[i].x();
		m_vertices(i, 1) = points[i].y();
		m_vertices(i, 2) = points[i].z();

		m_normals(i, 0) = normals[i].x();
		m_normals(i, 1) = normals[i].y();
		m_normals(i, 2) = normals[i].z();
	}

	for (int i = 0; i < vertices_num / 3; i++)
	{
		m_indices(i, 0) = 3 * i;
		m_indices(i, 1) = 3 * i + 1;
		m_indices(i, 2) = 3 * i + 2;
	}

	_buildAABBTree_();
	m_set_data = true;

	return;

fail:
	system("pause");
	exit(-1);
}

void InertiaSolver::reset()
{
	m_cloud->clear();
	m_obj_tree.reset();
	m_set_data = false;
}

void InertiaSolver::solve()
{
	if (!m_set_data)
	{
		cerr << m_class_name << "Error: You have not set data for mass center solver!" << endl;
		system("pause");
		exit(-1);
	}

	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(m_cloud);
	feature_extractor.compute();

	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Vector3f mass_center;
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, m_obj_rotation);
	feature_extractor.getMassCenter(mass_center);

	Eigen::Vector3f mass_center_by_obb;
	mass_center_by_obb.x() = position_OBB.x;
	mass_center_by_obb.y() = position_OBB.y;
	mass_center_by_obb.z() = position_OBB.z;

	{
		cout << m_class_name << "Output: mass center by getMassCenter " << mass_center.transpose() << endl;
		cout << m_class_name << "Output: mass center by getOBB " << mass_center_by_obb.transpose() << endl;
	}

	m_mass_center_by_obb_box = mass_center_by_obb;

	for (int i = 0; i < 3; i++)
		m_mass_center_by_obb_box_d(i) = m_mass_center_by_obb_box(i);
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
			m_obj_rotation_d(r, c) = m_obj_rotation(r, c);
	}
	
	m_obj_to_cano.setIdentity();
	m_obj_to_cano_d.setIdentity();
	m_obj_to_cano.block(0, 3, 3, 1) = m_mass_center_by_obb_box;
	m_obj_to_cano.block(0, 0, 3, 3) = m_obj_rotation;
	m_obj_to_cano_d.block(0, 3, 3, 1) = m_mass_center_by_obb_box_d;
	m_obj_to_cano_d.block(0, 0, 3, 3) = m_obj_rotation_d;

	Eigen::Matrix4f cano_to_obj = m_obj_to_cano.inverse();
	float point_mass = 1.0f / m_cloud->size();
	m_inertia_tensor.setZero();

	for (int i = 0; i < m_cloud->size(); i++) 
	{
		Eigen::Vector4f pos(m_cloud->at(i).x, m_cloud->at(i).y, m_cloud->at(i).z, 1.0f);
		pos = cano_to_obj * pos;
		m_inertia_tensor(0, 0) += point_mass * (pos.y() * pos.y() + pos.z() * pos.z());
		m_inertia_tensor(1, 1) += point_mass * (pos.x() * pos.x() + pos.z() * pos.z());
		m_inertia_tensor(2, 2) += point_mass * (pos.x() * pos.x() + pos.y() * pos.y());
		m_inertia_tensor(0, 1) -= point_mass * pos.x() * pos.y();
		m_inertia_tensor(0, 2) -= point_mass * pos.x() * pos.z();
		m_inertia_tensor(1, 2) -= point_mass * pos.y() * pos.z();
	}

	m_inertia_tensor(1, 0) = m_inertia_tensor(0, 1);
	m_inertia_tensor(2, 0) = m_inertia_tensor(0, 2);
	m_inertia_tensor(2, 1) = m_inertia_tensor(1, 2);

	m_inertia_tensor *= 1e-6;
}


Eigen::Vector3f InertiaSolver::getMassCenter() const
{
	return m_mass_center_by_obb_box;
}

Eigen::Matrix3f InertiaSolver::getObjRotation() const
{
	return m_obj_rotation;
}

Eigen::Vector3f InertiaSolver::getTriangleNormal(int idx) const
{
	assert(idx < m_indices.rows());
	Eigen::Vector3f ret(0, 0, 0);
	for (int i = 0; i < 3; i++)
	{
		int v_idx = m_indices(idx, i);
		ret += m_normals.block(v_idx, 0, 1, 3).transpose();
	}
	return ret.normalized();
}

void InertiaSolver::_buildAABBTree_()
{
	m_obj_tree = std::make_unique<igl::AABB<Eigen::MatrixXf, 3>>();
	m_obj_tree->init(m_vertices, m_indices);
}
	
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Eigen>
#include "vector_operations.hpp"

using namespace std;

class mesh
{
public:
	void loadobj(std::string object_path);
	void load_remesh(std::string data_path);
	void load_remesh_from_mano(std::string mano_path);
	void load_vt(std::string data_path);
	vector<string> split(const string &s, const string &seperator);
	void obtain_obj_attribute_arr();
	void obtain_obj_attribute_arr_color_cam(mat34 trasfm);
	void re_mesh(Eigen::MatrixX3f &model);

public:
	std::vector<int> edge_vertex;
	std::vector<float2> vt;

	std::vector<float3> vertex;
	std::vector<float3> normal;
	std::vector<unsigned int> element;

	std::vector<float3> vertex_color;
	std::vector<float3> normal_color;
	std::vector<float2> vt_color;

	std::vector<float3> vertex_arr;
	std::vector<float3> normal_arr;
	std::vector<float2> vt_arr;

	std::vector<float3> vertex_arr_color;
	std::vector<float3> normal_arr_color;
};

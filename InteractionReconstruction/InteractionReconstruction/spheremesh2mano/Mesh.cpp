#include "Mesh.h"
#include <igl/per_vertex_normals.h>
#include <json/json.h>

vector<string> mesh::split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		//找到字符串中首个不等于分隔符的字母；
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		//找到又一个分隔符，将两个分隔符之间的字符串取出；
		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

void mesh::loadobj(std::string object_path)
{

	vertex.clear();
	normal.clear();
	vt.clear();
	element.clear();

	cout << "loading obj model" << endl;
	std::ifstream in(object_path);//Keypoint_2D_GT_path + "/left_3D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open obj file" << endl;
		exit(0);
	}

	///--- Read in the matrix
	for (std::string line; std::getline(in, line); ) {
		stringstream str(line);
		std::string elem;
		str >> elem;

		if (elem == "v")
		{
			float3 vert;
			str >> elem;
			vert.x = std::stof(elem);
			str >> elem;
			vert.y = std::stof(elem);
			str >> elem;
			vert.z = std::stof(elem);

			vertex.push_back(vert);
		}

		if (elem == "vn")
		{
			float3 vert_n;
			str >> elem;
			vert_n.x = std::stof(elem);
			str >> elem;
			vert_n.y = std::stof(elem);
			str >> elem;
			vert_n.z = std::stof(elem);

			normal.push_back(vert_n);
		}

		if (elem == "vt")
		{
			float2 vt_temp;
			str >> elem;
			vt_temp.x = std::stof(elem);
			str >> elem;
			vt_temp.y = 1 - std::stof(elem);

			vt.push_back(vt_temp);
		}

		if (elem == "f")
		{
			std::vector<string> face_idx;
			
			str >> elem;
			face_idx = split(elem,"/");
			element.push_back((unsigned int)std::stoi(face_idx[0])-1);

			str >> elem;
			face_idx = split(elem, "/");
			element.push_back((unsigned int)std::stoi(face_idx[0])-1);

			str >> elem;
			face_idx = split(elem, "/");
			element.push_back((unsigned int)std::stoi(face_idx[0])-1);
		}
	}

	in.close();

	if (vertex.size() != normal.size())
		cout << "size of vertex doesn't equal size of normal";

	if (vertex.size() != vt.size())
		cout << "size of vertex doesn't equal size of vt";

}

void mesh::load_remesh(std::string data_path)
{
	std::cout << "load re-mesh data" << std::endl;

	//load edge vertex id
	std::string edge_vertex_file_path = data_path + "/edge_vertex_id_sorted.txt";
	std::ifstream edge_vertex_in(edge_vertex_file_path);
	if (!edge_vertex_in.is_open()) {
		cout << "cannot open edge_vertex_id file" << endl;
		exit(0);
	}

	edge_vertex.clear();
	for (std::string line; std::getline(edge_vertex_in, line); ) 
	{
		stringstream str(line);
		std::string elem;
		str >> elem;

		edge_vertex.push_back(std::stoi(elem));
	}
	edge_vertex_in.close();
	std::cout << "load edge vertex successfully" << std::endl;

	//load face idx
	element.clear();
	//load mano front face index
	std::string mano_front_face_idx_path = data_path + "/MANO_front_face.txt";
	std::ifstream mano_front_face_idx_in(mano_front_face_idx_path);
	if (!mano_front_face_idx_in.is_open()) {
		cout << "cannot open mano_front_face_idx file" << endl;
		exit(0);
	}
	
	for (std::string line; std::getline(mano_front_face_idx_in, line); )
	{
		stringstream str(line);
		std::string elem;

		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
	}
	mano_front_face_idx_in.close();
	std::cout << "load front face successfully" << std::endl;

	//load mano back face index
	std::string mano_back_face_idx_path = data_path + "/MANO_back_reidx_face.txt";
	std::ifstream mano_back_face_idx_in(mano_back_face_idx_path);
	if (!mano_back_face_idx_in.is_open()) {
		cout << "cannot open mano_back_face_idx file" << endl;
		exit(0);
	}

	for (std::string line; std::getline(mano_back_face_idx_in, line); )
	{
		stringstream str(line);
		std::string elem;

		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
		str >> elem;
		element.push_back((unsigned int)std::stoi(elem));
	}
	mano_back_face_idx_in.close();
	std::cout << "load back face successfully" << std::endl;

}

void mesh::load_remesh_from_mano(std::string mano_path)
{
	std::ifstream file(mano_path);
	if (!file.is_open())
	{
		cout << "can not open file " << mano_path << endl;
		return;
	}

	Json::Value root;
	Json::Reader reader;
	bool parsingSuccessful = reader.parse(file, root);
	file.close();

	if (!parsingSuccessful)
	{
		cout << "fail to parse file " << mano_path << "in JSON format" << endl;
		return;
	}

	Json::Value faces = root["faces"];
	int face_num = faces.size(); // 1538
	for (int i = 0; i < face_num; i++) {
		Json::Value single_face = faces[i];
		for (int j = 0; j < 3; j++) {
			Json::Value value = single_face[j];
			element.push_back(value.asUInt());
		}
	}
}

void mesh::load_vt(std::string data_path)
{
	cout << "loading vt file" << endl;
	std::ifstream in(data_path);//Keypoint_2D_GT_path + "/left_3D_keypoints.txt"
	if (!in.is_open()) {
		cout << "cannot open vt file" << endl;
		exit(0);
	}

	vt.clear();
	for (std::string line; std::getline(in, line); ) {
		stringstream str(line);
		std::string elem;
		str >> elem;

		if (elem == "vt")
		{
			float2 vt_temp;
			str >> elem;
			vt_temp.x = std::stof(elem);
			str >> elem;
			vt_temp.y = 1 - std::stof(elem);

			vt.push_back(vt_temp);
		}
	}

	std::cout << "load vt successfully" << std::endl;
}

//void mesh::re_mesh(Eigen::MatrixX3f &model)
//{
//	int base_vert_num = model.rows();
//	vertex.resize(vt.size());
//	
//	for (int i = 0; i < base_vert_num; i++)
//	{
//		Eigen::Vector3f v_temp = model.row(i);
//		vertex[i] = make_float3(v_temp(0), v_temp(1), v_temp(2));
//	}
//
//	for (int i = 0; i < edge_vertex.size(); i++)
//	{
//		Eigen::Vector3f v_temp = model.row(edge_vertex[i]);
//		vertex[base_vert_num + i] = make_float3(v_temp(0), v_temp(1), v_temp(2));
//	}
//
//	// calculate normal
//	Eigen::MatrixXf V(vertex.size(), 3);
//	Eigen::MatrixXi F(element.size() / 3, 3);
//	Eigen::MatrixXf N_v(vertex.size(), 3);
//	for (int i = 0; i < vertex.size(); ++i)
//	{
//		V(i, 0) = vertex[i].x;
//		V(i, 1) = vertex[i].y;
//		V(i, 2) = vertex[i].z;
//	}
//	for (int i = 0; i < element.size() / 3; ++i)
//	{
//		F(i, 0) = element[3 * i + 0];
//		F(i, 1) = element[3 * i + 1];
//		F(i, 2) = element[3 * i + 2];
//	}
//	igl::per_vertex_normals(V, F, N_v);
//	for (int i = 0; i < vertex.size(); i++)
//	{
//		normal.push_back(make_float3(N_v(i, 0), N_v(i, 1), N_v(i, 2)));
//	}
//}

void mesh::re_mesh(Eigen::MatrixX3f &model)
{
	int base_vert_num = model.rows();
	vertex.resize(base_vert_num);

	for (int i = 0; i < base_vert_num; i++)
	{
		Eigen::Vector3f v_temp = model.row(i);
		vertex[i] = make_float3(v_temp(0), v_temp(1), v_temp(2));
	}

	// calculate normal
	Eigen::MatrixXf V(vertex.size(), 3);
	Eigen::MatrixXi F(element.size() / 3, 3);
	Eigen::MatrixXf N_v(vertex.size(), 3);
	for (int i = 0; i < vertex.size(); ++i)
	{
		V(i, 0) = vertex[i].x;
		V(i, 1) = vertex[i].y;
		V(i, 2) = vertex[i].z;
	}
	for (int i = 0; i < element.size() / 3; ++i)
	{
		F(i, 0) = element[3 * i + 0];
		F(i, 1) = element[3 * i + 1];
		F(i, 2) = element[3 * i + 2];
	}
	igl::per_vertex_normals(V, F, N_v);
	normal.resize(vertex.size());
	for (int i = 0; i < vertex.size(); i++)
	{
		normal[i] = make_float3(N_v(i, 0), N_v(i, 1), N_v(i, 2));
	}
}

void mesh::obtain_obj_attribute_arr()
{
	vertex_arr.clear();
	normal_arr.clear();
	vt_arr.clear();
	for (int i = 0; i < element.size(); i++)
	{
		vertex_arr.push_back(vertex[element[i]]);
		normal_arr.push_back(normal[element[i]]);
		vt_arr.push_back(vt[element[i]]);
	}
}

void mesh::obtain_obj_attribute_arr_color_cam(mat34 trasfm)
{
	vertex_arr_color.resize(vertex_arr.size());
	normal_arr_color.resize(normal_arr.size());

	for (int i = 0; i < vertex_arr.size(); i++)
	{
		vertex_arr_color[i] = trasfm.rot*vertex_arr[i] + trasfm.trans;
		normal_arr_color[i] = trasfm.rot*normal_arr[i];
	}

	vertex_color.resize(vertex.size());
	normal_color.resize(normal.size());

	for (int i = 0; i < vertex.size(); i++)
	{
		vertex_color[i] = trasfm.rot*vertex[i] + trasfm.trans;
		normal_color[i] = trasfm.rot*normal[i];
	}
}
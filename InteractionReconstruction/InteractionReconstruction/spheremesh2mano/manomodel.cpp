#include "manomodel.h"
#include <fstream>
#include <iostream>
#include <ppl.h>


MANOModel::MANOModel(std::string filepath= "D:/Documents/Projects/QT/mano/mano_model/init.json")
{
    Load(filepath);
    /*pose_ = Eigen::MatrixXd::Zero(n_joints, 3);
    trans_ = Eigen::Vector3d::Zero(3);
    shape_ = Eigen::VectorXd::Zero(n_shape_params);

    for(int i = 0; i < 16;i++){
        R.push_back(Eigen::MatrixXd::Identity(3,3));
    }*/
    //Update(false);
}

bool MANOModel::Load(std::string filepath)
{
	std::ifstream file(filepath);
	if (!file.is_open())
		return false;

	Json::Value root;
	Json::Reader reader;
	bool parsingSuccessful = reader.parse(file, root);
	file.close();

	if (!parsingSuccessful)
		return false;


	Json::Value pose_pca_basis_d1 = root["pose_pca_basis"];
	int pose_pca_basis_d1_size = pose_pca_basis_d1.size();//45
	Eigen::MatrixXd temp_pose_pca_basis(pose_pca_basis_d1_size, pose_pca_basis_d1_size);
	for (int i = 0; i < pose_pca_basis_d1_size; i++) {
		Json::Value pose_pca_basis_d2 = pose_pca_basis_d1[i];
		for (int j = 0; j < pose_pca_basis_d1_size; j++) {
			Json::Value value = pose_pca_basis_d2[j];
			temp_pose_pca_basis(i, j) = value.asDouble();
		}
	}
	Eigen::MatrixXd temp_pose_pca_basis_T = temp_pose_pca_basis.transpose();

	//only use 12 pose pca
	pose_pca_basis_ = temp_pose_pca_basis_T.block(0, 0, 45, POSE_PCA_NUM);
	// std::cout << "pose pca basis full:" << temp_pose_pca_basis_T << std::endl;
	// std::cout << "pose pca basis:" << pose_pca_basis_ << std::endl;


	Json::Value pose_pca_mean = root["pose_pca_mean"];
	int pose_pca_mean_size = pose_pca_mean.size();//45
	Eigen::VectorXd temp_pca_mean(pose_pca_mean_size);
	for (int i = 0; i < pose_pca_mean_size; i++) {
		Json::Value value = pose_pca_mean[i];
		temp_pca_mean(i) = value.asDouble();
	}
	pose_pca_mean_ = temp_pca_mean;



	Json::Value J_regressor_d1 = root["J_regressor"];
	int J_regressor_d1_size = J_regressor_d1.size();//16
	int J_regressor_d2_size = J_regressor_d1[0].size();//778
	Eigen::MatrixXd temp_J_regressor(J_regressor_d1_size, J_regressor_d2_size);
	for (int i = 0; i < J_regressor_d1_size; i++) {
		Json::Value J_regressor_d2 = J_regressor_d1[i];
		for (int j = 0; j < J_regressor_d2_size; j++) {
			Json::Value value = J_regressor_d2[j];
			temp_J_regressor(i, j) = value.asDouble();
		}
	}
	J_regressor_ = temp_J_regressor;


	Json::Value J_regressor_d1_full = root["J_regressor_full"];
	int J_regressor_d1_full_size = J_regressor_d1_full.size();//16
	int J_regressor_d2_full_size = J_regressor_d1_full[0].size();//778
	Eigen::MatrixXd temp_J_regressor_full(J_regressor_d1_full_size, J_regressor_d2_full_size);
	for (int i = 0; i < J_regressor_d1_full_size; i++) {
		Json::Value J_regressor_d2_full = J_regressor_d1_full[i];
		for (int j = 0; j < J_regressor_d2_full_size; j++) {
			Json::Value value = J_regressor_d2_full[j];
			temp_J_regressor_full(i, j) = value.asDouble();
		}
	}
	J_regressor_full_ = temp_J_regressor_full;


	Json::Value skinning_weights_d1 = root["skinning_weights"];
	int skinning_weights_d1_size = skinning_weights_d1.size();//778
	int skinning_weights_d2_size = skinning_weights_d1[0].size();//16
	Eigen::MatrixXd temp_skinning_weights(skinning_weights_d1_size, skinning_weights_d2_size);

	for (int i = 0; i < skinning_weights_d1_size; i++) {
		Json::Value skinning_weights_d2 = skinning_weights_d1[i];
		for (int j = 0; j < skinning_weights_d2_size; j++) {
			Json::Value value = skinning_weights_d2[j];
			temp_skinning_weights(i, j) = value.asDouble();
		}
	}
	skinning_weights_ = temp_skinning_weights;



	Json::Value mesh_pose_basis_d1 = root["mesh_pose_basis"];
	int mesh_pose_basis_d1_size = mesh_pose_basis_d1.size();  //778
	std::vector<Eigen::Matrix3Xd> temp_mesh_pose_basis(mesh_pose_basis_d1_size);

	for (int i = 0; i < mesh_pose_basis_d1_size; i++) {
		Json::Value mesh_pose_basis_d2 = mesh_pose_basis_d1[i];
		int mesh_pose_basis_d2_size = 3; //mesh_pose_basis_d2.size();//3
		int mesh_pose_basis_d3_size = mesh_pose_basis_d2[0].size();//135
		Eigen::Matrix3Xd temp_mesh_pose_basis_matrix(mesh_pose_basis_d2_size, mesh_pose_basis_d2[0].size());

		for (int j = 0; j < mesh_pose_basis_d2_size; j++) {
			Json::Value mesh_pose_basis_d3 = mesh_pose_basis_d2[j];
			for (int k = 0; k < mesh_pose_basis_d3_size; k++) {
				Json::Value value = mesh_pose_basis_d3[k];
				temp_mesh_pose_basis_matrix(j, k) = value.asDouble();
			}
		}
		temp_mesh_pose_basis[i] = temp_mesh_pose_basis_matrix;
	}
	mesh_pose_basis_ = temp_mesh_pose_basis;



	Json::Value mesh_shape_basis_d1 = root["mesh_shape_basis"];
	int mesh_shape_basis_d1_size = mesh_shape_basis_d1.size();  //778
	std::vector<Eigen::Matrix3Xd> temp_mesh_shape_basis(mesh_shape_basis_d1_size);

	for (int i = 0; i < mesh_shape_basis_d1_size; i++) {
		Json::Value mesh_shape_basis_d2 = mesh_shape_basis_d1[i];
		int mesh_shape_basis_d2_size = 3;// mesh_shape_basis_d2.size();//3
		int mesh_shape_basis_d3_size = mesh_shape_basis_d2[0].size();//10
		Eigen::Matrix3Xd temp_mesh_shape_basis_matrix(mesh_shape_basis_d2_size, mesh_shape_basis_d2[0].size());

		for (int j = 0; j < mesh_shape_basis_d2_size; j++) {
			Json::Value mesh_shape_basis_d3 = mesh_shape_basis_d2[j];
			for (int k = 0; k < mesh_shape_basis_d3_size; k++) {
				Json::Value value = mesh_shape_basis_d3[k];
				temp_mesh_shape_basis_matrix(j, k) = value.asDouble();
			}
		}
		temp_mesh_shape_basis[i] = temp_mesh_shape_basis_matrix;
	}
	mesh_shape_basis_ = temp_mesh_shape_basis;
	//std::cout << "basis number:" << mesh_pose_basis_d1_size << std::endl;


	Json::Value mesh_template_d1 = root["mesh_template"];
	int mesh_template_d1_size = mesh_template_d1.size();
	int mesh_template_d2_size = 3;
	Eigen::MatrixXd temp_mesh_template(mesh_template_d1_size, mesh_template_d2_size);

	for (int i = 0; i < mesh_template_d1_size; i++) {
		Json::Value mesh_template_d2 = mesh_template_d1[i];
		for (int j = 0; j < mesh_template_d2_size; j++) {
			Json::Value value = mesh_template_d2[j];
			temp_mesh_template(i, j) = value.asDouble();
		}
	}
	mesh_template_ = temp_mesh_template;



	Json::Value faces_d1 = root["faces"];
	int faces_d1_size = faces_d1.size();
	int faces_d2_size = 3;
	Eigen::MatrixX3i temp_faces(faces_d1_size, faces_d2_size);

	for (int i = 0; i < faces_d1_size; i++) {
		Json::Value faces_d2 = faces_d1[i];
		for (int j = 0; j < faces_d2_size; j++) {
			Json::Value value = faces_d2[j];
			temp_faces(i, j) = value.asInt();
		}
	}
	faces_ = temp_faces;



	Json::Value parents_d1 = root["parents"];
	int parents_d1_size = parents_d1.size();
	parents_.resize(parents_d1_size);
	//    parents_[0] = 0;
	for (int i = 1; i < parents_d1_size; i++) {
		Json::Value value = parents_d1[i];
		parents_[i] = value.asInt();
	}

	return true;
}

void MANOModel::init_rest_model(Eigen::VectorXd shape)
{
	v_shaped = mesh_template_ + Dot(mesh_shape_basis_, shape);

	J = J_regressor_ * v_shaped;
	J_full_rest = J_regressor_full_ * v_shaped;

	full_joints_weight = Eigen::MatrixXd::Zero(21, 16);
	full_joints_weight(0, 0) = 1;
	full_joints_weight(1, 0) = full_joints_weight(2, 1) = full_joints_weight(3, 2) = full_joints_weight(16, 3) = 1;
	full_joints_weight(4, 0) = full_joints_weight(5, 4) = full_joints_weight(6, 5) = full_joints_weight(17, 6) = 1;
	full_joints_weight(7, 0) = full_joints_weight(8, 7) = full_joints_weight(9, 8) = full_joints_weight(18, 9) = 1;
	full_joints_weight(10, 0) = full_joints_weight(11, 10) = full_joints_weight(12, 11) = full_joints_weight(19, 12) = 1;
	full_joints_weight(13, 0) = full_joints_weight(14, 13) = full_joints_weight(15, 14) = full_joints_weight(20, 15) = 1;
}


Eigen::MatrixX3d MANOModel::get_posed_model( Eigen::VectorXd pose_pca, Eigen::Vector3d pose_global_R, Eigen::Vector3d pose_global_T)
{

    Eigen::VectorXd joints = (pose_pca_basis_* pose_pca).col(0) + pose_pca_mean_;//(pose_pca_basis_.transpose() * pose_pca)

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> joints_matrix(joints);
    joints_matrix.resize(joints.rows()/3, 3);

    Eigen::MatrixX3d pose(n_joints, 3);
    pose << pose_global_R.transpose(),
            joints_matrix;
    
	std::vector<Eigen::Vector3d> pose_rotation = Split(pose);
	std::vector<Eigen::Matrix3d> R = Rodrigues(pose_rotation);

	// Add pose bias to T-pose mesh
	Eigen::VectorXd pose_bias(15 * 9);
	for (int i = 0; i < 15; ++i)
	{
		auto& rot = R[i + 1];
		for (int r = 0; r < 3; ++r)
		{
			for (int c = 0; c < 3; ++c)
			{
				if (r == c)
					pose_bias[9 * i + 3 * r + c] = rot(r, c) - 1;
				else
					pose_bias[9 * i + 3 * r + c] = rot(r, c);
			}
		}
	}

	Eigen::MatrixX3d v_posed = v_shaped;
	if (1)
	{
		v_posed += Dot(mesh_pose_basis_, pose_bias);
	}

	std::vector<Eigen::Matrix4d> G(n_joints);
	G[0] << R[0](0, 0), R[0](0, 1), R[0](0, 2), J(0, 0) + pose_global_T(0),
		R[0](1, 0), R[0](1, 1), R[0](1, 2), J(0, 1) + pose_global_T(1),
		R[0](2, 0), R[0](2, 1), R[0](2, 2), J(0, 2) + pose_global_T(2),
		0.f, 0.f, 0.f, 1.f;
	for (int i = 1; i < n_joints; i++) {
		Eigen::Matrix4d transfrom(4, 4);
		transfrom << R[i](0, 0), R[i](0, 1), R[i](0, 2), J(i, 0) - J(parents_[i], 0),
			R[i](1, 0), R[i](1, 1), R[i](1, 2), J(i, 1) - J(parents_[i], 1),
			R[i](2, 0), R[i](2, 1), R[i](2, 2), J(i, 2) - J(parents_[i], 2),
			0.f, 0.f, 0.f, 1.f;
		G[i] = G[parents_[i]] * transfrom;
	}

	for (int i = 0; i < G.size(); i++) {
		Eigen::Matrix4d diff = Eigen::MatrixXd::Zero(4, 4);
		diff.col(3) = G[i] * Eigen::Vector4d(J(i, 0), J(i, 1), J(i, 2), 0.f);
		G[i] = G[i] - diff;
	}

	Eigen::MatrixXd T_mat = skinning_weights_ * ToMatrix(G);
	std::vector<Eigen::Matrix4d> T = ToTensor(T_mat);

	Eigen::MatrixX3d verts(T.size(), 3);


	for (int i = 0; i < T.size(); i++) {
		Eigen::Vector4d v = T[i] * Eigen::Vector4d(v_posed(i, 0), v_posed(i, 1), v_posed(i, 2), 1.f);
		verts.row(i) = Eigen::Vector3d(v(0), v(1), v(2));//to simplify
	}

	//int vert_num = T.size();
	//concurrency::parallel_for(0, vert_num, [this, &T, &v_posed, &verts](int i) {
	//	Eigen::Vector4d v = T[i] * Eigen::Vector4d(v_posed(i, 0), v_posed(i, 1), v_posed(i, 2), 1.f);
	//	verts.row(i) = Eigen::Vector3d(v(0), v(1), v(2));//to simplify
	//});

	return verts;
}


Eigen::MatrixX3d MANOModel::get_posed_joints(Eigen::VectorXd pose_pca,  Eigen::Vector3d pose_global_R, Eigen::Vector3d pose_global_T)
{
	Eigen::VectorXd joints = (pose_pca_basis_* pose_pca).col(0) + pose_pca_mean_;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> joints_matrix(joints);
	joints_matrix.resize(joints.rows() / 3, 3);

	Eigen::MatrixX3d pose(n_joints, 3);
	pose << pose_global_R.transpose(),
		joints_matrix;

	std::vector<Eigen::Vector3d> pose_rotation = Split(pose);
	std::vector<Eigen::Matrix3d> R = Rodrigues(pose_rotation);

	std::vector<Eigen::Matrix4d> G(n_joints);
	G[0] << R[0](0, 0), R[0](0, 1), R[0](0, 2), J(0, 0) + pose_global_T(0),
		R[0](1, 0), R[0](1, 1), R[0](1, 2), J(0, 1) + pose_global_T(1),
		R[0](2, 0), R[0](2, 1), R[0](2, 2), J(0, 2) + pose_global_T(2),
		0.f, 0.f, 0.f, 1.f;
	for (int i = 1; i < n_joints; i++) {
		Eigen::Matrix4d transfrom(4, 4);
		transfrom << R[i](0, 0), R[i](0, 1), R[i](0, 2), J(i, 0) - J(parents_[i], 0),
			R[i](1, 0), R[i](1, 1), R[i](1, 2), J(i, 1) - J(parents_[i], 1),
			R[i](2, 0), R[i](2, 1), R[i](2, 2), J(i, 2) - J(parents_[i], 2),
			0.f, 0.f, 0.f, 1.f;
		G[i] = G[parents_[i]] * transfrom;
	}

	for (int i = 0; i < G.size(); i++) {
		Eigen::Matrix4d diff = Eigen::MatrixXd::Zero(4, 4);
		diff.col(3) = G[i] * Eigen::Vector4d(J(i, 0), J(i, 1), J(i, 2), 0.f);
		G[i] = G[i] - diff;
	}

	Eigen::MatrixXd T_mat = full_joints_weight * ToMatrix(G);
	std::vector<Eigen::Matrix4d> T = ToTensor(T_mat);

	Eigen::MatrixX3d posed_joints(T.size(), 3);
	for (int i = 0; i < T.size(); i++) {
		Eigen::Vector4d v = T[i] * Eigen::Vector4d(J_full_rest(i, 0), J_full_rest(i, 1), J_full_rest(i, 2), 1.f);
		posed_joints.row(i) = Eigen::Vector3d(v(0), v(1), v(2));//to simplify
	}

	//std::cout << "posed_joints:"<<posed_joints << std::endl;


	return posed_joints;
}

Eigen::MatrixX3d MANOModel::get_posed_joints_solver(Eigen::VectorXd para)
{
	Eigen::VectorXd pose_pca = Eigen::VectorXd::Zero(POSE_PCA_NUM);
	Eigen::Vector3d global_R = Eigen::Vector3d::Zero();
	Eigen::Vector3d global_T = Eigen::Vector3d::Zero();

	for (int i = 0; i < POSE_PCA_NUM; i++)
		pose_pca(i) = para[i];

	global_R(0) = para[POSE_PCA_NUM + 0];
	global_R(1) = para[POSE_PCA_NUM + 1];
	global_R(2) = para[POSE_PCA_NUM + 2];

	global_T(0) = para[POSE_PCA_NUM + 3];
	global_T(1) = para[POSE_PCA_NUM + 4];
	global_T(2) = para[POSE_PCA_NUM + 5];

	return get_posed_joints(pose_pca, global_R, global_T);
}

std::vector<Eigen::Matrix3d> MANOModel::Rodrigues(std::vector<Eigen::Vector3d> r)
{
    std::vector<Eigen::Matrix3d> R;
    for(auto & rr: r){
        double theta = std::max(rr.norm(), 0.0000001);//FLT_EPSILON
//        float theta = rr.norm();
        Eigen::Vector3d hat = rr/theta;
        double cos = std::cos(theta);
        double sin = std::sin(theta);
        Eigen::Matrix3d I = Eigen::MatrixXd::Identity(3,3);
        Eigen::Matrix3d dot = hat * hat.transpose();
        Eigen::Matrix3d m;
        m << 0, -hat(2), hat(1), hat(2), 0, -hat(0), -hat(1), hat(0), 0;
        R.push_back(cos * I + (1 - cos) * dot + sin * m);
    }
    return R;
}

//void MANOModel::ExportObj(std::string filepath)
//{
//    std::ofstream file(filepath);
//    if(!file.is_open())
//        return;
//    for(int i = 0; i < verts_.rows();i++)
//    {
//        file << "v " << verts_(i, 0) << " " << verts_(i, 1) << " "<< verts_(i, 2) << "\n";
//    }
//
//    for(int i = 0; i < faces_.rows(); i++){
//        file << "f " << faces_(i, 0) + 1 << " " << faces_(i, 1) + 1 << " "<< faces_(i, 2) + 1 << "\n";
//    }
//
//    file.close();
//}

void MANOModel::ExportObj2(std::string filepath, Eigen::MatrixX3d verts, Eigen::MatrixX3i faces)
{
	std::ofstream file(filepath);
	if (!file.is_open())
		return;
	for (int i = 0; i < verts.rows(); i++)
	{
		file << "v " << verts(i, 0) << " " << verts(i, 1) << " " << verts(i, 2) << "\n";
	}

	for (int i = 0; i < faces.rows(); i++) {
		file << "f " << faces(i, 0) + 1 << " " << faces(i, 1) + 1 << " " << faces(i, 2) + 1 << "\n";
	}

	file.close();
}

void MANOModel::ExportJoints(std::string filepath, Eigen::MatrixX3d joints)
{
	std::ofstream file(filepath);
	if (!file.is_open())
		return;
	for (int i = 0; i < joints.rows(); i++)
	{
		file << "v " << joints(i, 0) << " " << joints(i, 1) << " " << joints(i, 2) << "\n";
	}
	file.close();
}

//bool MANOModel::LoadParams(std::string filepath)
//{
//    std::ifstream file(filepath);
//    if(!file.is_open())
//        return false;
//
//    Json::Value root;
//    Json::Reader reader;
//    bool parsingSuccessful = reader.parse(file, root);
//    file.close();
//
//    if(!parsingSuccessful)
//        return false;
//
//    Json::Value jv_pose_pca = root["pose_pca"];
//    int pose_pca_size = jv_pose_pca.size();
//    Eigen::VectorXf pose_pca(pose_pca_size);
//    for(int i = 0; i < pose_pca_size; i++){
//        Json::Value value = jv_pose_pca[i];
//        pose_pca(i) = value.asFloat();
//    }
//
//    Json::Value jv_shape = root["shape"];
//    int shape_size = jv_shape.size();
//    Eigen::VectorXf shape(shape_size);
//    for(int i = 0; i < shape_size; i++){
//        Json::Value value = jv_shape[i];
//        shape(i) = value.asFloat();
//    }
//
//    Json::Value jv_pose_global_R = root["pose_global_R"];
//    int pose_global_R_size = 3;
//    Eigen::VectorXf pose_global_R(pose_global_R_size);
//    for(int i = 0; i < pose_global_R_size; i++){
//        Json::Value value = jv_pose_global_R[i];
//        pose_global_R(i) = value.asFloat();
//    }
//
//    Json::Value jv_pose_global_T = root["pose_global_T"];
//    int pose_global_T_size = jv_pose_global_T.size();
//    Eigen::VectorXf pose_global_T(pose_global_T_size);
//    for(int i = 0; i < pose_global_T_size; i++){
//        Json::Value value = jv_pose_global_T[i];
//        pose_global_T(i) = value.asFloat();
//    }
//
//    Json::Value jv_pose_deform = root["pose_deform"];
//    bool pose_deform = jv_pose_deform.asBool();
//
//    SetParams(Eigen::MatrixX3d(0,3), pose_pca, shape, pose_global_R, pose_global_T, pose_deform);
//
//    return true;
//}

//bool MANOModel::updateMesh(Mesh * const mesh)
//{
//    mesh->vertices.clear();
//    mesh->indices.clear();
//    mesh->vertices.resize(verts_.rows());
//    mesh->indices.resize(faces_.rows() * 3);
//    int s = 10;
//    for(int i = 0; i < verts_.rows();i++)
//    {
//        mesh->vertices[i] = {QVec3(verts_(i, 0) * s, verts_(i, 1) * s, verts_(i, 2) * s  ), QVec3(0.f, 0.f, 0.f), QVec2(0.f, 0.f)};
//    }
//
//
//    for(int i = 0; i < faces_.rows(); i++){
//        mesh->indices[3 * i + 0] = unsigned(faces_(i, 0));
//        mesh->indices[3 * i + 1] = unsigned(faces_(i, 1));
//        mesh->indices[3 * i + 2] = unsigned(faces_(i, 2));
//    }
//
//    QVec3 ij, ik;
//    QVec3 cross;
//    real_t theta;
//    for(int id= 0; id < mesh->indices.size()/3;id++){
//        for(int i = 0, j, k ; i < 3; i++){
//            j = (i + 1) % 3;
//            k = (i + 2) % 3;
//
//            Vertex &pi = mesh->vertices[mesh->indices[3 * id + i]];
//            Vertex &pj = mesh->vertices[mesh->indices[3 * id + j]];
//            Vertex &pk = mesh->vertices[mesh->indices[3 * id + k]];
//
//            ij = pk.position - pi.position;
//            ik = pj.position - pi.position;
//
//            cross = QVector3D::crossProduct(ij, ik).normalized();
//
//            theta = acosf(QVector3D::dotProduct(ij, ik) /ij.length()/ik.length());
//
//            pi.normal += theta * cross;
//        }
//    }
//
//    return true;
//}

Eigen::MatrixX3d Dot(std::vector<Eigen::Matrix3Xd> tensor, Eigen::VectorXd vector)
{
    Eigen::MatrixX3d temp(tensor.size(), 3);//init
    int i = 0;
    for(auto& mat : tensor){
        if(mat.cols() == vector.rows()){
            temp.row(i) = mat * vector;
        }
        i++;
    }
    return temp;
}

//Eigen::MatrixX3d TensorDot(std::vector<Eigen::Matrix3Xd> tensor, Eigen::VectorXd vector)
//{
//	Eigen::MatrixX<double> ;
//}

std::vector<Eigen::Vector3d> Split(Eigen::MatrixX3d mat)
{
    std::vector<Eigen::Vector3d> vec;
    for(int i = 0;i < mat.rows();i++){
        vec.push_back(mat.row(i));
    }
    return vec;
}

Eigen::VectorXd Ravel(std::vector<Eigen::Matrix3d> mat)
{
    Eigen::VectorXd vec(mat.size() * 9);
    for(auto& m: mat){
//        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m2(m);
        //        Eigen::Map<Eigen::RowVectorXf> v(m2.data(), m2.size());
        vec << m;
    }
    return vec;
}

Eigen::MatrixXd ToMatrix(std::vector<Eigen::Matrix4d> mat4)
{
    Eigen::MatrixXd mat(mat4.size(), 16);
    for(int i = 0; i < mat4.size();i++){
        Eigen::Matrix4d m(mat4[i]);
        Eigen::Map<Eigen::VectorXd> v(m.data(), m.size());
        mat.row(i) = v;//todo check
//        Eigen::Map<Eigen::VectorXf> v(mat4[i].data(), 16);
//        mat.row(i) = mat4[i];
    }
    return mat;
}

std::vector<Eigen::Matrix4d> ToTensor(Eigen::MatrixXd tensor)
{
    std::vector<Eigen::Matrix4d> vec;
    if(tensor.cols()!=16)
        return vec;
    for(int i = 0;i < tensor.rows();i++){
        Eigen::VectorXd v = tensor.row(i);
        Eigen::Map<Eigen::MatrixXd> m(v.data(), 4, 4);
        vec.push_back(m); //todo check
    }
    return vec;
}

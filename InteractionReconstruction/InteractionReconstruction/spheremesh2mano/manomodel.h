#ifndef MANOMODEL_H
#define MANOMODEL_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
//#include "mesh.h"
#include <json/json.h>
#include <ctime>

#define POSE_PCA_NUM 12

//todo
//Eigen 中的Matrix默认的初始值是多少
class MANOModel
{
public:
    MANOModel(std::string filepath);

    /*Eigen::MatrixX3d SetParams(std::vector<Eigen::Matrix3d> mat, Eigen::Vector3d pose_global_T);
    Eigen::MatrixX3d SetParams(Eigen::MatrixX3d pose_abs);*/

	bool Load(std::string filepath);
	//bool LoadParams(std::string filepath);
	void init_rest_model(Eigen::VectorXd shape);

    Eigen::MatrixX3d get_posed_model(Eigen::VectorXd pose_pca, Eigen::Vector3d pose_global_R, Eigen::Vector3d pose_global_T);
    //Eigen::MatrixX3d Update(bool pose_deform);
    std::vector<Eigen::Matrix3d> Rodrigues(std::vector<Eigen::Vector3d> r);

	Eigen::MatrixX3d get_posed_joints(Eigen::VectorXd pose_pca, Eigen::Vector3d pose_global_R,Eigen::Vector3d pose_global_T);

	Eigen::MatrixX3d get_posed_joints_solver(Eigen::VectorXd para);

    void ExportObj(std::string filepath);
	void ExportObj2(std::string filepath, Eigen::MatrixX3d verts_, Eigen::MatrixX3i faces_);
	void ExportJoints(std::string filepath, Eigen::MatrixX3d joints);

	Eigen::MatrixX3d get_mesh_template() { return mesh_template_; }
	//Eigen::MatrixX3d get_posed_mesh() { return verts_; }
	Eigen::MatrixX3i get_faces() { return faces_; }

private:
    int n_joints = 16;
    int n_shape_params = 10;

    Eigen::MatrixX3i faces_;                                //1538*3
    std::vector<int> parents_;                              //16
    //Eigen::MatrixX3d pose_;                                 //16*3
    //Eigen::Vector3d trans_;                                 //3
    //Eigen::VectorXd shape_;                                 //10
    //Eigen::MatrixX3d verts_;                                //778*3
    //Eigen::MatrixX3d verts_mirror_X;                        //778*3

	Eigen::MatrixX3d v_shaped;                              //778*3
    Eigen::MatrixX3d J;                                     //16*3
	Eigen::MatrixX3d J_full_rest;                           //21*3
    //std::vector<Eigen::Matrix3d> R;                         //16*3*3
	Eigen::MatrixXd full_joints_weight;                       //21*16

    Eigen::MatrixXd pose_pca_basis_;                        //45*45
    Eigen::VectorXd pose_pca_mean_;                         //45
    Eigen::MatrixXd J_regressor_;                           //16*778
	Eigen::MatrixXd J_regressor_full_;                      //21*778
    Eigen::MatrixXd skinning_weights_;                      //778*16
    std::vector<Eigen::Matrix3Xd> mesh_pose_basis_;         //778*3*135
    std::vector<Eigen::Matrix3Xd> mesh_shape_basis_;        //778*3*10
    Eigen::MatrixX3d mesh_template_;                        //778*3
};

static Eigen::MatrixX3d Dot(std::vector<Eigen::Matrix3Xd> tensor, Eigen::VectorXd vector);
static Eigen::MatrixX3d TensorDot(std::vector<Eigen::Matrix3Xd> tensor, Eigen::VectorXd vector);

static Eigen::MatrixX3d Dot(Eigen::MatrixXd mat, Eigen::Matrix3Xd mat2);

static std::vector<Eigen::Vector3d> Split(Eigen::MatrixX3d mat);
static Eigen::VectorXd Ravel(std::vector<Eigen::Matrix3d> mat);

static Eigen::MatrixXd ToMatrix(std::vector<Eigen::Matrix4d> mat4);
static std::vector<Eigen::Matrix4d> ToTensor(Eigen::MatrixXd tensor);

#endif // MANOMODEL_H

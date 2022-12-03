#include "Solver.h"
#include <ppl.h>
#include<thread>

#define MULTI_THRE

//#include <ctime>

//Eigen::Affine3d TransformationSolver_CoarseAlign(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out)
//{
//	// Default output
//	Eigen::Affine3d A;
//	A.linear() = Eigen::Matrix3d::Identity(3, 3);
//	A.translation() = Eigen::Vector3d::Zero();
//
//	if (in.cols() != out.cols())
//		throw "Find3DAffineTransform(): input data mis-match";
//
//	// First find the scale, by finding the ratio of sums of some distances,
//	// then bring the datasets to the same scale.
//	double dist_in = 0, dist_out = 0;
//	for (int col = 0; col < in.cols() - 1; col++) {
//		dist_in += (in.col(col + 1) - in.col(col)).norm();
//		dist_out += (out.col(col + 1) - out.col(col)).norm();
//	}
//	if (dist_in <= 0 || dist_out <= 0)
//		return A;
//	double scale = dist_out / dist_in;
//	out /= scale;
//
//	// Find the centroids then shift to the origin
//	Eigen::Vector3d in_ctr = Eigen::Vector3d::Zero();
//	Eigen::Vector3d out_ctr = Eigen::Vector3d::Zero();
//	for (int col = 0; col < in.cols(); col++) {
//		in_ctr += in.col(col);
//		out_ctr += out.col(col);
//	}
//	in_ctr /= in.cols();
//	out_ctr /= out.cols();
//	for (int col = 0; col < in.cols(); col++) {
//		in.col(col) -= in_ctr;
//		out.col(col) -= out_ctr;
//	}
//
//	// SVD
//	Eigen::MatrixXd Cov = in * out.transpose();
//	Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
//
//	// Find the rotation
//	double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
//	if (d > 0)
//		d = 1.0;
//	else
//		d = -1.0;
//	Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
//	I(2, 2) = d;
//	Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();
//
//	// The final transform
//	A.linear() = R;// scale * R;
//	A.translation() = out_ctr - R*in_ctr;// scale*(out_ctr - R*in_ctr);
//
//	return A;
//}
//
//float EnergyTerm(const std::vector<float3> &data_points, const std::vector<float3> &corspd_points, Eigen::Matrix<float, 6, 6> &ATA, Eigen::Matrix<float, 6, 1> &ATb)
//{
//	int points_num = data_points.size();
//	int batch_size = 100;
//	int batch_num = (points_num + batch_size - 1) / batch_size;
//
//	float sum_dist = 0;
//
//	ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	for (int bn = 0; bn < batch_num; bn++)
//	{
//		Eigen::Matrix<float, 6, 6> ATA_temp = Eigen::Matrix<float, 6, 6>::Zero();
//		Eigen::Matrix<float, 6, 1> ATb_temp = Eigen::Matrix<float, 6, 1>::Zero();
//
//		float sum_dist_temp = 0;
//
//		for (int bs = 0; bs < batch_size; bs++)
//		{
//			Eigen::Matrix<float, 3, 6> A = Eigen::Matrix<float, 3, 6>::Zero();
//			Eigen::Matrix<float, 3, 1> b = Eigen::Matrix<float, 3, 1>::Zero();
//
//			int indx = bn*batch_size + bs;
//			if (indx < points_num)
//			{
//				float3 data_point_temp = data_points[indx];
//				float3 model_point_temp = corspd_points[indx];
//
//				A(0, 0) = 0; A(0, 1) = -data_point_temp.z; A(0, 2) = data_point_temp.y; A(0, 3) = 1; A(0, 4) = 0; A(0, 5) = 0;
//				A(1, 0) = data_point_temp.z; A(1, 1) = 0; A(1, 2) = -data_point_temp.x; A(1, 3) = 0; A(1, 4) = 1; A(1, 5) = 0;
//				A(2, 0) = -data_point_temp.y; A(2, 1) = data_point_temp.x; A(2, 2) = 0; A(2, 3) = 0; A(2, 4) = 0; A(2, 5) = 1;
//
//				/*A(0, 0) = 0; A(0, 1) = data_point_temp.z; A(0, 2) = -data_point_temp.y; A(0, 3) = 1; A(0, 4) = 0; A(0, 5) = 0;
//				A(1, 0) = -data_point_temp.z; A(1, 1) = 0; A(1, 2) = data_point_temp.x; A(1, 3) = 0; A(1, 4) = 1; A(1, 5) = 0;
//				A(2, 0) = data_point_temp.y; A(2, 1) = -data_point_temp.x; A(2, 2) = 0; A(2, 3) = 0; A(2, 4) = 0; A(2, 5) = 1;*/
//
//				float3 delta_point = model_point_temp - data_point_temp;
//				//float3 delta_point = data_point_temp - model_point_temp;
//				b(0) = delta_point.x;
//				b(1) = delta_point.y;
//				b(2) = delta_point.z;
//
//				ATA_temp = ATA_temp + A.transpose()*A;
//				ATb_temp = ATb_temp + A.transpose()*b;
//
//				sum_dist_temp += norm(delta_point);
//			}
//		}
//
//		ATA = ATA + ATA_temp;
//		ATb = ATb + ATb_temp;
//
//		sum_dist += sum_dist_temp;
//	}
//
//	return sum_dist / points_num;
//}
//
//mat34 TransformationSolver_FineAlign(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &data_points_Cube_face1, const std::vector<float3> &corspd_points_Cube_face1, const std::vector<float3> &data_points_Cube_face2, const std::vector<float3> &corspd_points_Cube_face2, float &mean_dist)
//{
//	Eigen::Matrix<float, 6, 6> ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	Eigen::Matrix<float, 6, 6> ATA_Sphere, ATA_Cube_face1, ATA_Cube_face2;
//	Eigen::Matrix<float, 6, 1> ATb_Sphere, ATb_Cube_face1, ATb_Cube_face2;
//	static float mean_dist_Sphere, mean_dist_Cube_face1, mean_dist_Cube_face2;
//
//	mean_dist_Sphere = EnergyTerm(data_points_Sphere, corspd_points_Sphere, ATA_Sphere, ATb_Sphere);
//	mean_dist_Cube_face1 = EnergyTerm(data_points_Cube_face1, corspd_points_Cube_face1, ATA_Cube_face1, ATb_Cube_face1);
//	mean_dist_Cube_face2 = EnergyTerm(data_points_Cube_face2, corspd_points_Cube_face2, ATA_Cube_face2, ATb_Cube_face2);
//
//	float factor_Sphere = 1;
//	float factor_Cube = 1;
//
//	ATA = ATA_Sphere*factor_Sphere + ATA_Cube_face1*factor_Cube+ ATA_Cube_face2*factor_Cube;
//	ATb = ATb_Sphere*factor_Sphere + ATb_Cube_face1*factor_Cube+ ATb_Cube_face2*factor_Cube;
////	mean_dist = (mean_dist_Sphere*factor_Sphere + mean_dist_Cube*factor_Cube) / (factor_Sphere + factor_Cube);
//	mean_dist = (mean_dist_Sphere*data_points_Sphere.size() + mean_dist_Cube_face1*data_points_Cube_face1.size() + mean_dist_Cube_face2*data_points_Cube_face2.size()) / (data_points_Sphere.size() + data_points_Cube_face1.size() + data_points_Cube_face2.size());
//
//	/*ATA = ATA_Sphere;
//	ATb = ATb_Sphere;
//	mean_dist = mean_dist_Sphere;*/
//
//	/*ATA = ATA_Cube;
//	ATb = ATb_Cube;
//	mean_dist = mean_dist_Cube;*/
//
//	Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(ATb);
//
//	Eigen::Vector3f axis(x(0), x(1), x(2));
//	float angle = 0;
//	// prevent squared float underflow (less than 1e-23f)
//	if (axis.cwiseAbs().sum() > 1e-20f) {
//		angle = axis.norm();
//		axis.normalize();
//	}
//
//	//Eigen::Matrix3d Rot = Eigen::AngleAxisf(angle, axis).matrix();
//	//Eigen::Matrix4f Transf = Eigen::Matrix4f::Identity();
//
//	//Transf.topLeftCorner<3, 3>() = Rot;
//	//Transf(0, 3) = x(3);
//	//Transf(1, 3) = x(4);
//	//Transf(2, 3) = x(5);
//
//	mat33 rot = Eigen::AngleAxisf(-angle, axis).matrix();
//	float3 traslt = make_float3(x(3), x(4), x(5));
////	traslt = -rot*traslt;
//	return mat34(rot, traslt);
//}
//
//float EnergyTerm2(const std::vector<float3> &data_points, const std::vector<float3> &corspd_points, const std::vector<float3> &corspd_norm, Eigen::Matrix<float, 6, 6> &ATA, Eigen::Matrix<float, 6, 1> &ATb)
//{
//	int points_num = data_points.size();
//	int batch_size = 100;
//	int batch_num = (points_num + batch_size - 1) / batch_size;
//
//	float sum_dist = 0;
//
//	ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	for (int bn = 0; bn < batch_num; bn++)
//	{
//		Eigen::Matrix<float, 6, 6> ATA_temp = Eigen::Matrix<float, 6, 6>::Zero();
//		Eigen::Matrix<float, 6, 1> ATb_temp = Eigen::Matrix<float, 6, 1>::Zero();
//
//		float sum_dist_temp = 0;
//
//		for (int bs = 0; bs < batch_size; bs++)
//		{
//			Eigen::Matrix<float, 1, 6> A = Eigen::Matrix<float, 1, 6>::Zero();
//			float b = 0;
//
//			int indx = bn*batch_size + bs;
//			if (indx < points_num)
//			{
//				float3 data_point_temp = data_points[indx];
//				float3 model_point_temp = corspd_points[indx];
//				float3 model_norm_temp = corspd_norm[indx];
//
//				A(0) = model_norm_temp.y*model_point_temp.z - model_norm_temp.z*model_point_temp.y;
//				A(1) = -model_norm_temp.x*model_point_temp.z + model_norm_temp.z*model_point_temp.x;
//				A(2) = model_norm_temp.x*model_point_temp.y - model_norm_temp.y*model_point_temp.x;
//				A(3) = model_norm_temp.x; 
//				A(4) = model_norm_temp.y; 
//				A(5) = model_norm_temp.z;
//
//				float3 delta_point = model_point_temp - data_point_temp;
//				//float3 delta_point = data_point_temp - model_point_temp;
//				b = dot(delta_point,model_norm_temp);
//				
//				ATA_temp = ATA_temp + A.transpose()*A;
//				ATb_temp = ATb_temp + A.transpose()*b;
//
//				sum_dist_temp += norm(delta_point);
//			}
//		}
//
//		ATA = ATA + ATA_temp;
//		ATb = ATb + ATb_temp;
//
//		sum_dist += sum_dist_temp;
//	}
//
//	return sum_dist / points_num;
//}
//
//mat34 TransformationSolver_FineAlign_CornerConstraint(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &data_points_Corner, const std::vector<float3> &corspd_points_Corner, float &mean_dist, float &mean_corner_dist)
//{
//	Eigen::Matrix<float, 6, 6> ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	Eigen::Matrix<float, 6, 6> ATA_Sphere, ATA_Cube, ATA_Corner;
//	Eigen::Matrix<float, 6, 1> ATb_Sphere, ATb_Cube, ATb_Corner;
//	static float mean_dist_Sphere, mean_dist_Cube, mean_dist_Corner;
//
//	mean_dist_Sphere = EnergyTerm(data_points_Sphere, corspd_points_Sphere, ATA_Sphere, ATb_Sphere);
//	mean_dist_Cube = EnergyTerm(data_points_Cube, corspd_points_Cube, ATA_Cube, ATb_Cube);
//	mean_dist_Corner = EnergyTerm(data_points_Corner, corspd_points_Corner, ATA_Corner, ATb_Corner);
//
//	float factor_Sphere = 0.1;
//	float factor_Cube = 0.1;
//	float factor_Corner = 100;
//
//	ATA = ATA_Sphere*factor_Sphere + ATA_Cube*factor_Cube + ATA_Corner*factor_Corner;
//	ATb = ATb_Sphere*factor_Sphere + ATb_Cube*factor_Cube + ATb_Corner*factor_Corner;
//	mean_dist = (mean_dist_Sphere*factor_Sphere + mean_dist_Cube*factor_Cube) / (factor_Sphere + factor_Cube);
//	mean_corner_dist = mean_dist_Corner;
//
//	/*ATA = ATA_Sphere;
//	ATb = ATb_Sphere;
//	mean_dist = mean_dist_Sphere;*/
//
//	/*ATA = ATA_Cube;
//	ATb = ATb_Cube;
//	mean_dist = mean_dist_Cube;*/
//
//	Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(ATb);
//
//	Eigen::Vector3f axis(x(0), x(1), x(2));
//	float angle = 0;
//	// prevent squared float underflow (less than 1e-23f)
//	if (axis.cwiseAbs().sum() > 1e-20f) {
//		angle = axis.norm();
//		axis.normalize();
//	}
//
//	//Eigen::Matrix3d Rot = Eigen::AngleAxisf(angle, axis).matrix();
//	//Eigen::Matrix4f Transf = Eigen::Matrix4f::Identity();
//
//	//Transf.topLeftCorner<3, 3>() = Rot;
//	//Transf(0, 3) = x(3);
//	//Transf(1, 3) = x(4);
//	//Transf(2, 3) = x(5);
//
//	mat33 rot = Eigen::AngleAxisf(-angle, axis).matrix();
//	float3 traslt = make_float3(x(3), x(4), x(5));
//	//	traslt = -rot*traslt;
//	return mat34(rot, traslt);
//}
//
//mat34 TransformationSolver_FineAlign_Point2Plane(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &corspd_norms_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &corspd_norms_Cube, float &mean_dist)
//{
//	Eigen::Matrix<float, 6, 6> ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	Eigen::Matrix<float, 6, 6> ATA_Sphere = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 6> ATA_Cube = Eigen::Matrix<float, 6, 6>::Zero();
//
//	Eigen::Matrix<float, 6, 1> ATb_Sphere = Eigen::Matrix<float, 6, 1>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb_Cube = Eigen::Matrix<float, 6, 1>::Zero();
//
//	static float mean_dist_Sphere, mean_dist_Cube;
//
//	mean_dist_Sphere = EnergyTerm2(data_points_Sphere, corspd_points_Sphere, corspd_norms_Sphere, ATA_Sphere, ATb_Sphere);
//	mean_dist_Cube = EnergyTerm2(data_points_Cube, corspd_points_Cube, corspd_norms_Cube, ATA_Cube, ATb_Cube);
//
//	float factor_Sphere = 1.0;
//	float factor_Cube = 1.0;
//
//	ATA = ATA_Sphere*factor_Sphere + ATA_Cube*factor_Cube;
//	ATb = ATb_Sphere*factor_Sphere + ATb_Cube*factor_Cube;
//	mean_dist = (mean_dist_Sphere*factor_Sphere + mean_dist_Cube*factor_Cube) / (factor_Sphere + factor_Cube);
//
//	/*ATA = ATA_Sphere;
//	ATb = ATb_Sphere;
//	mean_dist = mean_dist_Sphere;*/
//
//	/*ATA = ATA_Cube;
//	ATb = ATb_Cube;
//	mean_dist = mean_dist_Cube;*/
//
//	Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(ATb);
//
//	Eigen::Vector3f axis(x(0), x(1), x(2));
//	float angle = 0;
//	// prevent squared float underflow (less than 1e-23f)
//	if (axis.cwiseAbs().sum() > 1e-20f) {
//		angle = axis.norm();
//		axis.normalize();
//	}
//
//	//Eigen::Matrix3f Rot = Eigen::AngleAxisf(angle, axis).matrix();
//	//Eigen::Matrix4f Transf = Eigen::Matrix4f::Identity();
//
//	//Transf.topLeftCorner<3, 3>() = Rot;
//	//Transf(0, 3) = x(3);
//	//Transf(1, 3) = x(4);
//	//Transf(2, 3) = x(5);
//
//	mat33 rot = Eigen::AngleAxisf(-angle, axis).matrix();
//	float3 traslt = make_float3(x(3), x(4), x(5));
//	//	traslt = -rot*traslt;
//	return mat34(rot, traslt);
//}
//
//mat34 TransformationSolver_FineAlign_Point2Plane_CornerConstraint(const std::vector<float3> &data_points_Sphere, const std::vector<float3> &corspd_points_Sphere, const std::vector<float3> &corspd_norms_Sphere, const std::vector<float3> &data_points_Cube, const std::vector<float3> &corspd_points_Cube, const std::vector<float3> &corspd_norms_Cube, const std::vector<float3> &data_points_Corner, const std::vector<float3> &corspd_points_Corner, const std::vector<float3> &corspd_norms_Corner, float &mean_dist, float &mean_corner_dist)
//{
//	Eigen::Matrix<float, 6, 6> ATA = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb = Eigen::Matrix<float, 6, 1>::Zero();
//
//	Eigen::Matrix<float, 6, 6> ATA_Sphere = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 6> ATA_Cube = Eigen::Matrix<float, 6, 6>::Zero();
//	Eigen::Matrix<float, 6, 6> ATA_Corner = Eigen::Matrix<float, 6, 6>::Zero();
//
//	Eigen::Matrix<float, 6, 1> ATb_Sphere = Eigen::Matrix<float, 6, 1>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb_Cube = Eigen::Matrix<float, 6, 1>::Zero();
//	Eigen::Matrix<float, 6, 1> ATb_Corner = Eigen::Matrix<float, 6, 1>::Zero();
//
//	static float mean_dist_Sphere, mean_dist_Cube, mean_dist_Corner;
//
//	mean_dist_Sphere = EnergyTerm2(data_points_Sphere, corspd_points_Sphere, corspd_norms_Sphere, ATA_Sphere, ATb_Sphere);
//	mean_dist_Cube = EnergyTerm2(data_points_Cube, corspd_points_Cube, corspd_norms_Cube, ATA_Cube, ATb_Cube);
//	mean_dist_Corner = EnergyTerm(data_points_Corner, corspd_points_Corner, ATA_Corner, ATb_Corner);//corspd_norms_Corner
//
//	float factor_Sphere = 0.1;
//	float factor_Cube = 0.1;
//	float factor_Corner = 100;
//
//	ATA = ATA_Sphere*factor_Sphere + ATA_Cube*factor_Cube + ATA_Corner*factor_Corner;
//	ATb = ATb_Sphere*factor_Sphere + ATb_Cube*factor_Cube + ATb_Corner*factor_Corner;
//	mean_dist = (mean_dist_Sphere*factor_Sphere + mean_dist_Cube*factor_Cube) / (factor_Sphere + factor_Cube);
//	mean_corner_dist = mean_dist_Corner;
//
//	/*ATA = ATA_Sphere;
//	ATb = ATb_Sphere;
//	mean_dist = mean_dist_Sphere;*/
//
//	/*ATA = ATA_Cube;
//	ATb = ATb_Cube;
//	mean_dist = mean_dist_Cube;*/
//
//	Eigen::Matrix<float, 6, 1> x = ATA.ldlt().solve(ATb);
//
//	Eigen::Vector3f axis(x(0), x(1), x(2));
//	float angle = 0;
//	// prevent squared float underflow (less than 1e-23f)
//	if (axis.cwiseAbs().sum() > 1e-20f) {
//		angle = axis.norm();
//		axis.normalize();
//	}
//
//	//Eigen::Matrix3f Rot = Eigen::AngleAxisf(angle, axis).matrix();
//	//Eigen::Matrix4f Transf = Eigen::Matrix4f::Identity();
//
//	//Transf.topLeftCorner<3, 3>() = Rot;
//	//Transf(0, 3) = x(3);
//	//Transf(1, 3) = x(4);
//	//Transf(2, 3) = x(5);
//
//	mat33 rot = Eigen::AngleAxisf(-angle, axis).matrix();
//	float3 traslt = make_float3(x(3), x(4), x(5));
//	//	traslt = -rot*traslt;
//	return mat34(rot, traslt);
//}
//
//float CalculateEnergy(const std::vector<float3> &data_points, const std::vector<float3> &corspd_points)
//{
//	float total_energy = 0;
//
//	int points_num = data_points.size();
//	int batch_size = 100;
//	int batch_num = (points_num + batch_size - 1) / batch_size;
//
//	if (corspd_points.size() != data_points.size())
//	{
//		for (int bn = 0; bn < batch_num; bn++)
//		{
//			float energy_temp = 0;
//			for (int bs = 0; bs < batch_size; bs++)
//			{
//				int data_idx = bn*batch_size + bs;
//				if (data_idx < points_num)
//				{
//					float3 data_point_temp = data_points[data_idx];
//					energy_temp += squared_norm(data_point_temp);
//				}
//			}
//			total_energy += energy_temp;
//		}
//	}
//	else
//	{
//		for (int bn = 0; bn < batch_num; bn++)
//		{
//			float energy_temp = 0;
//			for (int bs = 0; bs < batch_size; bs++)
//			{
//				int data_idx = bn*batch_size + bs;
//				if (data_idx < points_num)
//				{
//					float3 data_point_temp = data_points[data_idx];
//					float3 correspondence_temp = corspd_points[data_idx];
//					energy_temp += squared_norm(data_point_temp - correspondence_temp);
//				}
//			}
//			total_energy += energy_temp;
//		}
//	}
//	return total_energy;
//}
//
//float3 SolveSpherePosition(float3 sphere_pos, const std::vector<float3> data_points, const float R_ball, float &mean_dist)
//{
//	std::vector<float> delta;
//	std::vector<float> dif_x, dif_y, dif_z;
//
//	delta.clear();
//	dif_x.clear();
//	dif_y.clear();
//	dif_z.clear();
//
//	double sum_dist_temp = 0;
//	//calculate the point-to-sphere error and distance
//	for (int i = 0; i < data_points.size(); i++)
//	{
//		float3 delta_vector = data_points[i] - sphere_pos;
//		delta.push_back(squared_norm(delta_vector) - R_ball*R_ball);
//		sum_dist_temp += abs(norm(delta_vector) - R_ball);
//	}
//
//	mean_dist = sum_dist_temp / data_points.size();
//
//	for (int i = 0; i < data_points.size(); i++)
//	{
//		dif_x.push_back(2 * (sphere_pos.x - data_points[i].x));
//		dif_y.push_back(2 * (sphere_pos.y - data_points[i].y));
//		dif_z.push_back(2 * (sphere_pos.z - data_points[i].z));
//	}
//
//	Eigen::Vector3f g = Eigen::Vector3f::Zero();
//	Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
//
//	for (int i = 0; i < delta.size(); i++)
//	{
//		g[0] += 1 * delta[i] * dif_x[i];
//		g[1] += 1 * delta[i] * dif_y[i];
//		g[2] += 1 * delta[i] * dif_z[i];
//
//		H(0, 0) += 1 * dif_x[i] * dif_x[i]; H(0, 1) += 1 * dif_x[i] * dif_y[i]; H(0, 2) += 1 * dif_x[i] * dif_z[i];
//		H(1, 0) += 1 * dif_y[i] * dif_x[i]; H(1, 1) += 1 * dif_y[i] * dif_y[i]; H(1, 2) += 1 * dif_y[i] * dif_z[i];
//		H(2, 0) += 1 * dif_z[i] * dif_x[i]; H(2, 1) += 1 * dif_z[i] * dif_y[i]; H(2, 2) += 1 * dif_z[i] * dif_z[i];
//	}
//
//	Eigen::Vector3f step = H.inverse()*g;
//
//	return  sphere_pos - make_float3(step[0], step[1], step[2]);
//}


Eigen::VectorXd GaussNewtonSolver::calculate_r(const Eigen::MatrixX3d &posed_joints, const Eigen::VectorXd &para)
{
	Eigen::VectorXd r(3 * posed_joints.rows() + regular_num);

	//std::cout << "posed_joints:" << posed_joints << std::endl;

	int joints_number = posed_joints.rows();
	//std::cout << "joints_number:" << std::endl;
	for (int jt_id = 0; jt_id < joints_number; jt_id++)
	{
		r(3 * jt_id + 0) = posed_joints(jt_id, 0);
		r(3 * jt_id + 1) = posed_joints(jt_id, 1);
		r(3 * jt_id + 2) = posed_joints(jt_id, 2);
	}

	for (int re_nu = 0; re_nu < regular_num; re_nu++)
	{
		double temp = para[re_nu];
		r[3 * joints_number + re_nu] = 0.005 * temp;
	}

	//std::cout << "r:" << r << std::endl;

	return r;
}

Eigen::VectorXd GaussNewtonSolver::calculate_derive(MANOModel &hand_model, const Eigen::VectorXd &para, int col_id)
{
	Eigen::VectorXd para1 = para;
	Eigen::VectorXd para2 = para;

	para1(col_id) = para1(col_id) + DERIV_STEP;
	para2(col_id) = para2(col_id) - DERIV_STEP;

	Eigen::MatrixX3d posed_joints1 = hand_model.get_posed_joints_solver(para1);
	Eigen::VectorXd res1 = calculate_r(posed_joints1, para1);

	Eigen::MatrixX3d posed_joints2 = hand_model.get_posed_joints_solver(para2);
	Eigen::VectorXd res2 = calculate_r(posed_joints2, para2);

	Eigen::VectorXd delta = (res1 - res2) / (2 * DERIV_STEP);

	return delta;
}

Eigen::VectorXd GaussNewtonSolver::Solve(MANOModel &hand_model, Eigen::VectorXd &para, const Eigen::VectorXd &given_joints_xyz)
{
	Eigen::Matrix<double, 63+POSE_PCA_NUM, POSE_PCA_NUM+6> Jacobian;

	float last_mse = 0;

	LARGE_INTEGER time_stmp;
	double count_freq, count_interv;
	double time_inter, time_total;

	QueryPerformanceFrequency(&time_stmp);
	count_freq = (double)time_stmp.QuadPart;


	for (int itr = 0; itr < max_itr; itr++)
	{
		Eigen::MatrixX3d posed_joints = hand_model.get_posed_joints_solver(para);
		Eigen::VectorXd residual = calculate_r(posed_joints, para) - given_joints_xyz;
		
		double mse= residual.transpose()*residual;
		mse /= residual.rows();

		//caculate the Jacobian
#ifdef MULTI_THRE
		int para_num = Jacobian.cols();
		concurrency::parallel_for(0, para_num, [this, &hand_model, &para, &Jacobian](int col_id) {
			Jacobian.col(col_id) = calculate_derive(hand_model, para, col_id);
		});
#else
		for (int col_id = 0; col_id < Jacobian.cols(); col_id++)
		{
			Jacobian.col(col_id) = calculate_derive(hand_model, para, col_id);
		}
#endif

		Eigen::Matrix<double, POSE_PCA_NUM+6, POSE_PCA_NUM+6> JTJ = Jacobian.transpose()*Jacobian;

		if (abs(mse - last_mse) < MSE_THRESHOLD)
			return para;

		if (JTJ.determinant()==0)//(abs(JTJ.determinant()) < 1e-9)
		{
			std::cout << "Cannot inverse" << std::endl;
			return para;
		}

		Eigen::VectorXd delta = -JTJ.inverse() * Jacobian.transpose() * residual;

		para += delta;
		last_mse = mse;
	}

	return para;
}

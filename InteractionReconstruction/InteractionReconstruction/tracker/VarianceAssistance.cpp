#include "VarianceAssistance.h"

pcl::device::Intr camera2device_intr(camera_intr& camera_intrinsic)
{
	pcl::device::Intr device_intrinsic(camera_intrinsic.fx, camera_intrinsic.fy, camera_intrinsic.cx, camera_intrinsic.cy);

	return device_intrinsic;
}

mat34 Eigen2mat(Eigen::Matrix4f mat_eigen)
{
	mat34 mat;
	mat.rot.m00() = mat_eigen(0, 0); mat.rot.m01() = mat_eigen(0, 1); mat.rot.m02() = mat_eigen(0, 2);
	mat.rot.m10() = mat_eigen(1, 0); mat.rot.m11() = mat_eigen(1, 1); mat.rot.m12() = mat_eigen(1, 2);
	mat.rot.m20() = mat_eigen(2, 0); mat.rot.m21() = mat_eigen(2, 1); mat.rot.m22() = mat_eigen(2, 2);
	mat.trans.x = mat_eigen(0, 3); mat.trans.y = mat_eigen(1, 3); mat.trans.z = mat_eigen(2, 3);

	return mat;
}
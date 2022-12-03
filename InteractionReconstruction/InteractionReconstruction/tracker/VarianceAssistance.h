#pragma once
#include "MultiView/DataParser.h"
#include "reconstruction/include/pcl/internal.h"

pcl::device::Intr camera2device_intr(camera_intr& camera_intrinsic);

mat34 Eigen2mat(Eigen::Matrix4f mat_eigen);

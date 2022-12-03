#pragma once
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cudax/cuda_glm.h"
#include <vector>

namespace energy{namespace fitting{struct Settings;}}
struct CustomJointInfo;
struct ChainElement;
typedef unsigned char uchar;

//== NAMESPACE ================================================================
namespace cudax {
//=============================================================================

///--- Jacobian type
#define NUM_THETAS 29 ///< hand parameter size (size of jacobian row)
struct J_row{ float data[NUM_THETAS]; }; ///< Row of Jacobian
typedef thrust::device_vector<J_row> Jacobian; ///< ROW major!!!

///--- Externally defined resources
extern cudaArray* sensor_depth;

//=============================================================================
} // namespace cudax
//=============================================================================

extern "C"
void kernel_init(energy::fitting::Settings* settings,
                 int _width,
                 int _height,
                 int thetas_size,
                 float H_focal_length_x,
                 float H_focal_length_y,
                 const float *H_inv_proj_matrix, 
				 int d, int num_centers, int num_blocks, int max_num_outlines, int num_tangent_fields, int num_outline_fields, bool htrack, bool test, int model_type);

extern "C"
void kernel_set_camera_para( float H_focal_length_x, float H_focal_length_y);//, const float *H_inv_proj_matrix

extern "C" 
void kernel_upload_kinematic(const std::vector<CustomJointInfo>& jointinfos, const std::vector<ChainElement>& H_kinchains);

extern "C" 
void kernel_upload_model(int d, int num_centers, int num_blocks, int num_outlines, int num_tangent_fields, int num_outline_fields, 
const float * host_pointer_centers, const float * host_pointer_radii, const int * host_pointer_blocks, 
const float * host_pointer_tangent_points, const float * host_pointer_outline, const int * host_pointer_blockid_to_jointid_map);

extern "C"
void kernel_assign_interaction_ptr_joints(float4* warped_vertice_ptr, float4* warped_normal_ptr, unsigned char* interaction_finger_idx_ptr);

extern "C"
void kernel_assign_interaction_ptr_blocks(float4* warped_vertice_ptr, float4* warped_normal_ptr, int3* interaction_sphere_block, float3* interaction_sphere_coordinate, unsigned char* interaction_block_idx_ptr);

extern "C"
void kernel_interaction_assign_rigid_motion(float* rigid_motion_r_ptr, float* rigid_motion_t_ptr);

extern "C" 
void kernel_upload_sensor_silhouette(uchar * H_silhouette_sensor);

extern "C"
void kernel_upload_sensor_indicator(int * sensor_indicator, int num_sensor_points);

extern "C"
void kernel_upload_point_cloud(float3 * point_cloud, int * point_index, int num_points);

extern "C"
void kernel_upload_camera_ray(float3 camera_ray_dir);

extern "C"
void kernel_upload_rendered_indicator(int * rendered_pixels, float * rendered_points, int * rendered_block_ids, int num_rendered_points);

extern "C" 
void kernel_upload_dtform_idxs(int* H_dtform_idxs);

extern "C" 
void kernel_bind();

extern "C" 
void kernel(float* eigen_JtJ, float* eigen_Jte, float & push_error, float & pull_error, bool eval_metric, bool reweight, int id, int iter, 
	int num_sensor_points, int num_rendered_points, int &push_num, int &pull_num);

extern "C"
void kernel2(float* eigen_JtJ, float* eigen_Jte, float & push_error, float & pull_error, bool eval_metric, bool reweight, int id, int iter,
	int num_points, int num_rendered_points/*, int &push_num, int &pull_num*/, bool cal_conf);

extern "C"
void kernel3(float* eigen_JtJ, float* eigen_Jte, bool reweight, int num_points);

extern "C"
void kernel_interaction_joints(float* eigen_JtJ, float* eigen_Jte, int num_points, bool store_result, int frame_idx);

extern "C"
void kernel_interaction_blocks(float* eigen_JtJ, float* eigen_Jte, int num_points, bool store_result, int frame_idx);

extern "C" 
void kernel_unbind();

extern "C"
void kernel_copy_extra_corresp(thrust::host_vector<float4>& H_queries, thrust::host_vector<float4>& H_target, thrust::host_vector<int>& H_idxs);

extern "C"
void kernel_cleanup();

extern "C"
void kernel_memory_tests();

extern "C"
void kernel_constraint_type_image(uchar *, int, int);

extern "C"
void kernel_simplify_jacobian();

extern "C"
void kernel_delete();

extern "C"
void kernel_get_conf(int* conf);

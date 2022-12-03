#ifndef _INTERNAL_H_
#define _INTERNAL_H_

#include <vector_types.h>
#include <pcl/gpu/containers/device_array.h>
//#include <pcl/point_types.h>
#include "gpu/dual_quaternion.hpp"
#include <Eigen/Eigen>
#include "pcl/internal.h"
#include "opencv2/core/core.hpp"
//#include "../Sparse/ITMScene.h"
//#include "../Sparse/ITMRenderState_VH.h"

/************************************************************************/
/* CUDA interface function declaration for warp_field.cu                */
/************************************************************************/

void precompute_knn_field_internal(const pcl::gpu::DeviceArray<float4> &_device_node_coords,
								   pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
								   const int3 &_vol_res, const float3 &_voxel_size);

void allocteVB_precompute_knn_field_nonrigid_node(pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<float4> &_device_weight_volume, pcl::gpu::DeviceArray<float4> &_device_node_coords,
									 pcl::gpu::DeviceArray<ushort4> &_device_knn_index, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device,
									 ITMLib::Objects::ITMScene *scene, float3 voxel_size, AllocationTempData *allocationTempData_host_for_loop, void *, void *initialVB_num);

void allocteVB_precompute_knn_field_nonrigid_node2(pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<int> &valid_block_idx, pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<float4> &_device_weight_volume, pcl::gpu::DeviceArray<float4> &_device_node_coords,
	pcl::gpu::DeviceArray<ushort4> &_device_knn_index, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device,
	ITMLib::Objects::ITMScene *scene, float3 voxel_size, AllocationTempData *allocationTempData_host_for_loop, void *, void *initialVB_num, const int validblock_num, int* validblock_num_device);


void precompute_knn_field(pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<float4> &_device_weight_volume, pcl::gpu::DeviceArray<float4> &_device_node_coords,
							pcl::gpu::DeviceArray<ushort4> &_device_knn_index, float3 voxel_size, void *initialVB_num);

void setVBaroundNode_gpu(pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<float4> &_device_node_coords, float factor, const short VB_w);

void calculate_depth_to_model_residual_gpu(cudaTextureObject_t _live_vmap, cudaTextureObject_t _live_nmap, cudaTextureObject_t _depth_vmap, cudaTextureObject_t _depth_nmap, int _map_rows, int _map_cols, pcl::gpu::DeviceArray<float>& _depth_to_model_residuals);

void flag_nonrigid_node_by_residual_gpu(cudaTextureObject_t _live_vmap, cudaTextureObject_t _depth_vmap, const pcl::gpu::DeviceArray<float>& _depth_to_model_residuals, const pcl::gpu::DeviceArray<float4> &_device_node_coords, const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field, 
							const int depth_width, const int depth_height, const mat34 _RT, const float fx, const float fy, const float cx, const float cy, pcl::gpu::DeviceArray<uchar>& _nonrigid_node_flag, pcl::gpu::DeviceArray<float>& _nonrigid_node_mean_residual, pcl::gpu::DeviceArray<uchar>& _node_can_see_flag);

void flag_nonrigid_node_by_inherit_gpu(const pcl::gpu::DeviceArray<short4> &valid_voxel_block, const pcl::gpu::DeviceArray<float4> &_device_node_coords, ITMLib::Objects::ITMScene *scene, pcl::gpu::DeviceArray<uchar>& _nonrigid_node_flag, float3 voxel_size);

void AllocateVBFromDepthMap_gpu(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose, pcl::gpu::DeviceArray<short4> &valid_voxel_block, 
									pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device, AllocationTempData *allocationTempData_host_for_loop, void *);

void AllocateVBFromDepthMap_gpu2(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose, pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<int> &valid_block_idx,
	pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device, AllocationTempData *allocationTempData_host_for_loop, void *, int* validblock_num_device);

void checkValidDepthForAllocate_DepthMap_gpu(cudaTextureObject_t _live_vmap, cudaTextureObject_t _depth_vmap, int _map_rows, int _map_cols, pcl::gpu::DeviceArray<uchar>& _allocate_flag_NonRigidPart);

void AllocateVBFromDepthMap_gpu_NonRigidPart(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose, const pcl::gpu::DeviceArray<uchar> &allocate_flag_NonRigidPart, pcl::gpu::DeviceArray<short4> &valid_voxel_block,
	pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device, AllocationTempData *allocationTempData_host_for_loop, void *);

void AllocateVBFromDepthMap_gpu_NonRigidNode(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose, const int invalid_area_left, const int invalid_area_right, const int invalid_area_up, const int invalid_area_down,
	pcl::gpu::DeviceArray<short4> &valid_voxel_block, pcl::gpu::DeviceArray<bool> &_device_flag_volume, pcl::gpu::DeviceArray<uchar> &entriesAllocType_device, pcl::gpu::DeviceArray<short4> &blockCoord_device, AllocationTempData *allocationTempData_host_for_loop, void *);

void SetFlagMap_NonRigidNodeArea_gpu(const int _map_rows, const int _map_cols, const int invalid_area_left, const int invalid_area_right, const int invalid_area_up, const int invalid_area_down, pcl::gpu::DeviceArray<uchar>& allocate_flag_NonRigidPart);

void construct_graph_internal(const pcl::gpu::DeviceArray<float4> &_device_node_coords,
							  pcl::gpu::DeviceArray<int> &_device_node_graph, pcl::gpu::DeviceArray<float> &_device_node_graph_dist);

void warp_vertices_normals_out_of_place_internal(const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
												 const pcl::gpu::DeviceArray<float4> &_valid_can_normals,
												 pcl::gpu::DeviceArray<float4> &_warp_vertices_buffer,
												 pcl::gpu::DeviceArray<float4> &_warp_normals_buffer,
												 /*const int3 &_vol_res, */const float3 &_voxel_size,
												 const pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
												 const pcl::gpu::DeviceArray<float4> &_device_node_coords,
												 const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field,
												 const pcl::gpu::DeviceArray<float4> &_weight_volume,
												 const pcl::gpu::DeviceArray<short4> &valid_voxel_block,
												 ITMLib::Objects::ITMScene *scene, mat34 rigidpart_DynamicObject);//ZH added scene

void find_interaction_finger_joint_correspondence_internal(const pcl::device::DeviceArray<float4>& warped_vertex,
											const pcl::device::DeviceArray<float4>& warped_normal,
											const pcl::device::DeviceArray<float4>& cano_vertex,
											const pcl::device::DeviceArray<float4>& cano_normal,
											const mat34 object_motion, const int joint_num, const int max_corres_num,
											const pcl::device::DeviceArray<float4>& joint_positions,
											const pcl::device::DeviceArray<float>& joint_radius,
											pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
											pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer,
											pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
											pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer,
											pcl::device::DeviceArray<unsigned char>& interaction_joint_idx,
											int* InteractCorr_num_device);

void find_interaction_hand_correspondence_internal(const pcl::device::DeviceArray<float4>& warped_vertex,
												   const pcl::device::DeviceArray<float4>& warped_normal,
												   const pcl::device::DeviceArray<float4>& cano_vertex,
												   const pcl::device::DeviceArray<float4>& cano_normal,
												   const mat34 object_motion, const int joint_num, const int max_corres_num,
												   const pcl::device::DeviceArray<float4>& hand_joint_positions,
												   const pcl::device::DeviceArray<float>& hand_joint_radius,
												   const pcl::device::DeviceArray<int3>& hand_blocks,
												   const pcl::device::DeviceArray<unsigned char>& hand_block_idx,
												   pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
												   pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer,
												   pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
												   pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer,
												   pcl::device::DeviceArray<int3>& interaction_sphere_block_buffer,
												   pcl::device::DeviceArray<float3>& interaction_sphere_coordinate_buffer,
												   pcl::device::DeviceArray<unsigned char>& interaction_block_idx_buffer,
												   int* InteractCorr_num_device);

void find_interaction_hand_correspondence_internal_surface_spheremesh(const pcl::device::DeviceArray<float4>& warped_vertex,
												   const pcl::device::DeviceArray<float4>& warped_normal,
												   const pcl::device::DeviceArray<float4>& cano_vertex,
												   const pcl::device::DeviceArray<float4>& cano_normal,
												   const mat34 object_motion, const int joint_num, const int max_corres_num,
												   const pcl::device::DeviceArray<float4>& hand_joint_positions,
												   const pcl::device::DeviceArray<float>& hand_joint_radius,
												   const pcl::device::DeviceArray<int3>& hand_blocks,
												   const pcl::device::DeviceArray<unsigned char>& hand_block_idx,
												   const pcl::device::DeviceArray<unsigned char>& block2phalange, 
												   const pcl::device::DeviceArray<mat34>& phalange_global2local,
												   const pcl::device::DeviceArray<unsigned char>& phalange_centerId,
												   pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
												   pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer,
												   pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
												   pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer,
												   pcl::device::DeviceArray<int3>& interaction_sphere_block_buffer,
												   pcl::device::DeviceArray<float3>& interaction_sphere_coordinate_buffer,
												   pcl::device::DeviceArray<unsigned char>& interaction_block_idx_buffer,
												   pcl::device::DeviceArray<float3>& local_contact_vector_spheremesh_buffer,
												   int* InteractCorr_num_device);

void warp_vertices_in_place_internal(pcl::gpu::DeviceArray<float4> &_device_can_geometry,
									 const int3 &_vol_res, const float3 &_voxel_size,
									 const pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
									 const pcl::gpu::DeviceArray<float4> &_device_node_coords,
									 const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field);

void warp_vertices_normals_in_place_internal(pcl::gpu::DeviceArray<float4> &_device_can_vertices,
											 pcl::gpu::DeviceArray<float4> &_device_can_normals,
											 const int3 &_vol_res, const float3 &_voxel_size,
											 const pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
											 const pcl::gpu::DeviceArray<float4> &_device_node_coords,
											 const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field);

// transform both vertices and normals to depth frame in place
void warp_vertices_normals_to_depth_frame_in_place_internal(pcl::gpu::DeviceArray<float4> &_device_can_vertices,
															pcl::gpu::DeviceArray<float4> &_device_can_normals,
															const int3 &_vol_res, const float3 &_voxel_size,
															const pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
															const pcl::gpu::DeviceArray<float4> &_device_node_coords,
															const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field,
															const mat33 &_w2d_r, const float3 &_w2d_t);

// transform both vertices and normals to color frame in place
void warp_vertices_normals_to_color_frame_in_place_internal(pcl::gpu::DeviceArray<float4> &_device_can_geometry,
															pcl::gpu::DeviceArray<float4> &_device_can_normals,
															const int3 &_vol_res, const float3 &_voxel_size,
															const pcl::gpu::DeviceArray<ushort4> &_device_knn_index,
															const pcl::gpu::DeviceArray<float4> &_device_node_coords,
															const pcl::gpu::DeviceArray<DualQuaternion> &_device_warp_field,
															const mat33 &_w2d_r, const float3 &_w2d_t,
															const mat33 &_d2c_r, const float3 &_d2c_t);

#if 0
void warp_internal2(pcl::gpu::DeviceArray<float4> &_device_can_geometry,
					int3 _vol_res, float3 _voxel_size,
					const pcl::gpu::PtrSz<float4> _device_node_coords,
					const pcl::gpu::PtrSz<Quaternion> _device_warp_field);
#endif

// query k-nearest nodes for each vertex, and return weights
void query_vert_knn_internal(const pcl::gpu::DeviceArray<float4> &_vert_list,
							 pcl::gpu::DeviceArray<int> &_knn_index,
							 pcl::gpu::DeviceArray<float> &_knn_weights,
							 const pcl::gpu::DeviceArray<ushort4> &_precompute_knn_index,
							 const pcl::gpu::DeviceArray<float4> &_node_coords,
							 const int3 &_volume_res, const float3 &_voxel_size);

/*interpolate se3 at each voxel center (debug only)*/
void interp_voxel_se3_internal(pcl::gpu::DeviceArray<mat34> &_voxel_se3,
							   const pcl::gpu::DeviceArray<int> &_knn_index,
							   const pcl::gpu::DeviceArray<float> &_knn_dist,
							   const pcl::gpu::DeviceArray<mat34> &_node_se3,
							   const int3 &_vol_res);

/************************************************************************/
/* CUDA interface function declaration for util_kernels.cu              */
/************************************************************************/

//pcl::gpu::DeviceArray<float4> calc_trimesh_normals_internal(const pcl::gpu::DeviceArray<float4> &_trimesh);
//
//inline pcl::gpu::DeviceArray<pcl::PointXYZ> calc_trimesh_normals(const pcl::gpu::DeviceArray<pcl::PointXYZ> &_trimesh)
//{
//	pcl::gpu::DeviceArray<float4> trimesh_normals = calc_trimesh_normals_internal((const pcl::gpu::DeviceArray<float4>&)_trimesh);
//	return *(pcl::gpu::DeviceArray<pcl::PointXYZ>*)(&trimesh_normals);
//}

/************************************************************************/
/* cuda interface function declaration for color.cu                     */
/************************************************************************/

/*extract color from volume*/
pcl::gpu::DeviceArray<float4> extract_albedo(const pcl::gpu::PtrStep<float4> &_color_volume,
											 int3 _vol_res, float3 _vol_size,
											 const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
											 pcl::gpu::DeviceArray<float4> &_vertex_albedos_buffer);

/*extract albedo of voxels in thin-shell region*/
int extract_shell_voxels(const pcl::gpu::PtrStep<short2> &_tsdf_volume,
						 const pcl::gpu::PtrStep<float4> &_albedo_volume,
						 int3 _vol_res, float3 _vol_size,
						 float _trunc_dist, float _shell_dist,
						 pcl::gpu::DeviceArray<float4> &_voxel_albedo,
						 pcl::gpu::DeviceArray<float3> &_voxel_coords);

/************************************************************************/
/* cuda interface function declaration for spherical_harmonics.cu       */
/************************************************************************/

/*download all color voxels in thin-shell region*/
void download_shell_color_voxels(const pcl::gpu::DeviceArray2D<int> &_tsdf_volume,
								 const pcl::gpu::DeviceArray2D<uchar4> &_color_volume,
								 int3 _vol_res, float3 _voxel_size, float _shell_dist, float _trunc_dist, int &_shell_voxels,
								 pcl::gpu::DeviceArray<float3> &_voxel_color,
								 pcl::gpu::DeviceArray<float3> &_voxel_normal,
								 pcl::gpu::DeviceArray<float3> &_voxel_coords);

/*find voxel candidates (voxel normals, voxel colors, voxel indices) for the first frame*/
void find_voxel_candidates(const pcl::gpu::PtrStepSz<uchar3> &_color_frame,
						   const pcl::device::Intr &_color_intr,
						   const mat33 &_w2d_r, const float3 &_w2d_t,
						   const mat33 &_d2c_r, const float3 &_d2c_t,
						   const pcl::gpu::PtrStep<short2> _tsdf_volume,
						   int3 _vol_res, float3 _vol_size,
						   float _shell_dist, float _trunc_dist,
						   pcl::gpu::DeviceArray<float3> &_voxel_colors,
						   pcl::gpu::DeviceArray<float3> &_voxel_normals,
						   pcl::gpu::DeviceArray<float3> &_voxel_coords,
						   pcl::gpu::DeviceArray<int> &_voxel_indices,
						   int &_candidate_num);

/*find all voxel candidates (with nonrigid deformation)*/
void find_voxel_candidates_nonrigid(const pcl::gpu::PtrStepSz<unsigned short> &_depth_frame,
									const pcl::gpu::PtrStepSz<uchar3> &_color_frame,
									const pcl::device::Intr &_depth_intr,
									const pcl::device::Intr &_color_intr,
									const mat33 &_w2d_r, const float3 &_w2d_t,
									const mat33 &_d2c_r, const float3 &_d2c_t,
									const pcl::gpu::PtrStep<short2> &_tsdf_volume,
									const pcl::gpu::PtrStep<float4> &_albedo_volume,
									const int3 &_vol_res, const float3 &_vol_size,
									float _shell_dist, float _trunc_dist,
									const pcl::gpu::PtrSz<float3> &_node_coords,
									const pcl::gpu::PtrSz<mat34> &_node_se3,
									const pcl::gpu::PtrSz<ushort4> &_knn_field,
									pcl::gpu::DeviceArray<float3> &_voxel_colors,
									pcl::gpu::DeviceArray<float3> &_voxel_normals,
									pcl::gpu::DeviceArray<float3> &_voxel_coords,
									pcl::gpu::DeviceArray<int> &_voxel_indices,
									pcl::gpu::DeviceArray<float3> &_voxel_albedos,
									int &_candidate_num);

/*calculate spherical harmonics lighting coefficients*/
std::vector<float> calc_SH_lighting(const pcl::gpu::DeviceArray<float3> &_voxel_color,
									const pcl::gpu::DeviceArray<float3> &_voxel_normal);

/*solve spherical harmonics lighting coefficients with previous lighting constraint*/
std::vector<float> calc_SH_lighting_constrained(const pcl::gpu::DeviceArray<float3> &_voxel_normals,
												const pcl::gpu::DeviceArray<float3> &_voxel_colors,
												const pcl::gpu::DeviceArray<float3> &_voxel_albedos,
												const std::vector<float> &_light_coeffs_prev,
												cv::Mat _vn_stats);

/*calculate lighting coefficients on 2D image (constant albedo without previous lighting constraint)*/
std::vector<float> calc_SH_lighting_2d(const cv::Mat &_normal_map, const cv::Mat &_color_map);

/*calculate spherical harmonics lighting coefficients on 2D image*/
std::vector<float> calc_SH_lighting_2d_constrained(const cv::Mat &_live_nmap,
												   const cv::Mat &_live_amap,
												   const cv::Mat &_color_image,
												   const std::vector<float> &_light_coeffs_prev);

// calculate lighting coefficients on device (with constraint)
std::vector<float> calc_SH_lighting_2d_constrained_gpu(cudaTextureObject_t _nmap,
													   cudaTextureObject_t _amap,
													   cudaTextureObject_t _cmap,
													   int _width, int _height,
													   const std::vector<float> &_light_coeffs_prev);

/*calculate point cloud albedo in live frame on GPU*/
void calc_point_albedo(const pcl::gpu::DeviceArray<float4> &_live_normals,
					   const pcl::gpu::DeviceArray<uchar4> &_vertex_color,
					   pcl::gpu::DeviceArray<uchar4> &_vertex_albedo,
					   const std::vector<float3> &_light_coeffs);

/*integrate albedo into albedo volume for the first time*/
void integrate_albedo(//intrusive_ptr<ImageTexture> _color_texture_ptr,
					  pcl::gpu::DeviceArray2D<uchar3> _color_frame,
					  const pcl::device::Intr &_color_intr,
					  const mat33 &_w2d_r, const float3 &_w2d_t,
					  const mat33 &_d2c_r, const float3 &_d2c_t,
					  const pcl::gpu::PtrStep<short2> _tsdf_volume,
					  const int3 &_vol_res, const float3 &_vol_size,
					  float _shell_dist, float _trunc_dist,
					  const std::vector<float> &_light_coeffs,
					  pcl::gpu::PtrStep<float4> _albedo_volume);

/*forward spherical harmonics lighting check (debug only)*/
void forward_SH_lighting_check(const pcl::gpu::PtrSz<float4> &_vertex_albedos,
							   const pcl::gpu::PtrSz<float4> &_vertex_normals,
							   const std::vector<float> &_light_coeffs,
							   pcl::gpu::DeviceArray<float4> &_vertex_colors);

/*extract thin shell voxels*/
int extract_thin_shell_voxel(const pcl::gpu::PtrStep<short2> &_tsdf_volume,
							 const pcl::gpu::PtrStep<float4> &_albedo_volume,
							 const int3 &_vol_res, const float3 &_vol_size,
							 float _shell_dist, float _trunc_dist,
							 pcl::gpu::DeviceArray<float3> &_voxel_coords,
							 pcl::gpu::DeviceArray<float3> &_voxel_albedos);

#endif
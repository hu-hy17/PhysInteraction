#ifndef _WARP_FIELD_H_
#define _WARP_FIELD_H_

#include <string>
#include <Eigen/Eigen>
#include <vector>
#include <array>
#include <pcl/gpu/containers/device_array.h>
#include "gpu/constants.h"
#include <vector_types.h>
#include "gpu/dual_quaternion.hpp"
//#include <pcl/point_types.h>
#include "Sparse/ITMScene.h"
#include "Sparse/ITMRenderState_VH.h"
#include "pcl/internal.h"
#include <iostream>

struct AllocationTempData {
	int noAllocatedVoxelEntries;
	int noAllocatedExcessEntries;
	int noVisibleEntries;
};

class WarpField {
public:
	WarpField(void) { 
		ITMSafeCall(cudaMalloc((void**)&allocationTempData_device_for_loop, sizeof(AllocationTempData))); 
		ITMSafeCall(cudaMalloc((void**)&initialVB_num, sizeof(int)));
		ITMSafeCall(cudaMalloc((void**)&noValidBlock_device, sizeof(int)));
		ITMSafeCall(cudaMalloc((void**)&InteractCorr_num_device, sizeof(int)));	}

	~WarpField(void) { 
		ITMSafeCall(cudaFree(allocationTempData_device_for_loop)); 
		ITMSafeCall(cudaFree(initialVB_num)); 
		ITMSafeCall(cudaFree(noValidBlock_device));
		ITMSafeCall(cudaFree(InteractCorr_num_device));	}

	// noncopyable
	WarpField(const WarpField &_src_warp_field) = delete;
	WarpField& operator= (const WarpField &_src_warp_field) = delete;

	/*return current node number on the host*/
	int node_size() const { return (int)h_node_coords.size(); }

	/*set volume resolution and voxel size*/
	void init_volume(const float3);//const Eigen::Vector3i &_volume_res, const Eigen::Vector3f &_volume_size);

	/*sample canonical geometry, construct node graph and initialize se3 of newly sampled nodes*/
	/*currently these are all on CPU, and then synchronized with GPU*/
	//void sample(const pcl::gpu::DeviceArray<pcl::PointXYZ> &_can_geometry);

	//find the correspondences for interaction term
	int find_interaction_finger_joint_corres(const pcl::device::DeviceArray<float4>& warped_vertex, const pcl::device::DeviceArray<float4>& warped_normal, 
		const pcl::device::DeviceArray<float4>& cano_vertex, const pcl::device::DeviceArray<float4>& cano_normal,
		const mat34 object_motion,const int joint_num, const int max_corres_num,
		const pcl::device::DeviceArray<float4>& joint_positions, const pcl::device::DeviceArray<float>& joint_radius, pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
		pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer, pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
		pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer, pcl::device::DeviceArray<unsigned char>& interaction_joint_idx);

	//find the correspondences between the finger surface and object surface for interaction term
	int find_interaction_hand_corres(const pcl::device::DeviceArray<float4>& warped_vertex, const pcl::device::DeviceArray<float4>& warped_normal,
									 const pcl::device::DeviceArray<float4>& cano_vertex, const pcl::device::DeviceArray<float4>& cano_normal,
									 const mat34 object_motion, const int max_corres_num,
									 const pcl::device::DeviceArray<float4>& hand_joints_positions, const pcl::device::DeviceArray<float>& hand_joints_radius,
									 const pcl::device::DeviceArray<int3>& hand_blocks, const pcl::device::DeviceArray<unsigned char>& hand_block_idx, pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
									 pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer, pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
									 pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer, pcl::device::DeviceArray<int3>& interaction_sphere_block_buffer,
									 pcl::device::DeviceArray<float3>& interaction_sphere_coordinate_buffer, pcl::device::DeviceArray<unsigned char>& interaction_block_idx_buffer);

	int find_interaction_hand_corres_surface_contact(const pcl::device::DeviceArray<float4>& warped_vertex, const pcl::device::DeviceArray<float4>& warped_normal,
									 const pcl::device::DeviceArray<float4>& cano_vertex, const pcl::device::DeviceArray<float4>& cano_normal,
									 const mat34 object_motion, const int max_corres_num,
									 const pcl::device::DeviceArray<float4>& hand_joints_positions, const pcl::device::DeviceArray<float>& hand_joints_radius,
									 const pcl::device::DeviceArray<int3>& hand_blocks, const pcl::device::DeviceArray<unsigned char>& hand_block_idx, 
									 const pcl::device::DeviceArray<unsigned char>& block2phalange,const pcl::device::DeviceArray<mat34>& phalange_global2local,
									 const pcl::device::DeviceArray<unsigned char>& phalange_centerId, pcl::device::DeviceArray<float4>& interaction_warped_vertex_buffer,
									 pcl::device::DeviceArray<float4>& interaction_warped_normal_buffer, pcl::device::DeviceArray<float4>& interaction_cano_vertex_buffer,
									 pcl::device::DeviceArray<float4>& interaction_cano_normal_buffer, pcl::device::DeviceArray<int3>& interaction_sphere_block_buffer,
									 pcl::device::DeviceArray<float3>& interaction_sphere_coordinate_buffer, pcl::device::DeviceArray<unsigned char>& interaction_block_idx_buffer,
									 pcl::device::DeviceArray<float3>& local_contact_vector_spheremesh_buffer);


	// new sample member. It chooses node candidates on GPU
	void sample_gpu(const pcl::gpu::DeviceArray<float4> &_can_geometry, const ITMHashEntry *hashTable/*, cudaStream_t m_object_stream*/);
	void sampleNodeForFirstFrame_gpu(const pcl::gpu::DeviceArray<float4> &_can_geometry);
	void sampleNodeFOV_gpu(const pcl::gpu::DeviceArray<float4> &_can_geometry, const pcl::gpu::DeviceArray<float4> &_warped_geometry, int depth_width, int depth_height, const mat34 &RT, float fx, float fy, float cx, float cy);
	void sampleNodeFOV_gpu2(const pcl::gpu::DeviceArray<float4> &_can_geometry, int depth_width, int depth_height, const mat34 &RT, float fx, float fy, float cx, float cy);

	void calculate_depth_to_model_residual(cudaTextureObject_t _live_vmap, cudaTextureObject_t _live_nmap, cudaTextureObject_t _depth_vmap, cudaTextureObject_t _depth_nmap, int _map_rows, int _map_cols, pcl::gpu::DeviceArray<float> & depth_to_model_residual);

	// new sample member. It chooses node candidates on GPU
	void sample_cpu(const pcl::gpu::DeviceArray<float4> &_can_geometry, const ITMHashEntry *hashTable,  ITMHashEntry *hashTable_host);

	/*construct node graph*/
	void construct_graph(/*cudaStream_t m_object_stream*/);

	/*precompute k-nearest node field (on GPU)*/
	//void precompute_knn_field();

	// faster version of precompute_knn_field
	void precompute_knn_field_nonrigid_part(ITMLib::Objects::ITMScene *scene/*, cudaStream_t m_object_stream*/);//, const ITMRenderState_VH *renderState_live
	void precompute_knn_field_allnode();

	/*construct node graph*/
	void construct_graph_for_nonrigid_node();

	void setVBaroundNode(float voxel_size, const short VB_w);
	void setVBaroundNonRigidNode(float voxel_size, const short VB_w);

	/*construct node graph for preliminary classification*/
	void construct_graph_pre_classify();

	/*DFS to classify the node*/
	ORUuchar DFS_classify_graph();
	void node_class_result();

	/*flag the node who has big mean residual*/
	void flag_nonrigid_node_by_residual(cudaTextureObject_t _live_vmap, cudaTextureObject_t _depth_vmap, const pcl::gpu::DeviceArray<float> & depth_to_model_residual, 
		const int depth_width, const int depth_height, const mat34 &RT, const float fx, const float fy, const float cx, const float cy);

	void flag_nonrigid_node_by_inherit(ITMLib::Objects::ITMScene *scene);

	std::vector<ORUuchar> calculate_nonrigid_node_class(FILE *fd, int frame_num);
	std::vector<ORUuchar> inherit_nonrigid_node_class();

	std::vector<ORUuchar> get_nonrigid_node_classId();

	void reorganize_nonrigid_node();

	void extract_nonrigid_mesh(const pcl::gpu::DeviceArray<float4> &_can_geometry, const pcl::gpu::DeviceArray<float4> &_can_normals, pcl::gpu::DeviceArray<float4> &_can_geometry_NonRigidPart, 
							   pcl::gpu::DeviceArray<float4> &_can_normals_NonRigidPart, pcl::gpu::DeviceArray<float4> &_can_geometry_RigidPart, pcl::gpu::DeviceArray<float4> &_can_normals_RigidPart);
	
	std::tuple<	pcl::gpu::DeviceArray<float4>,
				pcl::gpu::DeviceArray<float4>, 
				pcl::gpu::DeviceArray<float4>, 
				pcl::gpu::DeviceArray<float4>>
	extract_and_sychornize_nonrigid_mesh(ITMLib::Objects::ITMScene *scene, const pcl::gpu::DeviceArray<float4> &_can_geometry, const pcl::gpu::DeviceArray<float4> &_can_normals, pcl::gpu::DeviceArray<float4> &_can_geometry_NonRigidPart,
		pcl::gpu::DeviceArray<float4> &_can_normals_NonRigidPart, pcl::gpu::DeviceArray<float4> &_can_geometry_RigidPart, pcl::gpu::DeviceArray<float4> &_can_normals_RigidPart);

	void sync_to_buffers(const pcl::gpu::DeviceArray<float4> &_valid_can_vertices_NonRigidPart, const pcl::gpu::DeviceArray<float4> &_valid_can_normals_NonRigidPart,
						 const pcl::gpu::DeviceArray<float4> &_valid_can_vertices_RigidPart, const pcl::gpu::DeviceArray<float4> &_valid_can_normals_RigidPart,
						 pcl::gpu::DeviceArray<float4> &_valid_can_vertices_buffer_NonRigidPart, pcl::gpu::DeviceArray<float4> &_valid_can_normals_buffer_NonRigidPart,
						 pcl::gpu::DeviceArray<float4> &_valid_can_vertices_buffer_RigidPart, pcl::gpu::DeviceArray<float4> &_valid_can_normals_buffer_RigidPart);

	std::tuple < pcl::gpu::DeviceArray<float4>,
		pcl::gpu::DeviceArray < float4 >>
	copyto_RigidBuffer(const pcl::gpu::DeviceArray<float4> &_can_geometry, const pcl::gpu::DeviceArray<float4> &_can_normals, pcl::gpu::DeviceArray<float4> &_can_geometry_RigidPart_buffer, pcl::gpu::DeviceArray<float4> &_can_normals_RigidPart_buffer);

	/*update warping field on CPU (can also on GPU)*/
	//void update_warp_field(const Eigen::VectorXf &_update_warp_field);

	// warp both vertices and normals out of place
	std::pair<pcl::gpu::DeviceArray<float4>, pcl::gpu::DeviceArray<float4>>
	warp(const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
		 const pcl::gpu::DeviceArray<float4> &_valid_can_normals,
		 pcl::gpu::DeviceArray<float4> &_warp_vertices_buffer,
		 pcl::gpu::DeviceArray<float4> &_warp_normals_buffer,
		 ITMLib::Objects::ITMScene *scene) const;

	// warp both vertices and normals out of place
	std::pair<pcl::gpu::DeviceArray<float4>, pcl::gpu::DeviceArray<float4>>
	warp_nonrigid_mesh(const pcl::gpu::DeviceArray<float4> &_valid_can_vertices,
		 const pcl::gpu::DeviceArray<float4> &_valid_can_normals,
		 pcl::gpu::DeviceArray<float4> &_warp_vertices_buffer,
		 pcl::gpu::DeviceArray<float4> &_warp_normals_buffer,
		 ITMLib::Objects::ITMScene *scene, mat34 rigidpart_DynamicObject/*, cudaStream_t m_object_stream*/) const;

	//ZH added
	std::vector<float4> warpNode(const std::vector<float4> &node_coords_host, const std::vector<DualQuaternion> &h_warp_field) const;
	std::vector<float4> warpnonrigidNode(const std::vector<float4> &node_coords_host, const std::vector<DualQuaternion> &h_warp_field, mat34 rigidpart_DynamicObject) const;

	void CheckNodeinFOV(std::vector<float4> &h_warped_node, int depth_width, int depth_height, const mat34 &_RT, float fx, float fy, float cx, float cy);
	void deleteNode();

	void StoreOldWarpField();
	void CheckAndHerit_WarpField();

	// warp only vertices in place
	//void warp(pcl::gpu::DeviceArray<float4> &_can_geometry) const;

	// warp both vertices and normals in place
 	void warp(pcl::gpu::DeviceArray<float4> &_can_vertices,
 			  pcl::gpu::DeviceArray<float4> &_can_normals) const;

	// warp both vertices and normals to depth frame in place
	//void warp(pcl::gpu::DeviceArray<float4> &_can_vertices,
	//		  pcl::gpu::DeviceArray<float4> &_can_normals,
	//		  mat33 &_w2d_r, float3 &_w2d_t) const;

	// warp both vertices and normals to color frame in place
	//void warp(pcl::gpu::DeviceArray<float4> &_can_geometry,
	//		  pcl::gpu::DeviceArray<float4> &_can_normals,
	//		  mat33 &_w2d_r, float3 &_w2d_t,
	//		  mat33 &_d2c_r, float3 &_d2c_t) const;

	/*warp both point cloud and point normals to color live frame on GPU*/
	/*both point cloud and point normals on GPU will be modified*/
	//void warp(pcl::gpu::DeviceArray<float3> &_point_cloud,
	//		  pcl::gpu::DeviceArray<float3> &_point_normals,
	//		  mat33 &_w2d_r, float3 &_w2d_t,
	//		  mat33 &_d2c_r, float3 &_d2c_t) const;

	/*warp geometry on GPU (use all node se3 within 3-time sampling radius)*/
	//void warp2(pcl::gpu::DeviceArray<pcl::PointXYZ> &_can_geometry) const;

	/*search k-nearest for each vertex in the query and return weights*/
	//std::pair<std::vector<std::array<int, c_knn>>, std::vector<std::array<float, c_knn>>>
	//query_vert_knn(const std::vector<Eigen::Vector4f> &_vert_list) const;

	/*return k-nearest nodes for each node*/
	//std::vector<std::array<int, c_nn>> query_node_knn() const;

	/*return edge number of node graph*/
	//int get_edge_num() const;

	/*return se3 (R and t) of one node*/
	//Eigen::Matrix4f get_node_se3(int _nidx) const;

	/*return coordinates of one node (in canonical frame)*/
	//Eigen::Vector4f get_node_coords(int _nidx) const;

	/*return nonrigid voxel knn index device array*/
	pcl::gpu::DeviceArray<ushort4> get_voxel_knn_index_array() const;

	/*return voxel knn dist device array*/
	//pcl::gpu::DeviceArray<float> get_voxel_knn_dist_array() const;

	/*return nonrigid node coordinates array on device*/
	pcl::gpu::DeviceArray<float4> get_node_coords_array() const;

	/*return nonrigid node coordinates array on device*/
	std::vector<float4> get_node_coords_array_host();

	/*return all node coordinates array on device*/
	pcl::gpu::DeviceArray<float4> get_node_coords_array_all() const;

	/*return nonrigid warp field device array (node se3 array)*/
	pcl::gpu::DeviceArray<DualQuaternion> get_node_se3_array() const;

	/*return nonrigid warp field device array (node se3 array)*/
	std::vector<DualQuaternion> get_node_se3_array_host() const;

	/*return all warp field device array (node se3 array)*/
	pcl::gpu::DeviceArray<DualQuaternion> get_node_se3_array_all() const;

	/*return nonrigid node graph as a device array*/
	pcl::gpu::DeviceArray<int> get_node_graph() const;

	/*return all node graph as a device array*/
	pcl::gpu::DeviceArray<int> get_node_graph_all() const;

	/*return valid voxel flag volume*/
	pcl::gpu::DeviceArray<bool> get_flag_volume() const;

	/*return weight volume*/
	pcl::gpu::DeviceArray<float4> get_weight_volume() const;

	/*return valid voxel block*/
	pcl::gpu::DeviceArray<short4> get_valid_voxelblock() const;

	/*return valid voxel block index array*/
	pcl::gpu::DeviceArray<int> get_validblock_idx_array() const;

	/*return the class of node*/
	std::vector<ORUuchar> get_node_class() const;

	/*return nonrigid part allocation VB flag*/
	pcl::gpu::DeviceArray<ORUuchar> get_allocate_voxelblock_flag_NonRigidPart() const;

	/*return valid block number*/
	int get_ValidBlock_num()const;

	/*factor out rigid part from warp field (this change se3 of each node)*/
	Eigen::Matrix4f factored_rigid_part();
	Eigen::Matrix4f factored_rigid_part2(mat34);
	void regulate_node_motion(mat34 reg_mat);
	void recover_rigid_part(Eigen::Matrix4f &rigid_se3);

	// download warp field on device to host
	void sync_host_warp_field();

	// synchronize the warp field to the all node warp field array
	void sync_warp_field_to_all_node(mat34 rigidpart_DynamicObject);//from here on 20170330

	//////////////////////////////////////////////////////////////////////////
	//ZH added
	//////////////////////////////////////////////////////////////////////////
	Eigen::Vector3f get_voxel_size() const;
	void AllocateVBFromDepthMap(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose);

	void checkValidDepthForAllocate_DepthMap(cudaTextureObject_t _live_vmap, cudaTextureObject_t _depth_vmap, int _map_rows, int _map_cols);
	void AllocateVBFromDepthMap_NonRigidPart(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose);

	void CheckNonRigidNodeArea(std::vector<float4> &h_warped_node, int depth_width, int depth_height, const mat34 &_RT, float fx, float fy, float cx, float cy);
	void AllocateVBFromDepthMap_NonRigidNode(ITMLib::Objects::ITMScene *scene, cudaTextureObject_t _depth_frame, int width, int height, const pcl::device::Intr &_depth_intr, const ORUMatrix4f DepthPose);
	void SetFlagMap_NonRigidNodeArea(int _map_rows, int _map_cols);

	/*debug members*/
	//void draw_node(const std::string &_node_file) const;
	//void draw_graph(const std::string &_graph_file);
	//void load(const std::string &_node_file); /*load sampled nodes from file*/

	/*all kinds of visualizers*/
	//void visualize_warp_field(const pcl::gpu::DeviceArray<float4> &_can_geometry, const std::string &_visual_file) const;

	/*interpolate se3 at each voxel center, and download them back to host (debug only)*/
	//void interp_voxel_se3(const std::string &_se3_file) const;

	/*all kinds of dumpers*/
	//void dump_node_se3(const std::string &_se3_file) const;
	//void dump_solution(const Eigen::VectorXf &_solution, const std::string &_solution_file) const;

	/*only dump node coordinates and se3*/
	void dump(FILE *write_data) const;

	/*only load node coordinates and se3*/
	void load(FILE *read_data);

private:
	typedef Eigen::Matrix<float, 6, 1> Vector6f;

	/*interpolate se3 using DQB*/
	/*note that it's on CPU and uses partial sort. this may be slow*/
	DualQuaternion get_dqb(const Eigen::Vector4f &_point) const;
	DualQuaternion get_dqb_StaticScene(const Eigen::Vector4f &_point) const;//ZH added

	/*same with get_se3(), but return Eigen::Matrix4f*/
	//Eigen::Matrix4f get_se3_eigen_view(const Eigen::Vector4f &_point) const;

	/*transform twist to se3 matrix*/
	//mat34 twist_to_se3(const Vector6f &_twist) const;

	/*volume parameters*/
	Eigen::Vector3i m_volume_res;
	Eigen::Vector3f m_volume_size;
	Eigen::Vector3f m_voxel_size;

	// a set of updated nodes
	int new_node_offset = 0;
	int new_node_num = 0;

	int fileNumber = 0;
	/*host storage*/
	//ZH: for nonrigid motion estimation node
	//std::vector<Eigen::Vector3f> h_node_coords; /*synchronized with d_node_coords*/
	//std::vector<mat34> h_warp_field; /*synchronized with d_warp_field*/
	std::vector<DualQuaternion> h_warp_field; // host warp_field in dual quaternion format (2 float4)
	std::vector<Eigen::Vector4f> h_node_coords; // host node_coords in float4 with paddings
	std::vector<int> h_node_graph; /*synchronized with d_node_graph*/
	std::vector<float> h_node_graph_dist;
	/*device storage*/
	//pcl::gpu::DeviceArray<float3> d_node_coords;
	pcl::gpu::DeviceArray<float4> d_node_coords; // device node_coords in float4 with paddings
	pcl::gpu::DeviceArray<int> d_node_graph;
	pcl::gpu::DeviceArray<float> d_node_graph_dist;
	//pcl::gpu::DeviceArray<mat34> d_warp_field;
	pcl::gpu::DeviceArray<DualQuaternion> d_warp_field; // device warp_field in dual quaternion format (2 float4)
	pcl::gpu::DeviceArray<ushort4> d_knn_index;
	
	//ZH added
	pcl::gpu::DeviceArray<short4> d_valid_voxel_block;//store the correspondence voxel block coordinate
	pcl::gpu::DeviceArray<int> d_valid_block_idx_array;
	pcl::gpu::DeviceArray<ORUuchar> entriesAllocType_device_for_Allocation;  //What is this to do? this is used for loop fusion
	pcl::gpu::DeviceArray<short4> blockCoords_device_for_Allocation;
	std::vector<ORUuchar> h_node_flag;
	pcl::gpu::DeviceArray<ORUuchar> d_nonrigid_node_flag;
	pcl::gpu::DeviceArray<ORUuchar> d_node_can_see_flag;
	pcl::gpu::DeviceArray<float> d_nonrigid_node_mean_residual;

	//ZH: for basic nodes 
	std::vector<DualQuaternion> h_warp_field_old;
	std::vector<ORUuchar> h_node_class_all;
	int h_node_class_num_all;
	pcl::gpu::DeviceArray<float4> d_node_coords_all;
	std::vector<Eigen::Vector4f> h_node_coords_all;
	pcl::gpu::DeviceArray<DualQuaternion> d_warp_field_all;
	std::vector<DualQuaternion> h_warp_field_all;
	pcl::gpu::DeviceArray<int> d_node_graph_all;
	std::vector<int> h_node_graph_all;
	std::vector<ORUuchar> h_nonrigid_node_class_id;
	std::vector<int> h_nonrigid_node_idx;

	// valid voxel flag volume and weight volume
	pcl::gpu::DeviceArray<bool> d_flag_volume;//bool   uchar
	pcl::gpu::DeviceArray<float4> d_weight_volume;

	pcl::gpu::DeviceArray<ORUuchar> d_allocate_flag_NonRigidPart;

	void *allocationTempData_device_for_loop;
	AllocationTempData allocationTempData_host_for_loop;

	int *noValidBlock_device;
	int *InteractCorr_num_device;
	int ValidBlock_num=0;

	int nonrigid_width_left_end;
	int nonrigid_width_right_end;
	int nonrigid_height_up_end;
	int nonrigid_height_down_end;

	void *initialVB_num;
};

#endif
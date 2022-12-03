#pragma once
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "tracker/OpenGL/ObjectRenderer.h"
#include <cuda_runtime.h>
//#include "../../reconstruction/pcl/safe_call.hpp"
#include "opencv2/core/core.hpp"       ///< cv::Mat

#define TRIANGLES_BUFFER_SIZE  1000000*3

class KinectDataRenderer : public ObjectRenderer{
    QGLBuffer point_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
    QGLBuffer uvbuffer = QGLBuffer(QGLBuffer::VertexBuffer);
    QGLBuffer indexbuffer = QGLBuffer(QGLBuffer::IndexBuffer);

	QGLBuffer line_vertex_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
	QGLBuffer line_color_buffer = QGLBuffer(QGLBuffer::VertexBuffer);

	QGLBuffer img_coord_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
	QGLBuffer img_tex_coord_buffer = QGLBuffer(QGLBuffer::VertexBuffer);

	//QGLBuffer can_vertex_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
	//QGLBuffer can_normal_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
	//QGLBuffer live_vertex_buffer = QGLBuffer(QGLBuffer::VertexBuffer);
	//QGLBuffer live_normal_buffer = QGLBuffer(QGLBuffer::VertexBuffer);

    int num_indexes = 0;
    int num_vertices = 0;
    GLuint texture_id_color = 0;
    GLuint texture_id_depth = 0;
    GLuint texture_id_cmap = 0;
    Camera* camera = NULL;
    float alpha = 1.0;

public:
	//ZH:frame buffer object and texture for rigid motion estimation  
	GLuint fbo_rigid_motion_c0;
	GLuint tex_live_vmap_c0;
	GLuint tex_live_nmap_c0;
	GLuint tex_zbuffer; 
	GLuint fbo_rigid_motion_c1;
	GLuint tex_live_vmap_c1;
	GLuint tex_live_nmap_c1;

	//ZH:frame buffer object and texture for nonrigid motion estimation  
	GLuint fbo_nonrigid_motion_c0;
	GLuint tex_can_vmap_c0;
	GLuint tex_can_nmap_c0;
	GLuint tex_can_zbuffer;
	GLuint fbo_nonrigid_motion_c1;
	GLuint tex_can_vmap_c1;
	GLuint tex_can_nmap_c1;

	//array buffers (to render object model and mapped to cuda resources)
	GLuint can_vertex_buffer;
	GLuint can_normal_buffer;
	GLuint warp_vertex_buffer;
	GLuint warp_normal_buffer;

	//array buffers to render image
	GLuint quad_vbo;
	GLuint quad_vbo2;
	GLuint texture_img;

public:
    void init(Camera* camera);
    void setup(GLuint texture_id_color, GLuint texture_id_depth);
    void render();
	void render_point(std::vector<float3>& points_array, Eigen::Vector3f color_value);
	void render_object_nodes(std::vector<float3>& points_array, Eigen::Vector3f color_value);
	void render_object_index_nodes(std::vector<float3>& points_array, std::vector<int>& node_idx, Eigen::Vector3f color_value);
	void render_interaction_correspondences(std::vector<float3>& points_array, std::vector<unsigned char>& finger_idx);
	void render_object_nodes2tips(std::vector<float3>& nodes_array, std::vector<std::vector<int>>& nodes_tip, std::vector<float>& variant_smooth);
	void render_object_model(std::vector<float4>& points_v, std::vector<float4>& points_n, int vertex_number);
	void render_canonical_model(std::vector<float4>& points_v, std::vector<float4>& points_n, int vertex_number);
	void render_object_model_texture(int vertex_number);
	void render_object_model_texture_nonrigid(int vertex_number);
	void render3(std::vector<float3>& points_array, Eigen::Vector3f color_value);
	void render_img(cv::Mat &img);
	void render_line(std::vector<float3>& line_v, std::vector<float3>& live_c);

    void set_alpha(float alpha);
    void set_zNear(float alpha);
    void set_zFar(float alpha);
    void enable_colormap(bool enable);
    void set_discard_cosalpha_th(float val);

	void set_point_color(const Eigen::Vector3f& given_color);
	void set_projection_point(const Eigen::Matrix4f& value);

	void set_view_projection_object(const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& rigid_motion);

	void set_view_projection_texture(const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& rigid_motion);
	void set_view_projection_texture_nonrigid(const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& rigid_motion);

	void set_projection_line(const Eigen::Matrix4f& value);
};

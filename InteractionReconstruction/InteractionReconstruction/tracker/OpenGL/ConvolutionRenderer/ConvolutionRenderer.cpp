#pragma once
#include "ConvolutionRenderer.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <Eigen/Dense>
#include <GL/glew.h> 
#include <OpenGP/GL/EigenOpenGLSupport3.h>
#include <QGLBuffer>
#include <QGLShaderProgram>
#include <QOpenGLVertexArrayObject>

#include <iostream>
#include <fstream>

void create_smooth_thumb_fold_and_realistic_thumb_top(Model * model, std::vector<glm::vec3> & new_centers, std::vector<float> & new_radii, std::vector<glm::ivec3> & new_blocks, std::vector<Tangent> & new_tangent_points) {

	auto project_point_on_segment = [](const glm::vec3 & p, const glm::vec3 & c1, const glm::vec3 & c2) {
		glm::vec3 u = c2 - c1; glm::vec3 v = p - c1;
		float alpha = dot(u, v) / dot(u, u);
		if (alpha <= 0) return c1;
		if (alpha > 0 && alpha < 1) return c1 + alpha * u;
		if (alpha >= 1) return c2;
	};

	/// Shift palm-fold center
	float thumb_fold_alpha = glm::length(new_centers[33] - new_centers[17]) / glm::length(new_centers[17] - new_centers[18]);
	float thumb_fold_beta = 1 - thumb_fold_alpha;
	//new_centers[33] = new_centers[33] + new_radii[17] * (new_centers[24] - new_centers[33]) / glm::length(new_centers[24] - new_centers[33]);

	/// Shift palm-thumb
	if(false)
	{

		/*if (model->calibration_type == FULL &&
		(model->beta[model->shape_dofs_name_to_id_map["index_palm_center_x"]] + model->beta[model->shape_dofs_name_to_id_map["palm_index_radius"]] >
		model->beta[model->shape_dofs_name_to_id_map["index_base_x"]] + model->beta[model->shape_dofs_name_to_id_map["index_base_radius"]])) {
		new_centers[24] = new_centers[23] + (new_radii[23] - new_radii[24]) * glm::vec3(1, 0, 0);
		}*/
		glm::vec3 projection = project_point_on_segment(new_centers[24], new_centers[15], new_centers[14]);
		new_centers[24] = projection + (new_radii[15] - new_radii[24]) * (new_centers[24] - projection) / glm::length(new_centers[24] - projection);

		glm::vec3 palm_back = model->centers[model->centers_name_to_id_map["palm_back"]];
		glm::vec3 index_base = model->centers[model->centers_name_to_id_map["index_base"]];
		glm::vec3 pinky_base = model->centers[model->centers_name_to_id_map["pinky_base"]];
		glm::vec3 palm_normal = glm::cross(palm_back - index_base, palm_back - pinky_base);
		palm_normal = palm_normal / glm::length(palm_normal);
		glm::vec3 camera_direction = glm::vec3(0, 0, 1);
		if (glm::dot(camera_direction, palm_normal) > 0.3) {
			new_centers[24] += 0.3f * new_radii[15] * glm::vec3(0, 0, 1);
			new_centers[23][0] = new_centers[15][0] + new_radii[15] - new_radii[23];
			//new_centers[23][0] = new_centers[24][0] - (new_radii[23] - 1.8 * new_radii[24]);
			new_centers[23][2] = new_centers[15][2] + 0.5 * (new_radii[15] - new_radii[23]);
		}


		size_t block_index = 24;
		model->compute_tangent_point(model->camera_ray, new_centers[new_blocks[block_index][0]], new_centers[new_blocks[block_index][1]], new_centers[new_blocks[block_index][2]],
									 new_radii[new_blocks[block_index][0]], new_radii[new_blocks[block_index][1]], new_radii[new_blocks[block_index][2]], new_tangent_points[block_index].v1,
									 new_tangent_points[block_index].v2, new_tangent_points[block_index].v3, new_tangent_points[block_index].u1, new_tangent_points[block_index].u2,
									 new_tangent_points[block_index].u3, new_tangent_points[block_index].n, new_tangent_points[block_index].m);
		block_index = 25;
		model->compute_tangent_point(model->camera_ray, new_centers[new_blocks[block_index][0]], new_centers[new_blocks[block_index][1]], new_centers[new_blocks[block_index][2]],
									 new_radii[new_blocks[block_index][0]], new_radii[new_blocks[block_index][1]], new_radii[new_blocks[block_index][2]], new_tangent_points[block_index].v1,
									 new_tangent_points[block_index].v2, new_tangent_points[block_index].v3, new_tangent_points[block_index].u1, new_tangent_points[block_index].u2,
									 new_tangent_points[block_index].u3, new_tangent_points[block_index].n, new_tangent_points[block_index].m);
		block_index = 19;
		model->compute_tangent_point(model->camera_ray, new_centers[new_blocks[block_index][0]], new_centers[new_blocks[block_index][1]], new_centers[new_blocks[block_index][2]],
									 new_radii[new_blocks[block_index][0]], new_radii[new_blocks[block_index][1]], new_radii[new_blocks[block_index][2]], new_tangent_points[block_index].v1,
									 new_tangent_points[block_index].v2, new_tangent_points[block_index].v3, new_tangent_points[block_index].u1, new_tangent_points[block_index].u2,
									 new_tangent_points[block_index].u3, new_tangent_points[block_index].n, new_tangent_points[block_index].m);
	}



	/// Shift finger bases if the are infront of the palm
	bool shifting = false;
	if (shifting) {
		glm::vec3 palm_back = model->centers[model->centers_name_to_id_map["palm_back"]];
		glm::vec3 index_base = model->centers[model->centers_name_to_id_map["index_base"]];
		glm::vec3 pinky_base = model->centers[model->centers_name_to_id_map["pinky_base"]];
		glm::vec3 palm_normal = glm::cross(palm_back - index_base, palm_back - pinky_base);
		palm_normal = palm_normal / glm::length(palm_normal);
		glm::vec3 camera_direction = glm::vec3(0, 0, 1);
		if (glm::dot(camera_direction, palm_normal) > 0.6) {

			auto shift_center_behind_model_surface = [&](std::string first_center_name, std::string second_center_name, std::string shifted_center_name) {
				glm::vec3 a = new_centers[model->centers_name_to_id_map[first_center_name]] - new_radii[model->centers_name_to_id_map[first_center_name]] * glm::vec3(0, 0, 1);
				glm::vec3 b = new_centers[model->centers_name_to_id_map[second_center_name]] - new_radii[model->centers_name_to_id_map[second_center_name]] * glm::vec3(0, 0, 1);
				glm::vec3 c = new_centers[model->centers_name_to_id_map[shifted_center_name]] - new_radii[model->centers_name_to_id_map[shifted_center_name]] * glm::vec3(0, 0, 1);
				glm::vec3 p = project_point_on_segment(c, a, b);
				if (c[2] < p[2]) {
					new_centers[model->centers_name_to_id_map[shifted_center_name]] += glm::vec3(0, 0, p[2] - c[2] + 1.0);
				}
			};

			shift_center_behind_model_surface("palm_index", "palm_pinky", "index_base");
			shift_center_behind_model_surface("palm_index", "palm_pinky", "middle_base");
			shift_center_behind_model_surface("palm_index", "palm_pinky", "ring_base");
			shift_center_behind_model_surface("palm_index", "palm_pinky", "pinky_base");

			shift_center_behind_model_surface("index_base", "index_bottom", "index_membrane");
			shift_center_behind_model_surface("middle_base", "middle_bottom", "middle_membrane");
			shift_center_behind_model_surface("ring_base", "ring_bottom", "ring_membrane");
			shift_center_behind_model_surface("pinky_base", "pinky_bottom", "pinky_membrane");

			for (size_t block_index = 20; block_index < 26; block_index++) {
				model->compute_tangent_point(model->camera_ray, new_centers[new_blocks[block_index][0]], new_centers[new_blocks[block_index][1]], new_centers[new_blocks[block_index][2]],
											 new_radii[new_blocks[block_index][0]], new_radii[new_blocks[block_index][1]], new_radii[new_blocks[block_index][2]], new_tangent_points[block_index].v1,
											 new_tangent_points[block_index].v2, new_tangent_points[block_index].v3, new_tangent_points[block_index].u1, new_tangent_points[block_index].u2,
											 new_tangent_points[block_index].u3, new_tangent_points[block_index].n, new_tangent_points[block_index].m);
			}
		}
	}

	auto process_new_block = [=](size_t block_index, glm::ivec3 block, std::vector<glm::vec3> & centers, const std::vector<float> & radii, std::vector<glm::ivec3> & blocks, std::vector<Tangent> & tangent_points) {

		if (block_index == blocks.size()) {
			blocks.push_back(glm::ivec3(0));
			tangent_points.push_back(Tangent());
		}
		if (block_index >= blocks.size()) cout << "block index is outsize of the range" << endl;

		/// reindex
		if (radii[block[0]] <= radii[block[1]] && radii[block[1]] <= radii[block[2]]) block = glm::ivec3(block[2], block[1], block[0]);
		if (radii[block[0]] <= radii[block[2]] && radii[block[2]] <= radii[block[1]]) block = glm::ivec3(block[1], block[2], block[0]);
		if (radii[block[1]] <= radii[block[0]] && radii[block[0]] <= radii[block[2]]) block = glm::ivec3(block[2], block[0], block[1]);
		if (radii[block[1]] <= radii[block[2]] && radii[block[2]] <= radii[block[0]]) block = glm::ivec3(block[0], block[2], block[1]);
		if (radii[block[2]] <= radii[block[0]] && radii[block[0]] <= radii[block[1]]) block = glm::ivec3(block[1], block[0], block[2]);

		blocks[block_index] = block;

		/// compute tangent points
		model->compute_tangent_point(model->camera_ray, centers[blocks[block_index][0]], centers[blocks[block_index][1]], centers[blocks[block_index][2]],
									 radii[blocks[block_index][0]], radii[blocks[block_index][1]], radii[blocks[block_index][2]], tangent_points[block_index].v1,
									 tangent_points[block_index].v2, tangent_points[block_index].v3, tangent_points[block_index].u1, tangent_points[block_index].u2,
									 tangent_points[block_index].u3, tangent_points[block_index].n, tangent_points[block_index].m);
	};

	/// Smooth thumb fold
	{
		/// new thumb fold 38
		float alpha, beta, gamma; glm::vec3 new_center; float new_radius;
		alpha = 0.63f; beta = 0.18f; gamma = 0.19f;
		new_center = alpha * new_centers[24] + beta * new_centers[33] + gamma * new_centers[19];
		new_radius = alpha * new_radii[24] + beta * new_radii[33] + gamma * new_radii[19];
		new_centers[38] = new_center;
		new_radii[38] = new_radius;

		/// new thumb fold 39
		alpha = 0.42f; beta = 0.32f; gamma = 0.26f;
		new_center = alpha * new_centers[24] + beta * new_centers[33] + gamma * new_centers[19];
		new_radius = alpha * new_radii[24] + beta * new_radii[33] + gamma * new_radii[19];
		new_centers.push_back(new_center);
		new_radii.push_back(new_radius);

		/// new thumb fold 40
		alpha = 0.25f; beta = 0.55f; gamma = 0.20f;
		new_center = alpha * new_centers[24] + beta * new_centers[33] + gamma * new_centers[19];
		new_radius = alpha * new_radii[24] + beta * new_radii[33] + gamma * new_radii[19];
		new_centers.push_back(new_center);
		new_radii.push_back(new_radius);

		/// new center inside of thumb middle phalange 41
		float length = glm::length(new_centers[17] - new_centers[18]);
		alpha = (thumb_fold_alpha * length + thumb_fold_alpha * new_radii[18] + thumb_fold_beta * new_radii[17] - new_radii[33]) / length;
		beta = (thumb_fold_beta * length - thumb_fold_alpha * new_radii[18] - thumb_fold_beta * new_radii[17] + new_radii[33]) / length;
		new_center = alpha * new_centers[18] + beta * new_centers[17];
		new_radius = alpha * new_radii[18] + beta * new_radii[17];
		new_centers.push_back(new_center);
		new_radii.push_back(new_radius);

		process_new_block(14, glm::ivec3(18, 19, 40), new_centers, new_radii, new_blocks, new_tangent_points);

		process_new_block(26, glm::ivec3(19, 24, 38), new_centers, new_radii, new_blocks, new_tangent_points);
		process_new_block(30, glm::ivec3(19, 38, 39), new_centers, new_radii, new_blocks, new_tangent_points);
		process_new_block(31, glm::ivec3(19, 39, 40), new_centers, new_radii, new_blocks, new_tangent_points);
		process_new_block(32, glm::ivec3(19, 40, 33), new_centers, new_radii, new_blocks, new_tangent_points);
		process_new_block(33, glm::ivec3(18, 41, 40), new_centers, new_radii, new_blocks, new_tangent_points);
	}

	/// Realistic thumb top
	{
		size_t block_id = 27;
		Vec3f offset = model->phalanges[model->phalanges_name_to_id_map["HandThumb3"]].offsets[1].cast<float>();
		Mat4f global = model->phalanges[model->phalanges_name_to_id_map["HandThumb3"]].global;
		float new_radius = 1.1 * new_radii[32];

		/// new thumb top 32
		float offset_z = 3.2 - (model->phalanges[model->phalanges_name_to_id_map["HandThumb3"]].offsets[1][2] - model->phalanges[model->phalanges_name_to_id_map["HandThumb3"]].offsets[0][2]);
		if (offset_z < 0) offset_z = 0;

		offset = offset + Vec3f(0, -0.8, offset_z);
		Vec3f new_center = global.block(0, 0, 3, 3) * offset;
		new_centers[32] = new_centers[17] + glm::vec3(new_center[0], new_center[1], new_center[2]);
		new_radii[32] = new_radius;

		/// new center 42
		new_center = Vec3f(new_centers[17][0], new_centers[17][1], new_centers[17][2]) + global.block(0, 0, 3, 3) * (offset + Vec3f(0.7f, -1.2f, 0));
		new_centers.push_back(glm::vec3(new_center[0], new_center[1], new_center[2]));
		new_radii.push_back(new_radius);

		/// new center 43
		new_center = Vec3f(new_centers[17][0], new_centers[17][1], new_centers[17][2]) + global.block(0, 0, 3, 3) * (offset + Vec3f(-0.7f, -1.2f, 0));
		new_centers.push_back(glm::vec3(new_center[0], new_center[1], new_center[2]));
		new_radii.push_back(new_radius);

		process_new_block(34, glm::ivec3(16, 32, 43), new_centers, new_radii, new_blocks, new_tangent_points);
		process_new_block(block_id, glm::ivec3(16, 32, 42), new_centers, new_radii, new_blocks, new_tangent_points);
	}
}

void pass_centers_radii_block_to_shader(GLuint program_id, const std::vector<glm::vec3> & centers, const std::vector<float> & radii, const std::vector<glm::ivec3> & blocks, const std::vector<Tangent> & tangent_points) {
	glUniform1f(glGetUniformLocation(program_id, "num_blocks"), blocks.size());
	glUniform3fv(glGetUniformLocation(program_id, "centers"), centers.size(), (GLfloat *)centers.data());
	glUniform1fv(glGetUniformLocation(program_id, "radii"), radii.size(), (GLfloat *)radii.data());
	glUniform3iv(glGetUniformLocation(program_id, "blocks"), blocks.size(), (GLint *)blocks.data());

	std::vector<Eigen::Vector3f> tangents_v1 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());
	std::vector<Eigen::Vector3f> tangents_v2 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());
	std::vector<Eigen::Vector3f> tangents_v3 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());
	std::vector<Eigen::Vector3f> tangents_u1 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());
	std::vector<Eigen::Vector3f> tangents_u2 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());
	std::vector<Eigen::Vector3f> tangents_u3 = std::vector<Eigen::Vector3f>(tangent_points.size(), Eigen::Vector3f());

	for (size_t i = 0; i < tangent_points.size(); i++) {
		if (tangents_v1.size() <= i) tangents_v1.push_back(Eigen::Vector3f(0.0f));
		if (tangents_v2.size() <= i) tangents_v2.push_back(Eigen::Vector3f(0.0f));
		if (tangents_v3.size() <= i) tangents_v3.push_back(Eigen::Vector3f(0.0f));
		if (tangents_u1.size() <= i) tangents_u1.push_back(Eigen::Vector3f(0.0f));
		if (tangents_u2.size() <= i) tangents_u2.push_back(Eigen::Vector3f(0.0f));
		if (tangents_u3.size() <= i) tangents_u3.push_back(Eigen::Vector3f(0.0f));

		tangents_v1[i] = Eigen::Vector3f(tangent_points[i].v1[0], tangent_points[i].v1[1], tangent_points[i].v1[2]);
		tangents_v2[i] = Eigen::Vector3f(tangent_points[i].v2[0], tangent_points[i].v2[1], tangent_points[i].v2[2]);
		tangents_v3[i] = Eigen::Vector3f(tangent_points[i].v3[0], tangent_points[i].v3[1], tangent_points[i].v3[2]);
		tangents_u1[i] = Eigen::Vector3f(tangent_points[i].u1[0], tangent_points[i].u1[1], tangent_points[i].u1[2]);
		tangents_u2[i] = Eigen::Vector3f(tangent_points[i].u2[0], tangent_points[i].u2[1], tangent_points[i].u2[2]);
		tangents_u3[i] = Eigen::Vector3f(tangent_points[i].u3[0], tangent_points[i].u3[1], tangent_points[i].u3[2]);
	}

	glUniform3fv(glGetUniformLocation(program_id, "tangents_v1"), tangents_v1.size(), (GLfloat *)tangents_v1.data());
	glUniform3fv(glGetUniformLocation(program_id, "tangents_v2"), tangents_v2.size(), (GLfloat *)tangents_v2.data());
	glUniform3fv(glGetUniformLocation(program_id, "tangents_v3"), tangents_v3.size(), (GLfloat *)tangents_v3.data());
	glUniform3fv(glGetUniformLocation(program_id, "tangents_u1"), tangents_u1.size(), (GLfloat *)tangents_u1.data());
	glUniform3fv(glGetUniformLocation(program_id, "tangents_u2"), tangents_u2.size(), (GLfloat *)tangents_u2.data());
	glUniform3fv(glGetUniformLocation(program_id, "tangents_u3"), tangents_u3.size(), (GLfloat *)tangents_u3.data());
}

glm::vec3 ConvolutionRenderer::world_to_window_coordinates(glm::vec3 point) {
	Eigen::Matrix4f view_projection = projection * camera.view;
	glm::mat4 MVP_glm = glm::mat4(0);
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			MVP_glm[j][i] = projection(i, j);
		}
	}

	glm::vec4 point_gl = MVP_glm * glm::vec4(point, 1.0);
	glm::vec3 point_clip = glm::vec3(point_gl[0], point_gl[1], point_gl[2]) / point_gl[3];
	float f = camera.zFar;
	float n = camera.zNear;

	float ox = window_left + window_width / 2;
	float oy = window_bottom + window_height / 2;

	float xd = point_clip[0];
	float yd = point_clip[1];
	float zd = point_clip[2];

	glm::vec3 point_window = glm::vec3(0, 0, 0);
	point_window[0] = xd * window_width / 2 + ox;
	point_window[1] = yd * window_height / 2 + oy;
	point_window[2] = zd * (f - n) / 2 + (n + f) / 2;

	return point_window;
}

ConvolutionRenderer::ConvolutionRenderer(Model *model, bool real_color, std::string data_path) {
	this->data_path = data_path;
	this->model = model;
	this->real_color = real_color;	
}

ConvolutionRenderer::ConvolutionRenderer(Model *model, ConvolutionRenderer::SHADERMODE mode, const Eigen::Matrix4f& projection, std::string data_path) {
	this->data_path = data_path;
	this->model = model;
	this->init(mode);
	this->projection = projection;
}

void ConvolutionRenderer::send_vertices_to_shader(std::string vertices_name) {

	bool success = vertexbuffer.create(); assert(success);
	vertexbuffer.setUsagePattern(QGLBuffer::StaticDraw);
	success = vertexbuffer.bind(); assert(success);
	vertexbuffer.allocate(points.data(), sizeof(points[0]) * points.size());
	program.setAttributeBuffer(vertices_name.c_str(), GL_FLOAT, 0, 3);
	program.enableAttributeArray(vertices_name.c_str());
}

void ConvolutionRenderer::setup_canvas() {

	points = std::vector<Eigen::Vector3f>(4, Eigen::Vector3f::Zero());
	points[0] = Eigen::Vector3f(-1, -1, 0); points[1] = Eigen::Vector3f(1, -1, 0);
	points[2] = Eigen::Vector3f(-1, 1, 0); points[3] = Eigen::Vector3f(1, 1, 0);
	send_vertices_to_shader("position");

	/// Specify window bounds
	glUniform1f(glGetUniformLocation(program.programId(), "window_left"), window_left);
	glUniform1f(glGetUniformLocation(program.programId(), "window_bottom"), window_bottom);
	glUniform1f(glGetUniformLocation(program.programId(), "window_height"), window_height);
	glUniform1f(glGetUniformLocation(program.programId(), "window_width"), window_width);
}

void ConvolutionRenderer::pass_model_to_shader(bool fingers_only) {

	if (mode == FRAMEBUFFER) {
		glm::vec3 min_x_world = glm::vec3(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());
		glm::vec3 min_y_world = glm::vec3(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());
		glm::vec3 max_x_world = -glm::vec3(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());
		glm::vec3 max_y_world = -glm::vec3(numeric_limits<float>::max(), numeric_limits<float>::max(), numeric_limits<float>::max());

		int num_centers = model->centers.size();
		if (fingers_only) num_centers = 34;
		for (size_t i = 0; i < num_centers; i++) {
			if (model->centers[i][0] - model->radii[i] < min_x_world[0]) min_x_world = model->centers[i] - model->radii[i];
			if (model->centers[i][1] - model->radii[i] < min_y_world[1]) min_y_world = model->centers[i] - model->radii[i];
			if (model->centers[i][0] + model->radii[i] > max_x_world[0]) max_x_world = model->centers[i] + model->radii[i];
			if (model->centers[i][1] + model->radii[i] > max_y_world[1]) max_y_world = model->centers[i] + model->radii[i];
		}
		glm::vec3 min_x_window = world_to_window_coordinates(min_x_world);
		glm::vec3 min_y_window = world_to_window_coordinates(min_y_world);
		glm::vec3 max_x_window = world_to_window_coordinates(max_x_world);
		glm::vec3 max_y_window = world_to_window_coordinates(max_y_world);

		glUniform1f(glGetUniformLocation(program.programId(), "min_x"), min_x_window[0]);
		glUniform1f(glGetUniformLocation(program.programId(), "min_y"), min_y_window[1]);
		glUniform1f(glGetUniformLocation(program.programId(), "max_x"), max_x_window[0]);
		glUniform1f(glGetUniformLocation(program.programId(), "max_y"), max_y_window[1]);

		glUniform1i(glGetUniformLocation(program.programId(), "fingers_only"), fingers_only);
		/*for (size_t i = 0; i < 4; i++) {
			for (size_t j= 0; j< 4; j++) {
			cout << camera.MVP_glm[i][j] << " ";
			}
			cout << endl;
			}
			cout << endl << endl;

			cout << "min_x_world = " << min_x_world[0] << endl;
			cout << "min_y_world = " << min_y_world[1] << endl;
			cout << "max_x_world = " << max_x_world[0] << endl;
			cout << "max_y_world = " << max_y_world[1] << endl;

			cout << "min_x = " << min_x_window[0] << endl;
			cout << "min_y = " << min_y_window[1] << endl;
			cout << "max_x = " << max_x_window[0] << endl;
			cout << "max_y = " << max_y_window[1] << endl;*/
	}

	bool adjust_shape_for_display = true;

	if (adjust_shape_for_display) {
		if (mode == NORMAL) {
			std::vector<glm::vec3> new_centers = model->centers;
			std::vector<float> new_radii = model->radii;
			std::vector<glm::ivec3> new_blocks = model->blocks;
			std::vector<Tangent> new_tangent_points = model->tangent_points;
			create_smooth_thumb_fold_and_realistic_thumb_top(model, new_centers, new_radii, new_blocks, new_tangent_points);
			pass_centers_radii_block_to_shader(program.programId(), new_centers, new_radii, new_blocks, new_tangent_points);
		}
	}
	else {
		pass_centers_radii_block_to_shader(program.programId(), model->centers, model->radii, model->blocks, model->tangent_points);
	}

	/*tangents_v1 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	tangents_v2 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	tangents_v3 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	tangents_u1 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	tangents_u2 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	tangents_u3 = std::vector<Eigen::Vector3f>(model->tangent_points.size(), Eigen::Vector3f());
	for (size_t i = 0; i < model->tangent_points.size(); i++) {
		tangents_v1[i] = Eigen::Vector3f(model->tangent_points[i].v1[0], model->tangent_points[i].v1[1], model->tangent_points[i].v1[2]);
		tangents_v2[i] = Eigen::Vector3f(model->tangent_points[i].v2[0], model->tangent_points[i].v2[1], model->tangent_points[i].v2[2]);
		tangents_v3[i] = Eigen::Vector3f(model->tangent_points[i].v3[0], model->tangent_points[i].v3[1], model->tangent_points[i].v3[2]);
		tangents_u1[i] = Eigen::Vector3f(model->tangent_points[i].u1[0], model->tangent_points[i].u1[1], model->tangent_points[i].u1[2]);
		tangents_u2[i] = Eigen::Vector3f(model->tangent_points[i].u2[0], model->tangent_points[i].u2[1], model->tangent_points[i].u2[2]);
		tangents_u3[i] = Eigen::Vector3f(model->tangent_points[i].u3[0], model->tangent_points[i].u3[1], model->tangent_points[i].u3[2]);
	}

	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_v1"), tangents_v1.size(), (GLfloat *)tangents_v1.data());
	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_v2"), tangents_v2.size(), (GLfloat *)tangents_v2.data());
	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_v3"), tangents_v3.size(), (GLfloat *)tangents_v3.data());
	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_u1"), tangents_u1.size(), (GLfloat *)tangents_u1.data());
	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_u2"), tangents_u2.size(), (GLfloat *)tangents_u2.data());
	glUniform3fv(glGetUniformLocation(program.programId(), "tangents_u3"), tangents_u3.size(), (GLfloat *)tangents_u3.data());*/
}

void ConvolutionRenderer::setup_texture() {
	QImage texture_image;
	if (!texture_image.load(QString::fromUtf8(data_path.c_str()) + "shaders//skin_texture.png")) std::cerr << "error loading" << std::endl;
	QImage formatted_image = QGLWidget::convertToGLFormat(texture_image);
	if (formatted_image.isNull()) std::cerr << "error formatting" << std::endl;

	const GLfloat vtexcoord[] = { 0, 0, 1, 0, 0, 1, 1, 1 };

	glGenTextures(1, &synthetic_texture_id);
	glBindTexture(GL_TEXTURE_2D, synthetic_texture_id);

	bool success = texturebuffer.create(); assert(success);
	texturebuffer.setUsagePattern(QGLBuffer::StaticDraw);
	success = texturebuffer.bind(); assert(success);
	texturebuffer.allocate(vtexcoord, sizeof(vtexcoord));
	program.setAttributeBuffer("vtexcoord", GL_FLOAT, 0, 2);
	program.enableAttributeArray("vtexcoord");

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, formatted_image.width(), formatted_image.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, formatted_image.bits());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glUniform1i(glGetUniformLocation(program.programId(), "synthetic_texture"), 0);
}

void ConvolutionRenderer::setup_texture(cv::Mat & image) {

	glGenTextures(1, &real_texture_id);
	glBindTexture(GL_TEXTURE_2D, real_texture_id);

	cv::flip(image, image, 0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.ptr());

	glUniform1i(glGetUniformLocation(program.programId(), "real_texture"), 2);

}

void ConvolutionRenderer::setup_silhoeutte() {
	glGenTextures(1, &silhouette_texture_id);
	glBindTexture(GL_TEXTURE_2D, silhouette_texture_id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, model->silhouette_texture.cols, model->silhouette_texture.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, model->silhouette_texture.ptr());
	glUniform1i(glGetUniformLocation(program.programId(), "silhouette"), 1);

}

void ConvolutionRenderer::init(ConvolutionRenderer::SHADERMODE mode) {
	this->mode = mode;
	if (!vao.isCreated()) {
		bool success = vao.create();
		assert(success);
		vao.bind();
	}

	switch (mode) {
	case NORMAL:
		vertex_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_vshader.glsl";
		fragment_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_fshader.glsl";
		break;
	case FRAMEBUFFER:
		vertex_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_vshader.glsl";
		fragment_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_FB_fshader.glsl";
		window_width = 320; window_height = 240;
		break;
	case RASTORIZER:
		vertex_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_vshader.glsl";
		fragment_shader_name = QString::fromUtf8(data_path.c_str()) + "shaders//" + "model_rastorizer_fshader.glsl";
		window_width = 320; window_height = 240;
		break;
	}

	bool vok = program.addShaderFromSourceFile(QGLShader::Vertex, vertex_shader_name);
	bool fok = program.addShaderFromSourceFile(QGLShader::Fragment, fragment_shader_name);
	bool lok = program.link();
	if (!(lok && vok && fok)) {
		std::cout << "shader compile error: " << std::endl;
		std::cout << "vshader: " << vertex_shader_name.toStdString() << std::endl;
		std::cout << "fshader: " << fragment_shader_name.toStdString() << std::endl;
		std::cout << "shaders log: " << program.log().toStdString() << std::endl;
		exit(EXIT_FAILURE);
	}
	bool success = program.bind();
	assert(success);

	setup_canvas();
	setup_texture();
	setup_silhoeutte();

	material.setup(program.programId());
	light.setup(program.programId());

	camera.setup(program.programId(), projection);

	program.release();
	vao.release();
}

void ConvolutionRenderer::render() {
	vao.bind();
	program.bind();

	//cout << "render" << endl;
	camera.setup(program.programId(), projection);
	if (real_color) setup_texture(model->real_color);
	pass_model_to_shader(false);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, synthetic_texture_id);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, silhouette_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, model->silhouette_texture.cols, model->silhouette_texture.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, model->silhouette_texture.ptr());

	if (real_color) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, real_texture_id);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, points.size());

	program.release();
	vao.release();
}

void ConvolutionRenderer::render2(Eigen::Matrix4f view_mat) {
	vao.bind();
	program.bind();

	//cout << "render" << endl;
	camera.setup2(program.programId(), projection, view_mat);
	if (real_color) setup_texture(model->real_color);
	pass_model_to_shader(false);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, synthetic_texture_id);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, silhouette_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, model->silhouette_texture.cols, model->silhouette_texture.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, model->silhouette_texture.ptr());

	if (real_color) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, real_texture_id);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, points.size());

	program.release();
	vao.release();
}

void ConvolutionRenderer::render3(Eigen::Matrix4f camera_projection, Eigen::Matrix4f view_mat, Eigen::Vector3f camera_center) {
	vao.bind();
	program.bind();

	//cout << "render" << endl;
	camera.setup3(program.programId(), camera_projection, view_mat, camera_center);
	if (real_color) setup_texture(model->real_color);
	pass_model_to_shader(false);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, synthetic_texture_id);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, silhouette_texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, model->silhouette_texture.cols, model->silhouette_texture.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, model->silhouette_texture.ptr());

	if (real_color) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, real_texture_id);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, points.size());

	program.release();
	vao.release();
}


void ConvolutionRenderer::render_offscreen(bool fingers_only) {
	vao.bind();
	program.bind();

	//cout << "render_offscreen" << endl;
	camera.setup(program.programId(), projection);
	pass_model_to_shader(fingers_only);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, points.size());

	program.release();
	vao.release();
}
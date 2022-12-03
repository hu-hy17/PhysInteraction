#include "KinectDataRenderer.h"
#include "tracker/Data/Camera.h"
#include "util/mylogger.h"

struct Grid {
	std::vector<unsigned int> indices;
	std::vector<GLfloat> vertices;
	std::vector<GLfloat> texcoords;
	Grid(int grid_width, int grid_height) {
		///--- So that we don't have to bother with connectivity data structure!!
		int primitive_restart_idx = 0xffffffff;
		glPrimitiveRestartIndex(primitive_restart_idx);
		glEnable(GL_PRIMITIVE_RESTART);

		int skip = 1;

		///--- Vertices
		for (int row = 0; row < grid_height; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				Scalar x = col;
				Scalar y = row;
				vertices.push_back(x); /// i [0...width]
				vertices.push_back(y); /// y [0...height]
			}
		}

		///--- TexCoords
		for (int row = 0; row < grid_height; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				Scalar x = col / ((Scalar)grid_width);
				Scalar y = row / ((Scalar)grid_height);
				texcoords.push_back(x); /// u [0,1]
				texcoords.push_back(y); /// v [0,1]
			}
		}

		///--- Faces
		for (int row = 0; row < grid_height - 1; row += skip) {
			for (int col = 0; col < grid_width; col += skip) {
				indices.push_back((row + 1) * grid_width + col);
				indices.push_back(row * grid_width + col);
			}
			indices.push_back(primitive_restart_idx);
		}
	}
};

void KinectDataRenderer::init(Camera *camera) {
	this->camera = camera;

	int max_point_num = 320 * 240;

	///--- Create vertex array object
	if (!vao.isCreated()) {
		bool success = vao.create();
		assert(success);
		vao.bind();
	}

	///--- Load/compile shaders
	if (!program.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/KinectDataRenderer_vshader2.glsl";
		const char* fshader = ":/KinectDataRenderer/KinectDataRenderer_fshader2.glsl";
		bool vok = program.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program.link();
		assert(lok && vok && fok);
		bool success = program.bind();
		assert(success);
	}

	///--- Create vertex buffer/attributes "position"
	{
		bool success = point_buffer.create();
		assert(success);
		point_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success = point_buffer.bind();
		assert(success);
		point_buffer.allocate(max_point_num * 3 * sizeof(GLfloat));
		program.setAttributeBuffer("vpoint", GL_FLOAT, 0, 3);
		program.enableAttributeArray("vpoint");
	}

	///--- Create texture to colormap the point cloud
	{
		// const int sz=2; GLfloat tex[3*sz] = {/*green*/ 0.000, 1.000, 0, /*red*/ 1.000, 0.000, 0,};
		// const int sz=2; GLfloat tex[3*sz] = {/*gray*/ .1, .1, .1, /*black*/ 0.8, 0.8, 0.8};
		// const int sz=3; GLfloat tex[3*sz] = {/*red*/ 1.000, 0.000, 0, /*yellow*/ 1.0, 1.0, 0.0, /*green*/ 0.000, 1.000, 0};

		//const int sz = 3; GLfloat tex[3 * sz] = {/*red*/ 0.2, 0.5, 1.0, /*magenta*/ 0.2, 0.5, 1.0, /*blue*/ 0.000, 0.000, 1.0000 };
		const int sz = 3; GLfloat tex[3 * sz] = {/*red*/ 202.0 / 300, 86.0 / 300, 122.0 / 300, /*magenta*/  202.0 / 300, 86.0 / 300, 122.0 / 300, /*blue*/ 0.40, 0.0, 0.7 };
		//const int sz = 3; GLfloat tex[3 * sz] = {/*red*/ 1.0, 0.35, 0.0, /*magenta*/ 1.0, 0.35, 0.0, /*blue*/ 0.000, 0.000, 1.0000 };
		glActiveTexture(GL_TEXTURE2);
		glGenTextures(1, &texture_id_cmap);
		glBindTexture(GL_TEXTURE_1D, texture_id_cmap);
		glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, sz, 0, GL_RGB, GL_FLOAT, tex);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		program.setUniformValue("colormap", 2 /*GL_TEXTURE2*/);
	}

	///--- @todo upload data to do inverse projection
//	set_uniform("inv_proj_matrix", camera->inv_projection_matrix());

	///--- upload near/far planes
	set_zNear(camera->zNear());
	set_zFar(camera->zFar());
	// set_alpha(1.0); ///< default no alpha blending
	set_alpha(0.87); ///< default no alpha blending
//	set_discard_cosalpha_th(.3);  ///< default sideface clipping

	///--- Avoid pollution
	program.release();
	vao.release();

	/*************************************************************************/
	/*                  initialize gl_array_buffers                          */
	/*************************************************************************/

	{
		/****************************************************************/
		/*             initialize the opengl vertex buffers             */
		glGenBuffers(1, &can_vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, can_vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float)*TRIANGLES_BUFFER_SIZE, 0, GL_DYNAMIC_DRAW);

		glGenBuffers(1, &can_normal_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, can_normal_buffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float)*TRIANGLES_BUFFER_SIZE, 0, GL_DYNAMIC_DRAW);

		glGenBuffers(1, &warp_vertex_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float)*TRIANGLES_BUFFER_SIZE, 0, GL_DYNAMIC_DRAW);

		glGenBuffers(1, &warp_normal_buffer);
		glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
		glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float)*TRIANGLES_BUFFER_SIZE, 0, GL_DYNAMIC_DRAW);

		/****************************************************************/
		/*       register cuda resources to opengl vertex buffer        */
		//cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[0], can_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
		//cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[1], can_normal_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
		//cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[2], warp_vertex_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
		//cudaSafeCall(cudaGraphicsGLRegisterBuffer(&m_buffer_res[3], warp_normal_buffer, cudaGraphicsRegisterFlagsWriteDiscard));
	}


	/*************************************************************************/
	/*         initialize gl vao and program to render object model          */
	/*************************************************************************/
	///--- Create vertex array object
	if (!vao_obj.isCreated()) {
		bool success = vao_obj.create();
		assert(success);
		vao_obj.bind();
	}

	///--- Load/compile shaders
	if (!program_obj.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/ObjectModelRenderer_vshader.glsl";
		const char* fshader = ":/KinectDataRenderer/ObjectModelRenderer_fshader.glsl";
		bool vok = program_obj.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program_obj.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program_obj.link();
		assert(lok && vok && fok);
		bool success = program_obj.bind();
		assert(success);
	}

	///--- Create vertex buffer/attributes "position"
	{
		/*bool success_v = live_vertex_buffer.create();
		assert(success_v);
		live_vertex_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success_v = live_vertex_buffer.bind();
		assert(success_v);
		live_vertex_buffer.allocate(TRIANGLES_BUFFER_SIZE * 4 * sizeof(GLfloat));

		bool success_n = live_normal_buffer.create();
		assert(success_n);
		live_normal_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success_n = live_normal_buffer.bind();
		assert(success_n);
		live_normal_buffer.allocate(TRIANGLES_BUFFER_SIZE * 4 * sizeof(GLfloat));

		success_v = live_vertex_buffer.bind();
		assert(success_v);
		program_obj.setAttributeBuffer("point_v", GL_FLOAT, 0, 4);
		program_obj.enableAttributeArray("point_v");
		success_n = live_normal_buffer.bind();
		assert(success_n);
		program_obj.setAttributeBuffer("point_n", GL_FLOAT, 0, 4);
		program_obj.enableAttributeArray("point_n");*/

		/*********************************************************************/
		/*        use the opengl API other than the Qt gl api                */

		//use the opengl API
		glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
		program_obj.setAttributeBuffer("point_v", GL_FLOAT, 0, 4);
		program_obj.enableAttributeArray("point_v");
		
		glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
		program_obj.setAttributeBuffer("point_n", GL_FLOAT, 0, 4);
		program_obj.enableAttributeArray("point_n");

	}

	//set the light and material attributions
	{
		set_uniform_program(program_obj, "la", Eigen::Vector3f(1.0f, 1.0f, 1.0f));//0.2
		set_uniform_program(program_obj, "ld", Eigen::Vector3f(1.0f, 1.0f, 1.0f));
		set_uniform_program(program_obj, "ls", Eigen::Vector3f(1.0f, 1.0f, 1.0f));
		set_uniform_program(program_obj, "ldir", Eigen::Vector3f(0.0, 0.0, 1.0f));

		set_uniform_program(program_obj, "f_ma", Eigen::Vector3f(0.6f, 0, 0));//0.3f, 0.3f, 0.3f
		set_uniform_program(program_obj, "f_md", Eigen::Vector3f(0.3f, 0.3f, 0.3f));//0.4f, 0.4f, 0.4f
		set_uniform_program(program_obj, "f_ms", Eigen::Vector3f(0.1f, 0.1f, 0.1f));//0.3f, 0.3f, 0.3f
		set_uniform_program(program_obj, "f_ss", 2.0);
	}

	///--- Avoid pollution
	program_obj.release();
	vao_obj.release();

	/*************************************************************************/
	/*     initialize gl vao and program to render object model texture      */
	/*************************************************************************/

	/*****************     for rigid motion estimation     *******************/
	///--- Create vertex array object
	if (!vao_obj_texture.isCreated()) {
		bool success = vao_obj_texture.create();
		assert(success);
		vao_obj_texture.bind();
	}

	///--- Load/compile shaders
	if (!program_obj_texture.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/ObjectModelTextureRenderer_vshader1.glsl";
		const char* fshader = ":/KinectDataRenderer/ObjectModelTextureRenderer_fshader1.glsl";
		bool vok = program_obj_texture.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program_obj_texture.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program_obj_texture.link();
		assert(lok && vok && fok);
		bool success = program_obj_texture.bind();
		assert(success);
	}
	 
	///--- Create vertex buffer/attributes "position"
	{
		/*bool success_v = can_vertex_buffer.create();
		assert(success_v);
		can_vertex_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success_v = can_vertex_buffer.bind();
		assert(success_v);
		can_vertex_buffer.allocate(TRIANGLES_BUFFER_SIZE * 4 * sizeof(GLfloat));

		bool success_n = can_normal_buffer.create();
		assert(success_n);
		can_normal_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success_n = can_normal_buffer.bind();
		assert(success_n);
		can_normal_buffer.allocate(TRIANGLES_BUFFER_SIZE * 4 * sizeof(GLfloat));

		success_v = can_vertex_buffer.bind();
		assert(success_v);
		program_obj_texture.setAttributeBuffer("point_v", GL_FLOAT, 0, 4);
		program_obj_texture.enableAttributeArray("point_v");

		success_n = can_normal_buffer.bind();
		assert(success_n);
		program_obj_texture.setAttributeBuffer("point_n", GL_FLOAT, 0, 4);
		program_obj_texture.enableAttributeArray("point_n");*/

		/*********************************************************************/
		/*        use the opengl API other than the Qt gl api                */

		//use the opengl API
		glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
		program_obj_texture.setAttributeBuffer("point_v", GL_FLOAT, 0, 4);
		program_obj_texture.enableAttributeArray("point_v");

		glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
		program_obj_texture.setAttributeBuffer("point_n", GL_FLOAT, 0, 4);
		program_obj_texture.enableAttributeArray("point_n");

	}

	///--- Avoid pollution
	program_obj_texture.release();
	vao_obj_texture.release();

	//////////////////////////////////////////////////////////////////
	// create texture attachments of view0 for rigid motion estimation
	glCreateTextures(GL_TEXTURE_2D, 1, &tex_live_vmap_c0);
	glTextureStorage2D(tex_live_vmap_c0, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_live_vmap_c0, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(0, tex_live_vmap_c0);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_live_nmap_c0);
	glTextureStorage2D(tex_live_nmap_c0, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_live_nmap_c0, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(1, tex_live_nmap_c0);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_zbuffer);
	glTextureStorage2D(tex_zbuffer, 1, GL_DEPTH_COMPONENT32F, 320, 240);

	glCreateFramebuffers(1, &fbo_rigid_motion_c0);
	// attach textures to framebuffer
	glNamedFramebufferTexture(fbo_rigid_motion_c0, GL_COLOR_ATTACHMENT0, tex_live_vmap_c0, 0);
	glNamedFramebufferTexture(fbo_rigid_motion_c0, GL_COLOR_ATTACHMENT1, tex_live_nmap_c0, 0);
	glNamedFramebufferTexture(fbo_rigid_motion_c0, GL_DEPTH_ATTACHMENT, tex_zbuffer, 0);

	// check completeness of fbo_rigid_icp
	if (glCheckNamedFramebufferStatus(fbo_rigid_motion_c0, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw std::runtime_error("renderer error: fbo_rigid_icp_c0 is not complete.");
	}

	// set draw buffers
	GLenum draw_buffers[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};

	// set draw buffers
	glNamedFramebufferDrawBuffers(fbo_rigid_motion_c0, 2, draw_buffers);

	//////////////////////////////////////////////////////////////////
	// create texture attachments of view1 for rigid motion estimation
	glCreateTextures(GL_TEXTURE_2D, 1, &tex_live_vmap_c1);
	glTextureStorage2D(tex_live_vmap_c1, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_live_vmap_c1, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(0, tex_live_vmap_c1);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_live_nmap_c1);
	glTextureStorage2D(tex_live_nmap_c1, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_live_nmap_c1, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(1, tex_live_nmap_c1);

	glCreateFramebuffers(1, &fbo_rigid_motion_c1);
	// attach textures to framebuffer
	glNamedFramebufferTexture(fbo_rigid_motion_c1, GL_COLOR_ATTACHMENT0, tex_live_vmap_c1, 0);
	glNamedFramebufferTexture(fbo_rigid_motion_c1, GL_COLOR_ATTACHMENT1, tex_live_nmap_c1, 0);
	glNamedFramebufferTexture(fbo_rigid_motion_c1, GL_DEPTH_ATTACHMENT, tex_zbuffer, 0);

	// check completeness of fbo_rigid_icp
	if (glCheckNamedFramebufferStatus(fbo_rigid_motion_c1, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw std::runtime_error("renderer error: fbo_rigid_icp_c1 is not complete.");
	}
	// set draw buffers
	glNamedFramebufferDrawBuffers(fbo_rigid_motion_c1, 2, draw_buffers);

	/****************     for nonrigid motion estimation     ********************/
	///--- Create vertex array object
	if (!vao_obj_texture_nonrigid.isCreated()) {
		bool success = vao_obj_texture_nonrigid.create();
		assert(success);
		vao_obj_texture_nonrigid.bind();
	}

	///--- Load/compile shaders
	if (!program_obj_texture_nonrigid.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/ObjectModelTextureRendererNonRigid_vshader.glsl";
		const char* fshader = ":/KinectDataRenderer/ObjectModelTextureRendererNonRigid_fshader.glsl";
		bool vok = program_obj_texture_nonrigid.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program_obj_texture_nonrigid.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program_obj_texture_nonrigid.link();
		assert(lok && vok && fok);
		bool success = program_obj_texture_nonrigid.bind();
		assert(success);
	}

	///--- Create vertex buffer/attributes "position"
	{
		/*********************************************************************/
		/*        use the opengl API other than the Qt gl api                */

		//use the opengl API
		glBindBuffer(GL_ARRAY_BUFFER, can_vertex_buffer);
		program_obj_texture_nonrigid.setAttributeBuffer("can_point_v", GL_FLOAT, 0, 4);
		program_obj_texture_nonrigid.enableAttributeArray("can_point_v");

		glBindBuffer(GL_ARRAY_BUFFER, can_normal_buffer);
		program_obj_texture_nonrigid.setAttributeBuffer("can_point_n", GL_FLOAT, 0, 4);
		program_obj_texture_nonrigid.enableAttributeArray("can_point_n");

		glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
		program_obj_texture_nonrigid.setAttributeBuffer("warp_point_v", GL_FLOAT, 0, 4);
		program_obj_texture_nonrigid.enableAttributeArray("warp_point_v");

	}

	///--- Avoid pollution
	program_obj_texture_nonrigid.release();
	vao_obj_texture_nonrigid.release();

	//////////////////////////////////////////////////////////////////////
	// create texture attachments of view0 for non-rigid motion estimation
	glCreateTextures(GL_TEXTURE_2D, 1, &tex_can_vmap_c0);
	glTextureStorage2D(tex_can_vmap_c0, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_can_vmap_c0, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(0, tex_can_vmap_c0);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_can_nmap_c0);
	glTextureStorage2D(tex_can_nmap_c0, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_can_nmap_c0, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(1, tex_can_nmap_c0);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_can_zbuffer);
	glTextureStorage2D(tex_can_zbuffer, 1, GL_DEPTH_COMPONENT32F, 320, 240);

	glCreateFramebuffers(1, &fbo_nonrigid_motion_c0);
	// attach textures to framebuffer
	glNamedFramebufferTexture(fbo_nonrigid_motion_c0, GL_COLOR_ATTACHMENT0, tex_can_vmap_c0, 0);
	glNamedFramebufferTexture(fbo_nonrigid_motion_c0, GL_COLOR_ATTACHMENT1, tex_can_nmap_c0, 0);
	glNamedFramebufferTexture(fbo_nonrigid_motion_c0, GL_DEPTH_ATTACHMENT, tex_can_zbuffer, 0);

	// check completeness of fbo_rigid_icp
	if (glCheckNamedFramebufferStatus(fbo_nonrigid_motion_c0, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw std::runtime_error("renderer error: fbo_nonrigid_icp_c0 is not complete.");
	}

	// set draw buffers
	glNamedFramebufferDrawBuffers(fbo_nonrigid_motion_c0, 2, draw_buffers);


	//////////////////////////////////////////////////////////////////////
	// create texture attachments of view1 for non-rigid motion estimation
	glCreateTextures(GL_TEXTURE_2D, 1, &tex_can_vmap_c1);
	glTextureStorage2D(tex_can_vmap_c1, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_can_vmap_c1, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(0, tex_can_vmap_c1);

	glCreateTextures(GL_TEXTURE_2D, 1, &tex_can_nmap_c1);
	glTextureStorage2D(tex_can_nmap_c1, 1, GL_RGBA32F, 320, 240);
	glTextureParameteri(tex_can_nmap_c1, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glBindTextureUnit(1, tex_can_nmap_c1);

	glCreateFramebuffers(1, &fbo_nonrigid_motion_c1);
	// attach textures to framebuffer
	glNamedFramebufferTexture(fbo_nonrigid_motion_c1, GL_COLOR_ATTACHMENT0, tex_can_vmap_c1, 0);
	glNamedFramebufferTexture(fbo_nonrigid_motion_c1, GL_COLOR_ATTACHMENT1, tex_can_nmap_c1, 0);
	glNamedFramebufferTexture(fbo_nonrigid_motion_c1, GL_DEPTH_ATTACHMENT, tex_can_zbuffer, 0);

	// check completeness of fbo_rigid_icp
	if (glCheckNamedFramebufferStatus(fbo_nonrigid_motion_c1, GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		throw std::runtime_error("renderer error: fbo_nonrigid_c1 is not complete.");
	}

	// set draw buffers
	glNamedFramebufferDrawBuffers(fbo_nonrigid_motion_c1, 2, draw_buffers);

	/*************************************************************************/
	/*           initialize gl vao and program to show skeleton              */
	/*************************************************************************/
	///--- Create vertex array object
	if (!vao_line.isCreated()) {
		bool success = vao_line.create();
		assert(success);
		vao_line.bind();
	}

	///--- Load/compile shaders
	if (!program_line.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/SkeletonRenderer_vshader.glsl";
		const char* fshader = ":/KinectDataRenderer/SkeletonRenderer_fshader.glsl";
		bool vok = program_line.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program_line.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program_line.link();
		assert(lok && vok && fok);
		bool success = program_line.bind();
		assert(success);
	}

	///--- Create vertex buffer/attributes "position"
	{
		bool success = line_vertex_buffer.create();
		assert(success);
		line_vertex_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success = line_vertex_buffer.bind();
		assert(success);
		line_vertex_buffer.allocate(40 * 3 * sizeof(GLfloat));
		program_line.setAttributeBuffer("vpoint", GL_FLOAT, 0, 3);
		program_line.enableAttributeArray("vpoint");
	}

	///--- Create vertex buffer/attributes "position"
	{
		bool success = line_color_buffer.create();
		assert(success);
		line_color_buffer.setUsagePattern(QGLBuffer::DynamicDraw);
		success = line_color_buffer.bind();
		assert(success);
		line_color_buffer.allocate(40 * 3 * sizeof(GLfloat));
		program_line.setAttributeBuffer("vcolor", GL_FLOAT, 0, 3);
		program_line.enableAttributeArray("vcolor");
	}

	///--- Avoid pollution
	program_line.release();
	vao_line.release();

	/*************************************************************************/
	/*           initialize gl vao and program to show image                 */
	/*************************************************************************/
	if (!vao_img.isCreated()) {
		bool success = vao_img.create();
		assert(success);
		vao_img.bind();
	}

	///--- Load/compile shaders
	if (!program_img.isLinked()) {
		const char* vshader = ":/KinectDataRenderer/ImageRender_vshader.glsl";
		const char* fshader = ":/KinectDataRenderer/ImageRender_fshader.glsl";
		bool vok = program_img.addShaderFromSourceFile(QGLShader::Vertex, vshader);
		bool fok = program_img.addShaderFromSourceFile(QGLShader::Fragment, fshader);
		bool lok = program_img.link();
		assert(lok && vok && fok);
		bool success = program_img.bind();
		assert(success);
	}

	static const float img_cord_data[] =
	{
		-1.0f, -1.0f,
		1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f
	};

	static const float tex_cord_data[] =
	{
		0.0f, 1.0f,
		1.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	{
		bool success = img_coord_buffer.create();
		assert(success);
		img_coord_buffer.setUsagePattern(QGLBuffer::StaticDraw);
		success = img_coord_buffer.bind();
		assert(success);
		img_coord_buffer.allocate(8 * sizeof(float));
		program_img.setAttributeBuffer("in_position", GL_FLOAT, 0, 2);
		program_img.enableAttributeArray("in_position");

		img_coord_buffer.write(0, img_cord_data, 8 * sizeof(float));
	}

	///--- Create vertex buffer/attributes "position"
	{
		bool success = img_tex_coord_buffer.create();
		assert(success);
		img_tex_coord_buffer.setUsagePattern(QGLBuffer::StaticDraw);
		success = img_tex_coord_buffer.bind();
		assert(success);
		img_tex_coord_buffer.allocate(8 * sizeof(float));
		program_img.setAttributeBuffer("in_tex_coord", GL_FLOAT, 0, 2);
		program_img.enableAttributeArray("in_tex_coord");

		img_tex_coord_buffer.write(0, tex_cord_data, 8 * sizeof(float));
	}

	
	glGenTextures(1, &texture_img);
	glBindTexture(GL_TEXTURE_2D, texture_img);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 320, 240, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, NULL);

	vao_img.release();
	program_img.release();
}

void KinectDataRenderer::setup(GLuint texture_id_color, GLuint texture_id_depth) {
	this->texture_id_depth = texture_id_depth;
	program.bind();
	glUniform1i(glGetUniformLocation(program.programId(), "tex_depth"), 1 /*GL_TEXTURE_1*/);
	program.release();
}

void KinectDataRenderer::render() {

	if (texture_id_depth == 0) return;

	if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}

	vao.bind();
	program.bind();

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, texture_id_depth);
	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_POINTS, 0, num_vertices);
	glDisable(GL_PROGRAM_POINT_SIZE);

	program.release();
	vao.release();
}

void KinectDataRenderer::render_point(std::vector<float3>& points_array, Eigen::Vector3f color_value) {


	set_uniform_program(program, "given_color", color_value);

	if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}

	num_vertices = points_array.size();

	vao.bind();
	program.bind();

	point_buffer.bind();
	point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_POINTS, 0, num_vertices);
	glDisable(GL_PROGRAM_POINT_SIZE);

	point_buffer.release();

	program.release();
	vao.release();
}

void KinectDataRenderer::render_object_nodes(std::vector<float3>& points_array, Eigen::Vector3f color_value) {

	set_uniform_program(program, "given_color", color_value);

	if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}

	num_vertices = points_array.size();

	vao.bind();
	program.bind();

	point_buffer.bind();
	point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

//	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointSize(8);//50
	glDrawArrays(GL_POINTS, 0, num_vertices);
//	glDisable(GL_PROGRAM_POINT_SIZE);

	point_buffer.release();

	program.release();
	vao.release();
}

void KinectDataRenderer::render_object_index_nodes(std::vector<float3>& points_array, std::vector<int>& node_idx, Eigen::Vector3f color_value) {

	if (node_idx.size() > 0)
	{

		set_uniform_program(program, "given_color", color_value);

		if (alpha < 1.0) {
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}
		else {
			glDisable(GL_BLEND);
		}

		std::vector<float3> draw_point_array;
		draw_point_array.clear();

		for (int i = 0; i < node_idx.size(); i++)
			draw_point_array.push_back(points_array[node_idx[i]]);

		num_vertices = draw_point_array.size();

		vao.bind();
		program.bind();

		point_buffer.bind();
		point_buffer.write(0, draw_point_array.data(), num_vertices * 3 * sizeof(float));

		//	glEnable(GL_PROGRAM_POINT_SIZE);
		glPointSize(8);
		glDrawArrays(GL_POINTS, 0, num_vertices);
		//	glDisable(GL_PROGRAM_POINT_SIZE);

		point_buffer.release();

		program.release();
		vao.release();
	}
}

void KinectDataRenderer::render_interaction_correspondences(std::vector<float3>& interaction_vertex, std::vector<unsigned char>& finger_idx) {

	if (interaction_vertex.size() > 0)
	{
		for (int i = 0; i < 15; i++)
		{
			std::vector<float3> points_array;
			points_array.clear();

			for (int j = 0; j < interaction_vertex.size(); j++)
			{
				if (finger_idx[j] == i)
					points_array.push_back(interaction_vertex[j]);
			}

			Eigen::Vector3f color_value = Eigen::Vector3f(1, 0, 0);
			switch (i)
			{
			case 0:
			case 5:
				color_value = Eigen::Vector3f(1, 0, 0);
				break;
			case 1:
			case 6:
				color_value = Eigen::Vector3f(1, 1, 0);
				break;
			case 2:
			case 7:
				color_value = Eigen::Vector3f(0, 1, 0);
				break;
			case 3:
			case 8:
				color_value = Eigen::Vector3f(0, 1, 1);
				break;
			case 4:
			case 9:
				color_value = Eigen::Vector3f(0, 0, 1);
				break;
			default:
				color_value = Eigen::Vector3f(0, 0, 1);
				break;
			}

			set_uniform_program(program, "given_color", color_value);

			num_vertices = points_array.size();

			vao.bind();
			program.bind();

			point_buffer.bind();
			point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

			//	glEnable(GL_PROGRAM_POINT_SIZE);
			glPointSize(8);
			glDrawArrays(GL_POINTS, 0, num_vertices);
			//	glDisable(GL_PROGRAM_POINT_SIZE);

			point_buffer.release();

			program.release();
			vao.release();
		}
	}
}

void KinectDataRenderer::render_object_nodes2tips(std::vector<float3>& nodes_array, std::vector<std::vector<int>>& nodes_tip, std::vector<float>& variant_smooth) {

	if (nodes_array.size() > 0)
	{
//		int i = 0;
		//for (int i = 0; i < nodes_tip.size(); i++)
		//{
		//	std::vector<int> node_idx = nodes_tip[i];
		//	std::vector<float3> points_array;
		//	points_array.clear();
		//	for (int j = 0; j < node_idx.size(); j++)
		//	{
		//		points_array.push_back(nodes_array[node_idx[j]]);
		//	}

		//	Eigen::Vector3f color_value;
		//	switch (i)
		//	{
		//	case 0:
		//		color_value = Eigen::Vector3f(1, 0, 0);
		//		break;
		//	case 1:
		//		color_value = Eigen::Vector3f(1, 1, 0);
		//		break;
		//	case 2:
		//		color_value = Eigen::Vector3f(0, 1, 0);
		//		break;
		//	case 3:
		//		color_value = Eigen::Vector3f(0, 1, 1);
		//		break;
		//	case 4:
		//		color_value = Eigen::Vector3f(0, 0, 1);
		//		break;
		//	default:
		//		color_value = Eigen::Vector3f(0, 0, 1);
		//		break;
		//	}

		//	set_uniform_program(program, "given_color", color_value);

		//	/*if (alpha < 1.0) {
		//		glEnable(GL_BLEND);
		//		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		//	}
		//	else {
		//		glDisable(GL_BLEND);
		//	}*/

		//	num_vertices = points_array.size();

		//	vao.bind();
		//	program.bind();

		//	point_buffer.bind();
		//	point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

		//	//	glEnable(GL_PROGRAM_POINT_SIZE);
		//	glPointSize(8);
		//	glDrawArrays(GL_POINTS, 0, num_vertices);
		//	//	glDisable(GL_PROGRAM_POINT_SIZE);

		//	point_buffer.release();
		//}

		for (int i = 0; i < nodes_tip.size(); i++)
		{
			std::vector<int> node_idx = nodes_tip[i];
			for (int j = 0; j < node_idx.size(); j++)
			{
				std::vector<float3> points_array;
				points_array.clear();

				points_array.push_back(nodes_array[node_idx[j]]);

				float smooth_term = variant_smooth[node_idx[j]];
				
				float factor = 1- smooth_term / 100.0;

				Eigen::Vector3f color_value;
				switch (i)
				{
				case 0:
					color_value = Eigen::Vector3f(1, 0, 0)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				case 1:
					color_value = Eigen::Vector3f(1, 1, 0)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				case 2:
					color_value = Eigen::Vector3f(0, 1, 0)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				case 3:
					color_value = Eigen::Vector3f(0, 1, 1)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				case 4:
					color_value = Eigen::Vector3f(0, 0, 1)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				default:
					color_value = Eigen::Vector3f(0, 0, 1)*factor + Eigen::Vector3f(1, 1, 1)*(1 - factor);
					break;
				}

				set_uniform_program(program, "given_color", color_value);

				num_vertices = points_array.size();

				vao.bind();
				program.bind();

				point_buffer.bind();
				point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

				//	glEnable(GL_PROGRAM_POINT_SIZE);
				glPointSize(8);
				glDrawArrays(GL_POINTS, 0, num_vertices);
				//	glDisable(GL_PROGRAM_POINT_SIZE);

				point_buffer.release();
			}
		}

		program.release();
		vao.release();
	}
}

void KinectDataRenderer::render_object_model(std::vector<float4>& points_v, std::vector<float4>& points_n, int vertex_number) {

	/*if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}*/

	num_vertices = vertex_number;
	//num_vertices = points_v.size();
	//glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_v.data());

	//glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_n.data());

//	glBindBuffer(GL_ARRAY_BUFFER, 0);

	vao_obj.bind();
	program_obj.bind();

	//live_vertex_buffer.bind();
	//live_vertex_buffer.write(0, points_v.data(), num_vertices * 4 * sizeof(float));

	//live_normal_buffer.bind();
	//live_normal_buffer.write(0, points_n.data(), num_vertices * 4 * sizeof(float));

//	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_TRIANGLES, 0, num_vertices); //GL_TRIANGLES
//	glDisable(GL_PROGRAM_POINT_SIZE);

	//live_normal_buffer.release();
	//live_vertex_buffer.release();

	program_obj.release();
	vao_obj.release();
}

void KinectDataRenderer::render_canonical_model(std::vector<float4>& points_v, std::vector<float4>& points_n, int vertex_number) {

	/*if (alpha < 1.0) {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
	glDisable(GL_BLEND);
	}*/

	num_vertices = vertex_number;
	//num_vertices = points_v.size();
	//glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_v.data());

	//glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_n.data());

	//	glBindBuffer(GL_ARRAY_BUFFER, 0);

	vao_obj_texture_nonrigid.bind();
	program_obj.bind();

	//live_vertex_buffer.bind();
	//live_vertex_buffer.write(0, points_v.data(), num_vertices * 4 * sizeof(float));

	//live_normal_buffer.bind();
	//live_normal_buffer.write(0, points_n.data(), num_vertices * 4 * sizeof(float));

	//	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_TRIANGLES, 0, num_vertices);
	//	glDisable(GL_PROGRAM_POINT_SIZE);

	//live_normal_buffer.release();
	//live_vertex_buffer.release();

	program_obj.release();
	vao_obj_texture_nonrigid.release();
}

void KinectDataRenderer::render_object_model_texture(int vertex_number) {

	/*if (alpha < 1.0) {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
	glDisable(GL_BLEND);
	}*/

	num_vertices = vertex_number;
	//num_vertices = points_v.size();
	//glBindBuffer(GL_ARRAY_BUFFER, warp_vertex_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_v.data());

	//glBindBuffer(GL_ARRAY_BUFFER, warp_normal_buffer);
	//glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(float), points_n.data());


	vao_obj_texture.bind();
	program_obj_texture.bind();

	//can_vertex_buffer.bind();
	//can_vertex_buffer.write(0, points_v.data(), num_vertices * 4 * sizeof(float));

	//can_normal_buffer.bind();
	//can_normal_buffer.write(0, points_n.data(), num_vertices * 4 * sizeof(float));

//	glClearColor(1, 1, 1, 1);
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_TRIANGLES, 0, num_vertices);
	//	glDisable(GL_PROGRAM_POINT_SIZE);

	//can_normal_buffer.release();
	//can_vertex_buffer.release();

	program_obj_texture.release();
	vao_obj_texture.release();
}

void KinectDataRenderer::render_object_model_texture_nonrigid(int vertex_number) 
{
	num_vertices = vertex_number;

	vao_obj_texture_nonrigid.bind();
	program_obj_texture_nonrigid.bind();

	glDrawArrays(GL_TRIANGLES, 0, num_vertices);

	program_obj_texture_nonrigid.release();
	vao_obj_texture_nonrigid.release();
}

void KinectDataRenderer::render3(std::vector<float3>& points_array, Eigen::Vector3f color_value) {

	if (alpha < 1.0) {
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}
	else {
		glDisable(GL_BLEND);
	}

	num_vertices = points_array.size();

	vao.bind();
	program.bind();

	point_buffer.bind();
	point_buffer.write(0, points_array.data(), num_vertices * 3 * sizeof(float));

	glEnable(GL_PROGRAM_POINT_SIZE);
	glDrawArrays(GL_POINTS, 0, num_vertices);
	glDisable(GL_PROGRAM_POINT_SIZE);

	point_buffer.release();

	program.release();
	vao.release();
}

void KinectDataRenderer::render_img(cv::Mat &img)
{
	vao_img.bind();
	program_img.bind();

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_2D, texture_img);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, img.data);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	program_img.release();
	vao_img.release();
}

void KinectDataRenderer::render_line(std::vector<float3>& line_v, std::vector<float3>& line_c)
{
	int num_vertices = line_v.size();

	vao_line.bind();
	program_line.bind();

	line_vertex_buffer.bind();
	line_vertex_buffer.write(0, line_v.data(), num_vertices * 3 * sizeof(float));

	line_color_buffer.bind();
	line_color_buffer.write(0, line_c.data(), num_vertices * 3 * sizeof(float));

	glEnable(GL_LINE_SMOOTH);
	glLineWidth(8.0f);
	glDrawArrays(GL_LINES, 0, num_vertices);

	line_vertex_buffer.release();
	line_color_buffer.release();

	program_line.release();
	vao_line.release();
}

void KinectDataRenderer::set_discard_cosalpha_th(float val) {
	set_uniform("discard_cosalpha_th", val);
}

void KinectDataRenderer::set_alpha(float alpha) {
	set_uniform("alpha", alpha);
	this->alpha = alpha;
}
void KinectDataRenderer::set_zNear(float alpha) {
	set_uniform("zNear", alpha);
}
void KinectDataRenderer::set_zFar(float alpha) {
	set_uniform("zFar", alpha);
}
void KinectDataRenderer::enable_colormap(bool enable) {
	set_uniform("enable_colormap", (enable) ? +1.0f : -1.0f);
}

void KinectDataRenderer::set_point_color(const Eigen::Vector3f& given_color)
{
//	set_uniform("given_color", given_color);

	set_uniform_program(program, "given_color", given_color);
}

void KinectDataRenderer::set_projection_point(const Eigen::Matrix4f& value)
{
	set_uniform_program(program, "view_projection", value);
}

void KinectDataRenderer::set_view_projection_object(const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& rigid_motion)
{
	set_uniform_program(program_obj, "view_projection", proj_mat);
	set_uniform_program(program_obj, "view_matrix", view_mat);
	set_uniform_program(program_obj, "rigid_motion", rigid_motion);
}

void KinectDataRenderer::set_view_projection_texture(const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& rigid_motion)
{
	set_uniform_program(program_obj_texture, "camera_projection", proj_mat);
	set_uniform_program(program_obj_texture, "camera_view", view_mat);
	set_uniform_program(program_obj_texture, "object_motion", rigid_motion);
}

void KinectDataRenderer::set_view_projection_texture_nonrigid(const Eigen::Matrix4f& proj_mat, const Eigen::Matrix4f& view_mat, const Eigen::Matrix4f& rigid_motion)
{
	set_uniform_program(program_obj_texture_nonrigid, "camera_projection", proj_mat);
	set_uniform_program(program_obj_texture_nonrigid, "camera_view", view_mat);
	set_uniform_program(program_obj_texture_nonrigid, "object_motion", rigid_motion);
}

void KinectDataRenderer::set_projection_line(const Eigen::Matrix4f& value)
{
	set_uniform_program(program_line, "view_projection", value);
}

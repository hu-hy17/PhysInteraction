#include "MarkerRenderer.h"
#include "physhand/Utils.h"

#include<iostream>

using std::cout;
using std::cerr;
using std::endl;

namespace san {
	const char* ATTR_POSITION = "aPos";
	const char* ATTR_NORMAL = "aNormal";
	const char* ATTR_TEXTURE = "aTex";
	const char* ATTR_OPEN_LIGHT = "openLight";
	const char* ATTR_LIGHT_POS = "lightPos";
	const char* ATTR_DIFFUSE_STR = "diffuseStrength";
	const char* ATTR_SPECULAR_STR = "specularStrength";
	const char* ATTR_AMBIENT_STR = "ambientStrength";
	const char* ATTR_LIGHT_COLOR = "lightColor";
	const char* ATTR_VIEW_POS = "viewPos";
	const char* ATTR_OBJ_COLOR = "objectColor";
	const char* ATTR_OBJ_COLOR_ALPHA = "colorAlpha";
	const char* ATTR_VIEW_PROJ_MAT = "view_projection";
	const char* ATTR_LOC_TO_GLOB_MAT = "localToGlobal";
	const char* ATTR_CENTER_POS = "centerPos";
	const char* ATTR_RADIUS = "radius";
	const char* ATTR_LENGTH = "length";
	const char* ATTR_LENGTH_X = "lenX";
	const char* ATTR_LENGTH_Y = "lenY";
	const char* ATTR_LENGTH_Z = "lenZ";
	const char* ATTR_Z_RATIO = "zRatio";
	const char* ATTR_USE_TEX = "useTex";
	const char* ATTR_TEX_0 = "texture0";
}

/************************************************************************/
/* GLObject                                                             */
/************************************************************************/
GLObject::GLObject()
{
	m_vbo = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_ebo = QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
	this->initializeOpenGLFunctions();
}

GLObject::~GLObject()
{ }

void GLObject::destroy()
{
	m_vbo.destroy();
	m_ebo.destroy();
	m_vao.destroy();
}

void GLObject::setColor(QVector4D& color)
{
	m_color = color;
}

void GLObject::setLocToGlob(const QMatrix4x4& mat)
{
	m_local_to_global = mat;
}

void GLObject::setProjMat(const QMatrix4x4& mat)
{
	m_projection = mat;
}

void GLObject::setProjMat(const Eigen::Matrix4f& mat)
{
	for (int r = 0; r < 4; ++r)
	{
		for (int c = 0; c < 4; ++c)
			m_projection(r, c) = mat(r, c);
	}
}

void GLObject::setViewMat(const QMatrix4x4& mat, const QVector3D& center)
{
	m_view = mat;
	m_camera_pos = center;
}

void GLObject::setViewMat(const Eigen::Matrix4f& mat, const Eigen::Vector3f& center)
{
	for (int r = 0; r < 4; ++r)
	{
		for (int c = 0; c < 4; ++c)
			m_view(r, c) = mat(r, c);
	}

	m_camera_pos.setX(center.x());
	m_camera_pos.setY(center.y());
	m_camera_pos.setZ(center.z());
}

void GLObject::setZRatio(const float zRatio)
{
	m_z_ratio = zRatio;
}

void GLObject::_compileAndLinkShader_(
	const std::string& class_name,
	const std::string& vert_shader,
	const std::string& frag_shader)
{
	// 着色器编译和链接
	bool success = m_shader.addShaderFromSourceFile(QOpenGLShader::Vertex, vert_shader.c_str());
	if (!success)
	{
		cerr << class_name << "Error: ShaderProgram addShaderFromSourceFile failed!" << endl;
		goto fail;
	}

	success = m_shader.addShaderFromSourceFile(QOpenGLShader::Fragment, frag_shader.c_str());
	if (!success)
	{
		cerr << class_name << "Error: ShaderProgram addShaderFromSourceFile failed!" << endl;
		goto fail;
	}

	success = m_shader.link();
	if (!success)
	{
		cerr << class_name << "Error: shaderProgram link failed!" << endl;
		goto fail;
	}

	return;

fail:
	system("pause");
	exit(-1);
}

void GLObject::_setLightCond_()
{
	// 设置是否开启光照,几何体颜色,观察者位置
	m_shader.setUniformValue(san::ATTR_OPEN_LIGHT, m_open_light);
	m_shader.setUniformValue(san::ATTR_OBJ_COLOR, QVector3D(m_color));
	m_shader.setUniformValue(san::ATTR_OBJ_COLOR_ALPHA, m_color[3]);

	m_shader.setUniformValue(san::ATTR_VIEW_POS, m_camera_pos);

	if (!m_open_light)
		return;

	// 光源位置
	m_shader.setUniformValue(san::ATTR_LIGHT_POS, m_camera_pos);

	// 漫反射
	m_shader.setUniformValue(san::ATTR_DIFFUSE_STR, 0.9f);

	// 镜面反射
	m_shader.setUniformValue(san::ATTR_SPECULAR_STR, 0.0f);

	// 全局光照强度
	m_shader.setUniformValue(san::ATTR_AMBIENT_STR, 0.1f);

	// 光照颜色
	m_shader.setUniformValue(san::ATTR_LIGHT_COLOR, QVector3D(1.0f, 1.0f, 1.0f));
}

void GLObject::_setUniformVal_()
{
	m_shader.setUniformValue(san::ATTR_Z_RATIO, m_z_ratio);
}

/************************************************************************/
/* GLCylinder                                                           */
/************************************************************************/

GLCylinder::GLCylinder(int steps /* = 100 */)
{
	_compileAndLinkShader_(m_class_name, ":/MarkerRender/cylinder_vshader.glsl", ":/MarkerRender/cylinder_fshader.glsl");

	// 顶点与三角形数据
	QVector<GLfloat> vertices = QVector<GLfloat>();
	QVector<unsigned int> indices = QVector<unsigned int>();

	float radius = 1.0f;
	float length = 1.0f;

	float dTheta = 2 * M_PI / steps;

	// 设置上表面中心和下表面中心
	int bot_center_idx = 0;
	int top_center_idx = 1;
	vertices << 0 << 0 << 0 << 0 << 0 << -1;
	vertices << 0 << 0 << length << 0 << 0 << 1;

	for (float theta = 0; theta < 2 * M_PI; theta += dTheta)
	{
		unsigned int idx_base = vertices.size() / 6;
		// 上下表面三角形顶点
		vertices << radius * cos(theta) << radius * sin(theta) << 0;
		vertices << 0 << 0 << -1;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << 0;
		vertices << 0 << 0 << -1;

		vertices << radius * cos(theta) << radius * sin(theta) << length;
		vertices << 0 << 0 << 1;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << length;
		vertices << 0 << 0 << 1;

		// 侧面三角形顶点
		vertices << radius * cos(theta) << radius * sin(theta) << 0;
		vertices << cos(theta) << sin(theta) << 0;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << 0;
		vertices << cos(theta + dTheta) << sin(theta + dTheta) << 0;

		vertices << radius * cos(theta) << radius * sin(theta) << length;
		vertices << cos(theta) << sin(theta) << 0;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << length;
		vertices << cos(theta + dTheta) << sin(theta + dTheta) << 0;

		// 三角形下标
		indices << 0 << idx_base << idx_base + 1;
		indices << 1 << idx_base + 2 << idx_base + 3;
		indices << idx_base + 4 << idx_base + 5 << idx_base + 6;
		indices << idx_base + 5 << idx_base + 6 << idx_base + 7;
	}

	m_indices_size = indices.size();
	// VBO, EBO
	QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);

	m_vbo.create();
	m_vbo.bind();
	m_vbo.allocate(&(*vertices.begin()), vertices.size() * sizeof(GLfloat));

	m_ebo.create();
	m_ebo.bind();
	m_ebo.allocate(&(*indices.begin()), indices.size() * sizeof(unsigned int));

	// 坐标
	int attr = -1;
	attr = m_shader.attributeLocation(san::ATTR_POSITION);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	// 法向量
	attr = m_shader.attributeLocation(san::ATTR_NORMAL);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, sizeof(GL_FLOAT) * 3, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	m_vbo.release();
}

void GLCylinder::paint()
{
	paint(1.0f, 1.0f, QVector3D(0, 0, 1), QVector3D(0, 0, 0));
}

void GLCylinder::paint(float radius, float length,
					   const QVector3D& direction,
					   const QVector3D& start_pos)
{
	m_shader.bind();
	_setLightCond_();
	_setUniformVal_();

	// 设置起始位置和方向
	m_local_to_global(0, 3) = start_pos.x();
	m_local_to_global(1, 3) = start_pos.y();
	m_local_to_global(2, 3) = start_pos.z();

	Eigen::Matrix3f rot_mat = getRotMat(Eigen::Vector3f(0, 0, 1.0f),
										Eigen::Vector3f(direction.x(), direction.y(), direction.z()));
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
			m_local_to_global(r, c) = rot_mat(r, c);
	}

	// 设置半径和长度
	m_shader.setUniformValue(san::ATTR_RADIUS, radius);
	m_shader.setUniformValue(san::ATTR_LENGTH, length);

	m_shader.setUniformValue(san::ATTR_VIEW_PROJ_MAT, m_projection * m_view);	// 投影+摄像机矩阵
	m_shader.setUniformValue(san::ATTR_LOC_TO_GLOB_MAT, m_local_to_global);		// 局部到全局坐标转换
	{
		QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);
		glDrawElements(GL_TRIANGLES, m_indices_size, GL_UNSIGNED_INT, 0);
	}
	m_shader.release();
}

/************************************************************************/
/* GLCone                                                               */
/************************************************************************/

GLCone::GLCone(int steps /* = 100 */)
{
	_compileAndLinkShader_(m_class_name, ":/MarkerRender/cylinder_vshader.glsl", ":/MarkerRender/cylinder_fshader.glsl");

	// 顶点与三角形数据
	QVector<GLfloat> vertices = QVector<GLfloat>();
	QVector<unsigned int> indices = QVector<unsigned int>();

	float radius = 1.0f;
	float length = 1.0f;

	float dTheta = 2 * M_PI / steps;

	// 设置上表面中心和下表面中心
	int bot_center_idx = 0;
	int top_center_idx = 1;
	vertices << 0 << 0 << 0 << 0 << 0 << -1;
	vertices << 0 << 0 << length << 0 << 0 << 1;

	for (float theta = 0; theta < 2 * M_PI; theta += dTheta)
	{
		unsigned int idx_base = vertices.size() / 6;
		// 下表面三角形顶点
		vertices << radius * cos(theta) << radius * sin(theta) << 0;
		vertices << 0 << 0 << -1;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << 0;
		vertices << 0 << 0 << -1;

		// 侧面三角形顶点
		vertices << radius * cos(theta) << radius * sin(theta) << 0;
		vertices << cos(theta) << sin(theta) << 0;
		vertices << radius * cos(theta + dTheta) << radius * sin(theta + dTheta) << 0;
		vertices << cos(theta + dTheta) << sin(theta + dTheta) << 0;

		// 三角形下标
		indices << 0 << idx_base << idx_base + 1;
		indices << 1 << idx_base + 2 << idx_base + 3;
	}

	m_indices_size = indices.size();
	// VBO, EBO
	QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);

	m_vbo.create();
	m_vbo.bind();
	m_vbo.allocate(&(*vertices.begin()), vertices.size() * sizeof(GLfloat));

	m_ebo.create();
	m_ebo.bind();
	m_ebo.allocate(&(*indices.begin()), indices.size() * sizeof(unsigned int));

	// 坐标
	int attr = -1;
	attr = m_shader.attributeLocation(san::ATTR_POSITION);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	// 法向量
	attr = m_shader.attributeLocation(san::ATTR_NORMAL);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, sizeof(GL_FLOAT) * 3, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	m_vbo.release();
}

void GLCone::paint()
{
	paint(1.0f, 1.0f, QVector3D(0, 0, 1), QVector3D(0, 0, 0));
}

void GLCone::paint(float radius, float length,
				   const QVector3D& direction,
				   const QVector3D& start_pos)
{
	m_shader.bind();
	_setLightCond_();
	_setUniformVal_();

	// 设置起始位置和方向
	m_local_to_global(0, 3) = start_pos.x();
	m_local_to_global(1, 3) = start_pos.y();
	m_local_to_global(2, 3) = start_pos.z();

	Eigen::Matrix3f rot_mat = getRotMat(Eigen::Vector3f(0, 0, 1.0f),
										Eigen::Vector3f(direction.x(), direction.y(), direction.z()));
	for (int r = 0; r < 3; r++)
	{
		for (int c = 0; c < 3; c++)
			m_local_to_global(r, c) = rot_mat(r, c);
	}

	// 设置半径和长度
	m_shader.setUniformValue(san::ATTR_RADIUS, radius);
	m_shader.setUniformValue(san::ATTR_LENGTH, length);

	m_shader.setUniformValue(san::ATTR_VIEW_PROJ_MAT, m_projection * m_view);				// 投影+摄像机矩阵
	m_shader.setUniformValue(san::ATTR_LOC_TO_GLOB_MAT, m_local_to_global);					// 局部到全局坐标转换
	{
		QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);
		glDrawElements(GL_TRIANGLES, m_indices_size, GL_UNSIGNED_INT, 0);
	}
	m_shader.release();
}


/************************************************************************/
/* GLMixObject                                                          */
/************************************************************************/

GLMixObject::GLMixObject()
{
	m_globjects = std::vector<std::unique_ptr<GLObject>>();
}

GLMixObject::~GLMixObject()
{
	for (auto& obj : m_globjects)
		obj->destroy();
}

void GLMixObject::setColor(QVector4D& color)
{
	for (auto& obj : m_globjects)
		obj->setColor(color);
}

void GLMixObject::setLocToGlob(const QMatrix4x4& mat)
{
	for (auto& obj : m_globjects)
		obj->setLocToGlob(mat);
}

void GLMixObject::setProjMat(const QMatrix4x4& mat)
{
	for (auto& obj : m_globjects)
		obj->setProjMat(mat);
}

void GLMixObject::setProjMat(const Eigen::Matrix4f& mat)
{
	for (auto& obj : m_globjects)
		obj->setProjMat(mat);
}

void GLMixObject::setViewMat(const QMatrix4x4& mat, const QVector3D& center)
{
	for (auto& obj : m_globjects)
		obj->setViewMat(mat, center);
}

void GLMixObject::setViewMat(const Eigen::Matrix4f& mat, const Eigen::Vector3f& center)
{
	for (auto& obj : m_globjects)
		obj->setViewMat(mat, center);
}

void GLMixObject::setZRatio(float zRatio)
{
	for (auto& obj : m_globjects)
		obj->setZRatio(zRatio);
}

/************************************************************************/
/* GLArrow                                                              */
/************************************************************************/

GLArrow::GLArrow(int steps /* = 100 */)
{
	m_globjects.push_back(std::make_unique<GLCylinder>(steps));
	m_globjects.push_back(std::make_unique<GLCone>(steps));
}

void GLArrow::paint()
{
	paint(0.1f, 1.0f, QVector3D(0, 0, 1), QVector3D(0, 0, 0));
}

void GLArrow::paint(float radius, float length, QVector3D direction, QVector3D start_pos)
{
	QVector3D cone_start_pos = start_pos + m_len_ratio * length * direction.normalized();
	GLCylinder* p_cylinder = dynamic_cast<GLCylinder*>(m_globjects[0].get());
	GLCone* p_cone = dynamic_cast<GLCone*>(m_globjects[1].get());
	if (p_cone && p_cylinder)
	{
		float cone_len = std::max((1 - m_len_ratio) * length, 10.0f);
		p_cone->paint(radius * m_radius_ratio, cone_len, direction, cone_start_pos);
		p_cylinder->paint(radius, m_len_ratio * length, direction, start_pos);
	}
	else
	{
		cout << m_class_name << "Error: cannot cast smart ptr to common ptr!" << endl;
	}
}

GLMesh::GLMesh(const mesh* m)
{
	_compileAndLinkShader_(m_class_name, ":/MarkerRender/mesh_vshader.glsl", ":/MarkerRender/mesh_fshader.glsl");
	QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);

	QVector<GLfloat> vertices;
	for (int i = 0; i < m->vertex.size(); i++)
	{
		vertices << 1000 * m->vertex[i].x << -1000 * m->vertex[i].y << 1000 * m->vertex[i].z;
		vertices << m->normal[i].x << -m->normal[i].y << m->normal[i].z;
	}

	m_vbo.create();
	m_vbo.bind();
	m_vbo.allocate(&(*vertices.begin()), vertices.size() * sizeof(GLfloat));

	m_ebo.create();
	m_ebo.bind();
	m_ebo.allocate(&(*m->element.begin()), m->element.size() * sizeof(unsigned int));

	m_indices_size = m->element.size();

	// 坐标
	int attr = -1;
	attr = m_shader.attributeLocation(san::ATTR_POSITION);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, 0, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	// 法向量
	attr = m_shader.attributeLocation(san::ATTR_NORMAL);
	m_shader.setAttributeBuffer(attr, GL_FLOAT, sizeof(GL_FLOAT) * 3, 3, sizeof(GLfloat) * 6);
	m_shader.enableAttributeArray(attr);

	m_vbo.release();
}

void GLMesh::paint()
{
	m_shader.bind();
	m_open_light = true;
	_setLightCond_();
	// _setUniformVal_();

	m_local_to_global.setToIdentity();

	m_shader.setUniformValue(san::ATTR_VIEW_PROJ_MAT, m_projection * m_view);				// 投影+摄像机矩阵
	m_shader.setUniformValue(san::ATTR_LOC_TO_GLOB_MAT, m_local_to_global);					// 局部到全局坐标转换
	{
		QOpenGLVertexArrayObject::Binder vaoBind(&m_vao);
		glDrawElements(GL_TRIANGLES, m_indices_size, GL_UNSIGNED_INT, 0);
	}
	m_shader.release();
}
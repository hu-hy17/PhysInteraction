#pragma once

#include<string>
#include<memory>

#include<QOpenGLShader>
#include<QOpenGLShaderProgram>
#include<QOpenGLVertexArrayObject>
#include<QOpenGLBuffer>
#include<QOpenGLFunctions>

#include<QVector>
#include<QMatrix4x4>

#include"tracker/Data/Camera.h"
#include"SphereMesh2Mano/Mesh.h"

namespace san
{
	extern const char* ATTR_POSITION;
	extern const char* ATTR_NORMAL;
	extern const char* ATTR_TEXTURE;
	extern const char* ATTR_OPEN_LIGHT;
	extern const char* ATTR_LIGHT_POS;
	extern const char* ATTR_DIFFUSE_STR;
	extern const char* ATTR_SPECULAR_STR;
	extern const char* ATTR_AMBIENT_STR;
	extern const char* ATTR_LIGHT_COLOR;
	extern const char* ATTR_VIEW_POS;
	extern const char* ATTR_OBJ_COLOR;
	extern const char* ATTR_OBJ_COLOR_ALPHA;
	extern const char* ATTR_VIEW_PROJ_MAT;
	extern const char* ATTR_LOC_TO_GLOB_MAT;
	extern const char* ATTR_CENTER_POS;
	extern const char* ATTR_RADIUS;
	extern const char* ATTR_LENGTH;
	extern const char* ATTR_LENGTH_X;
	extern const char* ATTR_LENGTH_Y;
	extern const char* ATTR_LENGTH_Z;
	extern const char* ATTR_Z_RATIO;
	extern const char* ATTR_USE_TEX;
	extern const char* ATTR_TEX_0;
}

class GLObject : protected QOpenGLFunctions
{
protected:
	const std::string m_class_name = "(GLObject)";
	QVector4D m_color;
	QOpenGLShaderProgram m_shader;
	QOpenGLBuffer m_vbo, m_ebo;
	QOpenGLVertexArrayObject m_vao;
	QMatrix4x4 m_local_to_global;
	QMatrix4x4 m_projection;
	QMatrix4x4 m_view;
	QVector3D m_camera_pos;
	GLfloat m_z_ratio = 1.0f;

	bool m_open_light = false;
	// std::shared_ptr<GLLight> m_light;	// π‚’’

protected:
	void _compileAndLinkShader_(
		const std::string& class_name,
		const std::string& vert_shader,
		const std::string& frag_shader);

	void _setLightCond_();

	void _setUniformVal_();

public:
	GLObject();
	~GLObject();
	virtual void destroy();
	void setColor(QVector4D& color);
	void setLocToGlob(const QMatrix4x4& mat);
	void setProjMat(const QMatrix4x4& mat);
	void setProjMat(const Eigen::Matrix4f& mat);
	void setViewMat(const QMatrix4x4& mat, const QVector3D& center);
	void setViewMat(const Eigen::Matrix4f& mat, const Eigen::Vector3f& center);
	void setZRatio(const float zRatio);
	virtual void paint() = 0;
};

class GLCylinder : public GLObject
{
protected:
	const std::string m_class_name = "(GLCylinder)";
	int m_indices_size;

public:
	GLCylinder(int steps = 100);
	void paint();
	void paint(float radius, float length,
			   const QVector3D& direction,
			   const QVector3D& start_pos);
};

class GLCone : public GLObject
{
protected:
	const std::string m_class_name = "(GLCone)";
	int m_indices_size;
public:
	GLCone(int steps = 100);
	void paint();
	void paint(float radius, float length,
			   const QVector3D& direction,
			   const QVector3D& start_pos);
};

class GLMixObject
{
protected:
	const std::string m_class_name = "(GLMixObject)";
	std::vector<std::unique_ptr<GLObject>> m_globjects;
public:
	GLMixObject();
	~GLMixObject();
	virtual void setColor(QVector4D& color);
	void setLocToGlob(const QMatrix4x4& mat);
	void setProjMat(const QMatrix4x4& mat);
	void setProjMat(const Eigen::Matrix4f& mat);
	void setViewMat(const QMatrix4x4& mat, const QVector3D& center);
	void setViewMat(const Eigen::Matrix4f& mat, const Eigen::Vector3f& center);
	void setZRatio(float zRatio);

	virtual void paint() = 0;
};

class GLArrow : public GLMixObject
{
protected:
	const std::string m_class_name = "(GLArrow)";
private:
	const float m_len_ratio = 0.8f;
	const float m_radius_ratio = 2.0f;
public:
	GLArrow(int steps = 100);
	void paint();
	void paint(float radius, float length, QVector3D direction, QVector3D start_pos);
};

class GLMesh: public GLObject
{
protected:
	const std::string m_class_name = "(GLMesh)";
	int m_indices_size;
public:
	GLMesh(const mesh* m);
	void paint();
};
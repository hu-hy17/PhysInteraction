#pragma once
#include "util/gl_wrapper.h"
#include <Eigen/Dense>
#include <QGLBuffer>
#include <QGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <iostream>

/// Superclass for any renderer
class ObjectRenderer{
protected:
    QGLShaderProgram program;
    QOpenGLVertexArrayObject vao;

	//ZH added: object render
	QGLShaderProgram program_obj;
	QOpenGLVertexArrayObject vao_obj;

	//ZH added: texture render of object for rigid motion estimation
	QGLShaderProgram program_obj_texture;
	QOpenGLVertexArrayObject vao_obj_texture;

	//ZH added: texture render of object for nonrigid motion estimation
	QGLShaderProgram program_obj_texture_nonrigid;
	QOpenGLVertexArrayObject vao_obj_texture_nonrigid;

	//ZH added: image render
	QGLShaderProgram program_img;
	QOpenGLVertexArrayObject vao_img;

	//ZH added: line render
	QGLShaderProgram program_line;
	QOpenGLVertexArrayObject vao_line;

public:
    GLuint program_id(){ return program.programId(); }
    void set_uniform(const char* name, const Eigen::Matrix4f& value);
    void set_uniform(const char* name, const Eigen::Matrix3f& value);
    void set_uniform(const char* name, float value);
	void set_uniform(const char* name, const Eigen::Vector3f& value);
    void set_texture(const char* name, int value /*GL_TEXTURE_?*/);
    void set_uniform_uint(const char* name, unsigned int value);

	void set_uniform_program(QGLShaderProgram& program, const char* name, const Eigen::Matrix4f& value);
	void set_uniform_program(QGLShaderProgram& program, const char* name, const Eigen::Vector3f& value);
	void set_uniform_program(QGLShaderProgram& program, const char* name, float value);

public:
    virtual ~ObjectRenderer(){} ///< safe polymorphic destruct
    virtual void init(){}
    virtual void render()=0;
};


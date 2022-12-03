#version 450 core

in vec3 points;
in vec3 normals;              ///< normal of point in camera coordination

layout(location=0) out vec4 can_point;
layout(location=1) out vec4 can_normal;

///out vec4 live_normal;

void main()
{
	can_point=vec4(points.xyz, 1.0);
	can_normal=vec4(normals.xyz, 0.0);
}


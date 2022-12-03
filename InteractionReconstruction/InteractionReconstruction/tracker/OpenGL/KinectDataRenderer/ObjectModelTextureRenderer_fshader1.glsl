#version 450 core

in vec4 global_point;
in vec3 global_norm;              ///< normal of point in camera coordination

layout(location=0) out vec4 live_point;
layout(location=1) out vec4 live_normal;

///out vec4 live_normal;

void main()
{
	live_point=global_point;
	live_normal=vec4(global_norm.xyz, 0.0);
}


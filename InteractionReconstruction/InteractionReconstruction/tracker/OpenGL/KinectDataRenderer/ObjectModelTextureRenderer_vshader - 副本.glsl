#version 450

///--- 
uniform mat4 camera_projection; ///< for the texture render of object model on one camera 
uniform mat4 camera_view;
uniform mat4 object_motion;

in vec4 point_v;
in vec4 point_n;

out vec3 global_point;
out vec3 global_norm;

void main() {   
    
    ///--- obtain the local position and normal
	vec4 global_point_v4=object_motion*vec4(point_v.xyz,1.0);
	global_point=global_point_v4.xyz;
    global_norm=mat3(object_motion)*point_n.xyz;
	
    gl_Position = camera_projection *camera_view* vec4(global_point_v4.xyz, 1.0);
}
   

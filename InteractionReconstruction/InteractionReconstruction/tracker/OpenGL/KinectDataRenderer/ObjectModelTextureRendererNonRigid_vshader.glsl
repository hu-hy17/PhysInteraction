#version 450

///--- 
uniform mat4 camera_projection; ///< for the texture render of object model on one camera 
uniform mat4 camera_view;
uniform mat4 object_motion;

in vec4 can_point_v;
in vec4 can_point_n;
in vec4 warp_point_v;

out vec3 points;
out vec3 normals;

void main() {   
    
    ///--- obtain the global position and normal of canonical model
	///vec4 global_point_v4=object_motion*vec4(can_point_v.xyz,1.0);
	points=can_point_v.xyz;
    normals=can_point_n.xyz;
	
	vec4 global_warp_point=object_motion*vec4(warp_point_v.xyz,1.0);
    gl_Position = camera_projection*camera_view*vec4(global_warp_point[0],global_warp_point[1],global_warp_point[2], 1.0);
}
   

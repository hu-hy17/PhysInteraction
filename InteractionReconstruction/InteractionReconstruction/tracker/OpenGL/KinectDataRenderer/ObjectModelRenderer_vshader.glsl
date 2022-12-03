#version 450

///---
uniform mat4 view_projection; ///< for visualization 
uniform mat4 view_matrix; ///< for visualization
uniform mat4 rigid_motion;///<this is the rigid motion of camera or object

in vec4 point_v;
in vec4 point_n;

out vec3 local_point;
out vec3 local_norm;

void main() {   
    
    ///--- calculate the local position and normal
	vec4 full_point_v=rigid_motion*vec4( point_v.x, point_v.y, point_v.z, 1.0);
	vec3 full_point_n=mat3( rigid_motion )*vec3( point_n.x, point_n.y, point_n.z);
	
	vec4 local_point_v4=view_matrix*vec4( full_point_v.x, -full_point_v.y, full_point_v.z,1.0);
    local_point =local_point_v4.xyz;
    local_norm=mat3(view_matrix)*vec3( full_point_n.x, -full_point_n.y, full_point_n.z);
	
    gl_Position = view_projection * vec4( full_point_v[0],-full_point_v[1],full_point_v[2], 1.0);
}
   

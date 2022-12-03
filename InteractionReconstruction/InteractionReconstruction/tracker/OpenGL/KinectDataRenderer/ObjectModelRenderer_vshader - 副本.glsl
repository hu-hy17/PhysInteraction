#version 330

///--- 
uniform mat4 view_projection; ///< for visualization 
uniform mat4 view_matrix; ///< for visualization 

in vec3 point_v;
in vec3 point_n;

out vec3 local_point;
out vec3 local_norm;

void main() {   
    
    ///--- calculate the local position and normal
	vec4 local_point_v4=view_matrix*vec4(point_v.xyz,1.0);
	local_point=local_point_v4.xyz;
    local_norm=mat3(view_matrix)*point_n;
	
    gl_Position = view_projection * vec4(point_v[0],point_v[1],point_v[2], 1.0);
}
   

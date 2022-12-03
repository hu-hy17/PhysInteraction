#version 450

uniform mat4 view_projection; ///< for visualization 
in vec3 vpoint;
in vec3 vcolor;

out vec3 vcolor_out;

void main() {   
    
    gl_Position = view_projection * vec4(vpoint[0],vpoint[1],vpoint[2], 1.0);
	vcolor_out=vcolor;
	
}
   

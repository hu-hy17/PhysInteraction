#version 450

in vec3 vcolor_out;

out vec4 color;

void main() {   
    
    color=vec4(vcolor_out, 1.0);
}

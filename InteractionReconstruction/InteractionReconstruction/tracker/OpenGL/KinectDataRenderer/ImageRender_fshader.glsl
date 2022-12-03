#version 450

in vec2 tex_coord;
layout (location = 0) out vec4 color;

uniform sampler2D tex;

void main(void)
{
    color = vec4(texture2D(tex, tex_coord).rgb, 1.0);
	//color = vec4(1.0, 0.0, 0.0, 1.0);
}

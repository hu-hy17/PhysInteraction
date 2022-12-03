#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view_projection;
uniform mat4 localToGlobal;

void main(){
	vec3 bPos = vec3(aPos.x, aPos.y, aPos.z);
	gl_Position = view_projection * localToGlobal * vec4(bPos, 1.0f);
	FragPos = vec3(localToGlobal * vec4(bPos, 1.0f));
	Normal = mat3(localToGlobal) * aNormal;
}
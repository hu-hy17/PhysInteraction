#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform float radius;
uniform float length;
uniform float zRatio;

uniform mat4 view_projection;
uniform mat4 localToGlobal;

void main(){
	vec3 bPos = vec3(aPos.x * radius, aPos.y * radius, aPos.z * length);
	gl_Position = view_projection * localToGlobal * vec4(bPos, 1.0f);
	gl_Position.z *= zRatio;
	FragPos = vec3(localToGlobal * vec4(bPos, 1.0f));
	Normal = mat3(localToGlobal) * aNormal;
}
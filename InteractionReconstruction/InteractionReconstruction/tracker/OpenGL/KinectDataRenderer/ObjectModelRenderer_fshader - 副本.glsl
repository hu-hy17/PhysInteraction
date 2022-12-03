#version 330 core

///light attribution
uniform vec3 la;
uniform vec3 ld;
uniform vec3 ls;
uniform vec3 ldir;

///front material attribution
uniform vec3 f_ma;
uniform vec3 f_md;
uniform vec3 f_ms;
uniform float f_ss; 

///back material attribution
//uniform vec3 b_ma;
//uniform vec3 b_md;
//uniform vec3 b_ms;
//uniform vec3 b_ss;

//uniform float alpha;         ///< alpha blending value
in vec3 local_point;
in vec3 local_norm;              ///< normal of point in camera coordination
out vec4 color;

void main() {

	///calculate the direction of reflected light 
    vec3 ldir_ref=normalize(reflect(ldir, local_norm));
	///calculate the direction of eye
	vec3 eye_dir=normalize(-local_point);
	
	///base light
	vec3 ka=la*f_ma;
	
	///diffuse reflection
	vec3 kd=ld*f_md*max(dot(-ldir,local_norm),0.0);
	
	///specular reflection
	vec3 ks=ls*f_ms*pow(max(dot(ldir_ref, eye_dir),0.0), f_ss);
	
	//composited color
	color=vec4(clamp(ka+kd+ks, 0.0, 1.0), 1.0);
}


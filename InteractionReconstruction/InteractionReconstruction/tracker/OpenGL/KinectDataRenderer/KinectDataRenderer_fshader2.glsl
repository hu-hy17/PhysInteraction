#version 330 core
///--- Stuff to colormap the depth mesh
uniform sampler1D colormap;
uniform float zNear;
uniform float zFar;

///--- Stuff to discard not front-facing

uniform float alpha;         ///< alpha blending value
uniform vec3 given_color;
in float depth;              ///< interpolated depth value
in float discardme;          ///< used to discard grazing angle triangles
out vec4 color;

void main() {

    if(discardme>0) discard;
    
    //--- Discard out of z-range
    if(depth<zNear || depth>zFar) discard;
            
	///--- Apply a colormap
	float range = zFar-zNear;
	float w = (depth-zNear)/range;
	///color = vec4( texture(colormap, w).rgb, alpha);
	color=vec4(given_color, alpha);
		
	vec2 pc = gl_PointCoord - vec2(0.5); 
    float distance_to_center = 0.5 - length(pc); 
    if (distance_to_center < 0.0) discard;
}


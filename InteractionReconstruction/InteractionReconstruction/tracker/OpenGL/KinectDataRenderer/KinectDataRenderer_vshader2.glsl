#version 330
/// texture with depth from sensor
uniform float zNear;
uniform float zFar;
///--- 
uniform mat4 view_projection; ///< for visualization 
in vec3 vpoint;

out float depth;
out float discardme;

void main() {   
    
    ///--- Depth evaluation
    depth = vpoint[2];//float( texture(tex_depth, uv).x );
    discardme = float(depth<zNear || depth>zFar);
    gl_Position = view_projection * vec4(vpoint[0],vpoint[1],vpoint[2], 1.0);
    
    ///--- Splat size
    gl_PointSize = 2.9; ///< Screen
    //gl_PointSize = 4; ///< RETINA
}
   

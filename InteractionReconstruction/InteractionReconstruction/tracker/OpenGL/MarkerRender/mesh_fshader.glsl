#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  

uniform bool openLight;

uniform float ambientStrength;
uniform float diffuseStrength;
uniform float specularStrength;

uniform vec3 viewPos;
uniform vec3 lightPos; 
uniform vec3 lightColor;

uniform vec3 objectColor;
uniform float colorAlpha;

void main()
{
	if(!openLight)
	{
		FragColor = vec4(objectColor, colorAlpha);
	}
	else
	{
		// ambient
		vec3 ambient = vec3(0.0f, 0.0f, 0.0f);
		ambient = ambientStrength * lightColor;

    
		// diffuse 
		vec3 diffuse = vec3(0.0f, 0.0f, 0.0f);
		vec3 norm = normalize(Normal);
		vec3 lightDir = normalize(lightPos - FragPos);
		float diff = max(dot(norm, lightDir), 0.0);
		diffuse = diffuseStrength * diff * lightColor;

		// specular
		vec3 viewDir = normalize(viewPos - FragPos);
		vec3 reflectDir = reflect(-lightDir, norm);  
		float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
		vec3 specular = specularStrength * spec * lightColor;  
        
		vec3 result = (ambient + diffuse + specular) * objectColor;
		FragColor = vec4(result, colorAlpha);
		
		// debug
		// FragColor = vec4(lightDir, 1.0f);
	}
}
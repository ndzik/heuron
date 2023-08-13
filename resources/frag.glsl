#version 330 core

out vec4 FragColor;
in vec3 ourColor;
uniform float bias;

void main()
{
    FragColor = vec4(bias * ourColor,1.0);
}

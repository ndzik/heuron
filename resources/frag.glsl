#version 330 core

out vec4 FragColor;
in vec3 ourColor;
uniform float bias;

void main()
{
    FragColor = vec4(bias * ourColor.r, bias + ourColor.g, bias * ourColor.b ,1.0);
}

"""
OpenGL rendering and shader setup
"""

import ctypes
import numpy as np
from .constants import PYOPENXR_AVAILABLE

if PYOPENXR_AVAILABLE:
    from OpenGL import GL


def create_stereo_shaders():
    """Create OpenGL shaders for rendering stereo content"""
    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec2 texCoord;

    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;

    out vec2 TexCoord;

    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
        TexCoord = texCoord;
    }
    """

    fragment_shader_source = """
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;

    uniform sampler2D texture1;
    uniform int stereoFormat;
    uniform int eyeIndex;
    uniform bool swapEyes;

    void main() {
        vec2 uv = TexCoord;

        // Adjust UV based on stereo format
        if (stereoFormat == 0) {  // Side-by-Side
            uv.x = uv.x * 0.5;
            if (eyeIndex == 1) {
                uv.x += 0.5;
            }
            if (swapEyes) {
                uv.x = uv.x < 0.5 ? uv.x + 0.5 : uv.x - 0.5;
            }
        } else if (stereoFormat == 1) {  // Over-Under
            uv.y = uv.y * 0.5;
            if (eyeIndex == 1) {
                uv.y += 0.5;
            }
            if (swapEyes) {
                uv.y = uv.y < 0.5 ? uv.y + 0.5 : uv.y - 0.5;
            }
        }
        // Mono and anaglyph use full texture

        FragColor = texture(texture1, uv);
    }
    """

    # Compile vertex shader
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_shader_source)
    GL.glCompileShader(vertex_shader)

    if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
        error = GL.glGetShaderInfoLog(vertex_shader).decode()
        raise RuntimeError(f"Vertex shader compilation failed: {error}")

    # Compile fragment shader
    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_shader_source)
    GL.glCompileShader(fragment_shader)

    if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
        error = GL.glGetShaderInfoLog(fragment_shader).decode()
        raise RuntimeError(f"Fragment shader compilation failed: {error}")

    # Link shader program
    shader_program = GL.glCreateProgram()
    GL.glAttachShader(shader_program, vertex_shader)
    GL.glAttachShader(shader_program, fragment_shader)
    GL.glLinkProgram(shader_program)

    if not GL.glGetProgramiv(shader_program, GL.GL_LINK_STATUS):
        error = GL.glGetProgramInfoLog(shader_program).decode()
        raise RuntimeError(f"Shader program linking failed: {error}")

    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)

    return shader_program


def create_help_overlay_shaders():
    """Create simple 2D shaders for help overlay rendering"""
    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 texCoord;

    out vec2 TexCoord;

    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        TexCoord = texCoord;
    }
    """

    fragment_shader_source = """
    #version 330 core
    in vec2 TexCoord;
    out vec4 FragColor;

    uniform sampler2D helpTexture;

    void main() {
        FragColor = texture(helpTexture, TexCoord);
    }
    """

    # Compile vertex shader
    vertex_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
    GL.glShaderSource(vertex_shader, vertex_shader_source)
    GL.glCompileShader(vertex_shader)

    if not GL.glGetShaderiv(vertex_shader, GL.GL_COMPILE_STATUS):
        error = GL.glGetShaderInfoLog(vertex_shader).decode()
        raise RuntimeError(f"Help overlay vertex shader compilation failed: {error}")

    # Compile fragment shader
    fragment_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
    GL.glShaderSource(fragment_shader, fragment_shader_source)
    GL.glCompileShader(fragment_shader)

    if not GL.glGetShaderiv(fragment_shader, GL.GL_COMPILE_STATUS):
        error = GL.glGetShaderInfoLog(fragment_shader).decode()
        raise RuntimeError(f"Help overlay fragment shader compilation failed: {error}")

    # Link shader program
    shader_program = GL.glCreateProgram()
    GL.glAttachShader(shader_program, vertex_shader)
    GL.glAttachShader(shader_program, fragment_shader)
    GL.glLinkProgram(shader_program)

    if not GL.glGetProgramiv(shader_program, GL.GL_LINK_STATUS):
        error = GL.glGetProgramInfoLog(shader_program).decode()
        raise RuntimeError(f"Help overlay shader program linking failed: {error}")

    GL.glDeleteShader(vertex_shader)
    GL.glDeleteShader(fragment_shader)

    return shader_program


def setup_vao_vbo(vertices, indices):
    """Set up VAO, VBO, and EBO for geometry"""
    # Create VAO
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)

    # Create VBO
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(
        GL.GL_ARRAY_BUFFER,
        vertices.nbytes,
        vertices,
        GL.GL_STATIC_DRAW
    )

    # Create EBO (Element Buffer Object)
    ebo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
    GL.glBufferData(
        GL.GL_ELEMENT_ARRAY_BUFFER,
        indices.nbytes,
        indices,
        GL.GL_STATIC_DRAW
    )

    # Position attribute
    GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, None)
    GL.glEnableVertexAttribArray(0)

    # Texture coordinate attribute
    GL.glVertexAttribPointer(
        1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4,
        ctypes.c_void_p(3 * 4)
    )
    GL.glEnableVertexAttribArray(1)

    GL.glBindVertexArray(0)

    return vao, vbo, ebo


def setup_help_overlay_geometry():
    """Set up VAO and VBO for help overlay quad"""
    # Full-screen quad in normalized device coordinates (-1 to 1)
    # Position (x, y) and texture coordinates (u, v)
    vertices = np.array([
        # Positions   # TexCoords
        -1.0, -1.0,   0.0, 1.0,  # Bottom-left
         1.0, -1.0,   1.0, 1.0,  # Bottom-right
         1.0,  1.0,   1.0, 0.0,  # Top-right
        -1.0,  1.0,   0.0, 0.0,  # Top-left
    ], dtype=np.float32)

    # Create VAO
    vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)

    # Create VBO
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

    # Position attribute
    GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, None)
    GL.glEnableVertexAttribArray(0)

    # Texture coordinate attribute
    GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
    GL.glEnableVertexAttribArray(1)

    GL.glBindVertexArray(0)

    return vao, vbo

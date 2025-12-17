/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "render_particles.hpp"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

#include "gl_includes.hpp"

#include <print>
#include <source_location>

#include <cassert>
#include <cmath>

namespace {
constexpr static auto fp64_colour = std::array{0.4f, 0.8f, 0.1f, 1.0f};
constexpr static auto fp32_colour = std::array{1.0f, 0.6f, 0.3f, 1.0f};

//
//  used if PBO = 0?
//
// template <std::floating_point T> auto draw_points(std::span<const T> positions) -> void {
//     glBegin(GL_POINTS);
//     {
//         if constexpr (std::is_same_v<T, double>) {
//             for (auto i = 0; i < positions.size(); i += 4) {
//                 glVertex3dv(&positions[i]);
//             }
//         } else {
//             for (auto i = 0; i < positions.size(); i += 4) {
//                 glVertex3fv(&positions[i]);
//             }
//         }
//     }
//     glEnd();
// }

auto initialise_colours(std::size_t nb_bodies) -> std::vector<float> {
    auto colours = std::vector(nb_bodies * 4, 0.f);

    auto v = std::size_t{0};

    for (auto i = 0; i < nb_bodies; ++i) {
        colours[v++] = static_cast<float>(std::max((i % 3) - 1, 0));
        colours[v++] = static_cast<float>(std::max(((i + 1) % 3) - 1, 0));
        colours[v++] = static_cast<float>(std::max(((i + 2) % 3) - 1, 0));
        colours[v++] = 1.0f;
    }

    return colours;
}

template <std::floating_point T> auto make_position_pbo(std::size_t nb_bodies) -> BufferObjects<1> {
    const auto data = std::vector<T>(4 * nb_bodies, T{0});
    return BufferObjects<1>::create_dynamic(std::array{std::span{data}});
}

}    // namespace

ParticleRenderer::ParticleRenderer(std::size_t nb_bodies)
    : colour_(initialise_colours(nb_bodies)), vbo_colour_(BufferObjects<1>::create_static(std::array{std::span<const float>{colour_}})), pbo_32_(make_position_pbo<float>(nb_bodies)),
      pbo_64_(make_position_pbo<double>(nb_bodies)) {
    _initGL();
}

template <std::floating_point T> auto ParticleRenderer::draw_points(bool color, unsigned int pbo) -> void {
    assert(pbo != 0);

    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    check_OpenGL_error();
    glEnableClientState(GL_VERTEX_ARRAY);

    if constexpr (std::is_same_v<T, double>) {
        glVertexPointer(4, GL_DOUBLE, 0, 0);

    } else {
        glVertexPointer(4, GL_FLOAT, 0, 0);
    }

    const auto nb_particles = colour_.size() / 4;

    if (color) {
        [[maybe_unused]] const auto vbo_buffer = vbo_colour_.use(0);

        // vbo_colour_.use([&]() noexcept {
        glEnableClientState(GL_COLOR_ARRAY);
        // glActiveTexture(GL_TEXTURE1);
        // glTexCoordPointer(4, GL_FLOAT, 0, 0);
        glColorPointer(4, GL_FLOAT, 0, 0);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(nb_particles));

        glDisableClientState(GL_COLOR_ARRAY);
        // });
    } else {
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(nb_particles));
    }
    glDisableClientState(GL_VERTEX_ARRAY);
    check_OpenGL_error();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    check_OpenGL_error();
}

template <std::floating_point T> auto ParticleRenderer::display(DisplayMode mode, float sprite_size, unsigned int pbo) -> void {
    assert(pbo != 0u);

    constexpr auto& base_colour_ = std::is_same_v<T, double> ? fp64_colour : fp32_colour;

    switch (mode) {
        case PARTICLE_POINTS:
            {
                constexpr static auto point_size_ = 1.f;

                glColor3f(1, 1, 1);
                glPointSize(point_size_);
                glUseProgram(program_points_);
                draw_points<T>(false, pbo);
                glUseProgram(0);
            }
            break;

        case PARTICLE_SPRITES:
        default:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(sprite_size);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(program_sprites_);
                GLuint texLoc = glGetUniformLocation(program_sprites_, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texture_);

                glColor3f(1, 1, 1);
                glSecondaryColor3fv(base_colour_.data());

                draw_points<T>(false, pbo);

                glUseProgram(0);

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }

            break;

        case PARTICLE_SPRITES_COLOR:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(sprite_size);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(program_sprites_);
                GLuint texLoc = glGetUniformLocation(program_sprites_, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, texture_);

                glColor3f(1, 1, 1);
                glSecondaryColor3fv(base_colour_.data());

                draw_points<T>(true, pbo);

                glUseProgram(0);

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }

            break;
    }

    check_OpenGL_error();
}

auto ParticleRenderer::display(DisplayMode mode, float sprite_size, std::span<const float> pos) -> void {
    assert(pos.size() == colour_.size());

    pbo_32_.bind_data(0, pos);

    display<float>(mode, sprite_size, pbo_32_.buffer(0));
}

auto ParticleRenderer::display(DisplayMode mode, float sprite_size, std::span<const double> pos) -> void {
    assert(pos.size() == colour_.size());

    pbo_64_.bind_data(0, pos);

    display<double>(mode, sprite_size, pbo_64_.buffer(0));
}

void ParticleRenderer::_initGL() {
    constexpr static auto vertexShaderPoints =
        R"(void main() {
                vec4 vert = vec4(gl_Vertex.xyz, 1.0);
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;
                gl_FrontColor = gl_Color;
            })";

    constexpr static auto vertexShader =
        R"(void main() {
                float pointSize = 500.0 * gl_Point.size;
                vec4 vert = gl_Vertex;
                vert.w = 1.0;
                vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);
                gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));
                gl_TexCoord[0] = gl_MultiTexCoord0;
                gl_TexCoord[1] = gl_MultiTexCoord1;
                gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vert;
                gl_FrontColor = gl_Color;
                gl_FrontSecondaryColor = gl_SecondaryColor;
            })";

    constexpr static auto pixelShader =
        R"(uniform sampler2D splatTexture;
           void main() {
                vec4 color2 = gl_SecondaryColor;
                vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st);
                gl_FragColor = color * color2;      // mix(vec4(0.1, 0.0, 0.0, color.w), color2, color.w);
            })";

    auto m_vertexShader       = glCreateShader(GL_VERTEX_SHADER);
    auto m_vertexShaderPoints = glCreateShader(GL_VERTEX_SHADER);
    auto m_pixelShader        = glCreateShader(GL_FRAGMENT_SHADER);

    glShaderSource(m_vertexShader, 1, &vertexShader, 0);
    glShaderSource(m_pixelShader, 1, &pixelShader, 0);
    glShaderSource(m_vertexShaderPoints, 1, &vertexShaderPoints, 0);

    glCompileShader(m_vertexShader);
    glCompileShader(m_vertexShaderPoints);
    glCompileShader(m_pixelShader);

    program_sprites_ = glCreateProgram();
    glAttachShader(program_sprites_, m_vertexShader);
    glAttachShader(program_sprites_, m_pixelShader);
    glLinkProgram(program_sprites_);

    program_points_ = glCreateProgram();
    glAttachShader(program_points_, m_vertexShaderPoints);
    glLinkProgram(program_points_);

    _createTexture();
}

//------------------------------------------------------------------------------
// Function           : EvalHermite
// Description      :
//------------------------------------------------------------------------------
///
/// EvalHermite(float pA, float pB, float vA, float vB, float u)
/// @brief Evaluates Hermite basis functions for the specified coefficients.
///
constexpr auto evalHermite(float u) -> float {
    const auto u2 = u * u;
    const auto u3 = u2 * u;
    return 2 * u3 - 3 * u2 + 1;
}

template <std::size_t N> auto createGaussianMap() {
    constexpr auto Incr = 2.0f / N;

    auto M = std::array<float, 2 * N * N>{};
    auto B = std::array<unsigned char, 4 * N * N>{};
    auto i = 0;
    auto j = 0;

    // float mmax = 0;
    for (auto y = 0u; y < N; ++y) {
        const auto Y  = y * Incr - 1.0f;
        const auto Y2 = Y * Y;

        for (auto x = 0u; x < N; ++x, i += 2, j += 4) {
            const auto X     = x * Incr - 1.0f;
            const auto X2_Y2 = X * X + Y2;

            const auto dist = X2_Y2 > 1 ? 1.0f : std::sqrt(X2_Y2);

            M[i + 1] = M[i] = evalHermite(dist);
            B[j + 3] = B[j + 2] = B[j + 1] = B[j] = static_cast<unsigned char>(M[i] * 255);
        }
    }

    return B;
}

void ParticleRenderer::_createTexture() {
    constexpr auto resolution = 32;
    const auto     data       = createGaussianMap<resolution>();
    glGenTextures(1, reinterpret_cast<GLuint*>(&texture_));
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
}

template auto ParticleRenderer::display<float>(DisplayMode mode, float sprite_size, unsigned int pbo) -> void;
template auto ParticleRenderer::display<double>(DisplayMode mode, float sprite_size, unsigned int pbo) -> void;
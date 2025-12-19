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

#pragma once

#include "buffer_objects.hpp"

#include <array>
#include <span>
#include <vector>

class ParticleRenderer {
 public:
    explicit ParticleRenderer(std::size_t nb_bodies);

    auto colour() noexcept -> std::span<float> { return colour_; }

    enum DisplayMode { PARTICLE_POINTS, PARTICLE_SPRITES, PARTICLE_SPRITES_COLOR, PARTICLE_NUM_MODES };

    // invoked by CPU impl or GPU impl using host memory
    auto display(DisplayMode mode, float sprite_size, std::span<const float> pos) -> void;
    auto display(DisplayMode mode, float sprite_size, std::span<const double> pos) -> void;

    // invoked by GPU impl using OpenGL interop
    template <std::floating_point T> auto display(DisplayMode mode, float sprite_size, const BufferObject& pbo) -> void;

 private:    // methods
    void _initGL();
    void _createTexture();

    template <std::floating_point T, bool UseColour> auto draw_points() -> void;

    std::vector<float> colour_;

    std::span<const float>  pos_;
    std::span<const double> pos_fp64_;

    unsigned int     program_points_  = 0;
    unsigned int     program_sprites_ = 0;
    unsigned int     texture_         = 0;
    BufferObjects<1> vbo_colour_;
    BufferObjects<1> pbo_32_;
    BufferObjects<1> pbo_64_;
};

extern template auto ParticleRenderer::display<float>(DisplayMode mode, float sprite_size, const BufferObject& pbo) -> void;
extern template auto ParticleRenderer::display<double>(DisplayMode mode, float sprite_size, const BufferObject& pbo) -> void;
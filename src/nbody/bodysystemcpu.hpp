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

#include "nbody_config.hpp"

#include <filesystem>
#include <span>
#include <vector>

struct NBodyParams;

// CPU Body System
template <std::floating_point T> class BodySystemCPU {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = true;

    BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params);

    BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto update(T deltaTime) noexcept -> void;

    auto update_params(const NBodyParams& active_params) noexcept -> void;

    auto get_position() const noexcept -> std::span<const T> { return m_pos; }
    auto get_velocity() const noexcept -> std::span<const T> { return m_vel; }

    auto get_position() noexcept -> std::span<T> { return m_pos; }
    auto get_velocity() noexcept -> std::span<T> { return m_vel; }

    auto set_position(std::span<const T> data) noexcept -> void;
    auto set_velocity(std::span<const T> data) noexcept -> void;

 private:
    auto setSoftening(T softening) noexcept -> void { m_softeningSquared = softening * softening; }

    auto _computeNBodyGravitation() noexcept -> void;

    std::size_t m_numBodies;

    std::vector<T> m_pos;
    std::vector<T> m_vel;
    std::vector<T> m_force = std::vector(m_numBodies * 3, T{0.f});

    T m_softeningSquared = 0.00125f;
    T m_damping          = 0.995f;
};

extern template BodySystemCPU<float>;
extern template BodySystemCPU<double>;
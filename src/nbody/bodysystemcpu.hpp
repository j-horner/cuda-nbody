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

#include <array>
#include <span>
#include <vector>

struct NBodyParams;

// CPU Body System
template <std::floating_point T> class BodySystemCPU {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = true;

    BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params);

    BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::span<const T> positions, std::span<const T> velocities);

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto update(T dt) noexcept -> void;

    auto update_params(const NBodyParams& active_params) noexcept -> void;

    auto get_position() const noexcept -> std::span<const T> { return {pos_.front().data(), nb_bodies_ * 4}; }
    auto get_velocity() const noexcept -> std::span<const T> { return {vel_.front().data(), nb_bodies_ * 4}; }

    auto get_position() noexcept -> std::span<T> { return {pos_.front().data(), nb_bodies_ * 4}; }
    auto get_velocity() noexcept -> std::span<T> { return {vel_.front().data(), nb_bodies_ * 4}; }

    auto set_position(std::span<const T> data) noexcept -> void;
    auto set_velocity(std::span<const T> data) noexcept -> void;

 private:
    std::size_t nb_bodies_;

    std::vector<std::array<T, 4>> pos_{nb_bodies_};
    std::vector<std::array<T, 4>> vel_{nb_bodies_};
    std::vector<std::array<T, 4>> dv_{nb_bodies_};

    T softening_squared_ = 0.00125f;
    T damping_           = 0.995f;
};

extern template BodySystemCPU<float>;
extern template BodySystemCPU<double>;
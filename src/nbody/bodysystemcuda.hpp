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

#include <concepts>
#include <span>
#include <vector>

struct NBodyParams;

template <std::floating_point T> class BodySystemCUDA {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = false;

    BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);

    auto virtual get_position() const -> std::span<const T> = 0;
    auto virtual get_velocity() const -> std::span<const T> = 0;

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto virtual update(T deltaTime) -> void = 0;

    auto update_params(const NBodyParams& active_params) -> void;

    auto virtual set_position(std::span<const T> data) -> void = 0;
    auto virtual set_velocity(std::span<const T> data) -> void = 0;

    virtual ~BodySystemCUDA() = default;

 protected:
    unsigned int nb_bodies_;

    std::vector<T> host_pos_vec_ = std::vector<T>(4 * nb_bodies_, T{0});
    std::vector<T> host_vel_vec_ = std::vector<T>(4 * nb_bodies_, T{0});

    T damping_ = 0.995f;

    unsigned int current_read_  = 0u;
    unsigned int current_write_ = 1u;

    unsigned int block_size_;
};

extern template BodySystemCUDA<float>;
extern template BodySystemCUDA<double>;

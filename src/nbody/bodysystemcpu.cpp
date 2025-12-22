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

#include "bodysystemcpu.hpp"

#include "params.hpp"
#include "randomise_bodies.hpp"
#include "vec.hpp"

#include <omp.h>

#include <algorithm>
#include <array>
#include <span>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using std::ranges::copy;

namespace {

template <std::floating_point T> auto interaction(std::array<T, 3>& acc, const std::array<T, 3>& pos_i, const std::array<T, 4>& pos_j, T softening_squared) noexcept -> void {
    // r_01  [3 FLOPS]
    const auto dx = std::array{pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};

    // d^2 + e^2 [6 FLOPS]
    const auto r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] + softening_squared;

    // NOTE: sqrt and / are long calculations
    // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

    // 1 FLOP
    const auto r = std::sqrt(r2);

    // 2 FLOPs
    const auto m_r4 = pos_j[3] / (r2 * r2);

    // 1 FLOP
    const auto s = m_r4 * r;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    acc[0] += dx[0] * s;
    acc[1] += dx[1] * s;
    acc[2] += dx[2] * s;
}

}    // namespace

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params)
    : nb_bodies_(nb_bodies), pos_(nb_bodies_), vel_(nb_bodies_), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::span<const T> positions, std::span<const T> velocities)
    : nb_bodies_(nb_bodies), pos_(nb_bodies_), vel_(nb_bodies_), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    assert(pos_.size() == nb_bodies_ * 4);
    assert(vel_.size() == pos_.size());

    set_position(positions);
    set_velocity(velocities);
}

template <std::floating_point T> auto BodySystemCPU<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, get_position(), get_velocity(), params.cluster_scale, params.velocity_scale);
}

template <std::floating_point T> auto BodySystemCPU<T>::update_params(const NBodyParams& active_params) noexcept -> void {
    softening_squared_ = static_cast<T>(active_params.softening) * active_params.softening;
    damping_           = active_params.damping;
}

template <std::floating_point T> auto BodySystemCPU<T>::set_position(std::span<const T> data) noexcept -> void {
    assert(data.size() == pos_.size());
    copy(data, pos_.front().data());
}
template <std::floating_point T> auto BodySystemCPU<T>::set_velocity(std::span<const T> data) noexcept -> void {
    assert(data.size() == vel_.size());
    copy(data, vel_.front().data());
}

template <std::floating_point T> auto BodySystemCPU<T>::update(T dt) noexcept -> void {
#pragma omp                           parallel for
    for (int i = 0; i < nb_bodies_; i++) {
        auto       acc   = std::array<T, 3>{0, 0, 0};
        const auto pos_i = std::array<T, 3>{pos_[i][0], pos_[i][1], pos_[i][2]};

        // We unroll this loop 4X for a small performance boost.
        for (auto j = 0; j < nb_bodies_; j += 4) {
            interaction(acc, pos_i, pos_[j], softening_squared_);
            interaction(acc, pos_i, pos_[j + 1], softening_squared_);
            interaction(acc, pos_i, pos_[j + 2], softening_squared_);
            interaction(acc, pos_i, pos_[j + 3], softening_squared_);
        }

        accel_[i] = acc;
    }

#pragma omp parallel for

    for (int i = 0; i < nb_bodies_; ++i) {
        auto pos = pos_[i];
        auto vel = vel_[i];

        const auto& accel = accel_[i];

        // new velocity = old velocity + acceleration * dt
        vel[0] += accel[0] * dt;
        vel[1] += accel[1] * dt;
        vel[2] += accel[2] * dt;

        vel[0] *= damping_;
        vel[1] *= damping_;
        vel[2] *= damping_;

        // new position = old position + velocity * dt
        pos[0] += vel[0] * dt;
        pos[1] += vel[1] * dt;
        pos[2] += vel[2] * dt;

        pos_[i] = pos;
        vel_[i] = vel;
    }
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
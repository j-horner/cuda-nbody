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
#include <numeric>
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
    const auto dr = std::array{pos_j[0] - pos_i[0], pos_j[1] - pos_i[1], pos_j[2] - pos_i[2]};

    // NOTE: sqrt and / (and r2) are long calculations
    // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

    const auto dx2 = dr[0] * dr[0];
    const auto dy2 = dr[1] * dr[1];
    const auto dz2 = dr[2] * dr[2];

    const auto dx2_dy2   = dx2 + dy2;
    const auto dz2_soft2 = dz2 + softening_squared;

    // d^2 + e^2 [6 FLOPS]
    const auto r2 = dx2_dy2 + dz2_soft2;

    // 1 FLOP
    const auto r = std::sqrt(r2);

    // 2 FLOPs
    const auto m_r4 = pos_j[3] / (r2 * r2);

    // 1 FLOP
    const auto s = m_r4 * r;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    acc[0] += dr[0] * s;
    acc[1] += dr[1] * s;
    acc[2] += dr[2] * s;
}

}    // namespace

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params) : nb_bodies_(nb_bodies), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::span<const T> positions, std::span<const T> velocities)
    : nb_bodies_(nb_bodies), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    set_position(positions);
    set_velocity(velocities);
}

template <std::floating_point T> auto BodySystemCPU<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, positions_, velocities_, masses_, params.cluster_scale, params.velocity_scale);
}

template <std::floating_point T> auto BodySystemCPU<T>::update_params(const NBodyParams& active_params) noexcept -> void {
    softening_squared_ = static_cast<T>(active_params.softening) * active_params.softening;
    damping_           = active_params.damping;
}

template <std::floating_point T> auto BodySystemCPU<T>::set_position(std::span<const T> data) noexcept -> void {
    assert(data.size() == nb_bodies_ * 4);

    for (auto i = std::size_t{0}; i < nb_bodies_; ++i) {
        positions_.x[i] = data[4 * i];
        positions_.y[i] = data[4 * i + 1];
        positions_.z[i] = data[4 * i + 2];
        masses_[i]      = data[4 * i + 3];
    }
}
template <std::floating_point T> auto BodySystemCPU<T>::set_velocity(std::span<const T> data) noexcept -> void {
    assert(data.size() == nb_bodies_ * 4);

    for (auto i = std::size_t{0}; i < nb_bodies_; ++i) {
        velocities_.x[i] = data[4 * i];
        velocities_.y[i] = data[4 * i + 1];
        velocities_.z[i] = data[4 * i + 2];
    }
}

template <std::floating_point T> auto BodySystemCPU<T>::update(T dt) noexcept -> void {
    const auto nb_bodies         = nb_bodies_;
    const auto softening_squared = softening_squared_;

#pragma omp parallel for
    for (int i = 0; i < nb_bodies; i++) {
        auto       acc   = std::array<T, 3>{0, 0, 0};
        const auto pos_i = std::array<T, 3>{positions_.x[i], positions_.y[i], positions_.z[i]};

        // We unroll this loop 4X for a small performance boost.
        for (auto j = 0; j < nb_bodies; ++j) {
            // r_01  [3 FLOPS]
            const auto dr = std::array{positions_.x[j] - pos_i[0], positions_.y[j] - pos_i[1], positions_.z[j] - pos_i[2]};

            // NOTE: sqrt and / (and r2) are long calculations
            // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

            const auto dx2 = dr[0] * dr[0];
            const auto dy2 = dr[1] * dr[1];
            const auto dz2 = dr[2] * dr[2];

            const auto dx2_dy2   = dx2 + dy2;
            const auto dz2_soft2 = dz2 + softening_squared;

            // d^2 + e^2 [6 FLOPS]
            const auto r2 = dx2_dy2 + dz2_soft2;

            // 1 FLOP
            const auto r = std::sqrt(r2);

            // 2 FLOPs
            const auto m_r4 = masses_[j] / (r2 * r2);

            // 1 FLOP
            const auto s = m_r4 * r;

            // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
            acc[0] += dr[0] * s;
            acc[1] += dr[1] * s;
            acc[2] += dr[2] * s;
        }

        dv_.x[i] = acc[0] * dt;
        dv_.y[i] = acc[1] * dt;
        dv_.z[i] = acc[2] * dt;
    }

    const auto damping = damping_;

#pragma omp parallel for
    for (auto i = std::size_t{0}; i < nb_bodies; ++i) {
        // new velocity = old velocity + acceleration * dt
        velocities_.x[i] = (velocities_.x[i] + dv_.x[i]) * damping;
        velocities_.y[i] = (velocities_.y[i] + dv_.y[i]) * damping;
        velocities_.z[i] = (velocities_.z[i] + dv_.z[i]) * damping;

        // new position = old position + velocity * dt
        positions_.x[i] += velocities_.x[i] * dt;
        positions_.y[i] += velocities_.y[i] * dt;
        positions_.z[i] += velocities_.z[i] * dt;
    }
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
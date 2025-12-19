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

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params)
    : nb_bodies_(nb_bodies), pos_(nb_bodies_ * 4, T{0}), vel_(nb_bodies_ * 4, T{0}), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : nb_bodies_(nb_bodies), pos_(std::move(positions)), vel_(std::move(velocities)), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    assert(pos_.size() == nb_bodies_ * 4);
    assert(vel_.size() == pos_.size());
}

template <std::floating_point T> auto BodySystemCPU<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, pos_, vel_, params.cluster_scale, params.velocity_scale);
}

template <std::floating_point T> auto BodySystemCPU<T>::update_params(const NBodyParams& active_params) noexcept -> void {
    softening_squared_ = static_cast<T>(active_params.softening) * active_params.softening;
    damping_           = active_params.damping;
}

template <std::floating_point T> auto BodySystemCPU<T>::set_position(std::span<const T> data) noexcept -> void {
    assert(data.size() == pos_.size());
    copy(data, pos_.begin());
}
template <std::floating_point T> auto BodySystemCPU<T>::set_velocity(std::span<const T> data) noexcept -> void {
    assert(data.size() == vel_.size());
    copy(data, vel_.begin());
}

template <std::floating_point T> auto interaction(T accel[3], const T pos_mass_0[4], const T pos_mass_1[4], T softening_squared) noexcept -> void {
    // r_01  [3 FLOPS]
    const auto r = std::array{pos_mass_1[0] - pos_mass_0[0], pos_mass_1[1] - pos_mass_0[1], pos_mass_1[2] - pos_mass_0[2]};

    // d^2 + e^2 [6 FLOPS]
    const auto dist_sqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + softening_squared;

    // [3 FLOPS (2 mul, 1 sqrt)]
    const auto dist_cubed = std::sqrt(dist_sqr * dist_sqr * dist_sqr);

    // m/r^3 [1 FLOP]
    const auto s = pos_mass_1[3] / dist_cubed;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}

template <std::floating_point T> auto BodySystemCPU<T>::calculate_force() noexcept -> void {
#pragma omp                           parallel for
    for (int i = 0; i < nb_bodies_; i++) {
        int indexForce = 3 * i;

        T acc[3] = {0, 0, 0};

        // We unroll this loop 4X for a small performance boost.
        int j = 0;

        while (j < nb_bodies_) {
            interaction<T>(acc, &pos_[4 * i], &pos_[4 * j], softening_squared_);
            j++;
            interaction<T>(acc, &pos_[4 * i], &pos_[4 * j], softening_squared_);
            j++;
            interaction<T>(acc, &pos_[4 * i], &pos_[4 * j], softening_squared_);
            j++;
            interaction<T>(acc, &pos_[4 * i], &pos_[4 * j], softening_squared_);
            j++;
        }

        force_[indexForce]     = acc[0];
        force_[indexForce + 1] = acc[1];
        force_[indexForce + 2] = acc[2];
    }
}

template <std::floating_point T> auto BodySystemCPU<T>::update(T dt) noexcept -> void {
    calculate_force();

#pragma omp parallel for

    for (int i = 0; i < nb_bodies_; ++i) {
        int index      = 4 * i;
        int indexForce = 3 * i;

        T pos[3], vel[3], force[3];
        pos[0]    = pos_[index + 0];
        pos[1]    = pos_[index + 1];
        pos[2]    = pos_[index + 2];
        T invMass = pos_[index + 3];

        vel[0] = vel_[index + 0];
        vel[1] = vel_[index + 1];
        vel[2] = vel_[index + 2];

        force[0] = force_[indexForce + 0];
        force[1] = force_[indexForce + 1];
        force[2] = force_[indexForce + 2];

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * dt
        vel[0] += (force[0] * invMass) * dt;
        vel[1] += (force[1] * invMass) * dt;
        vel[2] += (force[2] * invMass) * dt;

        vel[0] *= damping_;
        vel[1] *= damping_;
        vel[2] *= damping_;

        // new position = old position + velocity * dt
        pos[0] += vel[0] * dt;
        pos[1] += vel[1] * dt;
        pos[2] += vel[2] * dt;

        pos_[index + 0] = pos[0];
        pos_[index + 1] = pos[1];
        pos_[index + 2] = pos[2];

        vel_[index + 0] = vel[0];
        vel_[index + 1] = vel[1];
        vel_[index + 2] = vel[2];
    }
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
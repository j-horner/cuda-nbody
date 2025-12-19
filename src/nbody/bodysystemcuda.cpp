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

#include "bodysystemcuda.hpp"

#include "params.hpp"
#include "randomise_bodies.hpp"

#include <vector>

auto set_softening_squared(float softeningSq) -> void;
auto set_softening_squared(double softeningSq) -> void;

namespace {

template <std::floating_point T> auto set_softening(T softening) -> void {
    const auto softening2 = softening * softening;

    set_softening_squared(softening2);
}

}    // namespace

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params)
    : BodySystemCUDA(nb_bodies, blockSize, params, std::vector<T>(nb_bodies * 4, T{0}), std::vector<T>(nb_bodies * 4, T{0})) {}

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : nb_bodies_(nb_bodies), block_size_(blockSize), host_pos_vec_(std::move(positions)), host_vel_vec_(std::move(velocities)), damping_(params.damping) {
    set_softening(static_cast<T>(params.softening));
}

template <std::floating_point T> auto BodySystemCUDA<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, host_pos_vec_, host_vel_vec_, params.cluster_scale, params.velocity_scale);
    set_position(host_pos_vec_);
    set_velocity(host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDA<T>::update_params(const NBodyParams& active_params) -> void {
    set_softening(static_cast<T>(active_params.softening));
    damping_ = active_params.damping;
}

template BodySystemCUDA<float>;
template BodySystemCUDA<double>;

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

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>
#include <omp.h>
#include <xsimd/xsimd.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <print>
#include <span>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using std::ranges::copy;
using std::ranges::fill;

namespace {

auto current_clock_cycle() noexcept {
    return __rdtsc();
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
    namespace xs = xsimd;

    constexpr auto stride = static_cast<int>(xs::simd_type<T>::size);

    // const auto nb_interactions = nb_bodies_ * nb_bodies_;

    // const auto cycle_start = current_clock_cycle();

    const auto nb_bodies = static_cast<int>(nb_bodies_);

    // could separate 1st particles contribution and initialise dv there but doesnt noticeably improve performance
    fill(dv_.x, T{0});
    fill(dv_.y, T{0});
    fill(dv_.z, T{0});

    {
        const auto softening_squared = xs::batch<T>{softening_squared_};

#pragma omp parallel for
        for (auto i = 0; i < nb_bodies; i += stride) {
            const auto pos_i_x = xs::load_aligned(positions_.x.data() + i);
            const auto pos_i_y = xs::load_aligned(positions_.y.data() + i);
            const auto pos_i_z = xs::load_aligned(positions_.z.data() + i);

            auto dv_x = xs::load_aligned(dv_.x.data() + i);
            auto dv_y = xs::load_aligned(dv_.y.data() + i);
            auto dv_z = xs::load_aligned(dv_.z.data() + i);

            for (auto j = 0; j < nb_bodies; ++j) {
                // dr  [3 FLOPS]
                const auto dx = xs::batch<T>{positions_.x[j]} - pos_i_x;
                const auto dy = xs::batch<T>{positions_.y[j]} - pos_i_y;
                const auto dz = xs::batch<T>{positions_.z[j]} - pos_i_z;

                // NOTE: sqrt and / (and r2) are long calculations
                // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

                // d^2 + e^2 [6 FLOPS]
                const auto r2 = (softening_squared + (dx * dx)) + ((dy * dy) + (dz * dz));

                // 4 FLOPS
                const auto m_r3 = (xs::batch<T>{masses_[j]} / (r2 * r2)) * xs::sqrt(r2);

                // 6 FLOPS
                dv_x += (m_r3 * dx);
                dv_y += (m_r3 * dy);
                dv_z += (m_r3 * dz);
            }
            dv_x.store_aligned(dv_.x.data() + i);
            dv_y.store_aligned(dv_.y.data() + i);
            dv_z.store_aligned(dv_.z.data() + i);
        }
    }

    const auto damping = xs::batch<T>{damping_};
    const auto dt_     = xs::batch<T>{dt};

    const auto integrate = [&]<auto Dim> {
        for (auto i = 0; i < nb_bodies; i += stride) {
            auto dv  = xs::load_aligned((dv_.*Dim).data() + i);
            auto v   = xs::load_aligned((velocities_.*Dim).data() + i);
            auto pos = xs::load_aligned((positions_.*Dim).data() + i);

            dv *= dt_;

            v += dv;
            v *= damping;

            pos += (v * dt_);

            v.store_aligned((velocities_.*Dim).data() + i);
            pos.store_aligned((positions_.*Dim).data() + i);
        }
    };

    integrate.operator()<&Coordinates<T>::x>();
    integrate.operator()<&Coordinates<T>::y>();
    integrate.operator()<&Coordinates<T>::z>();

    // std::println("{}", (current_clock_cycle() - cycle_start) / nb_interactions);
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
;
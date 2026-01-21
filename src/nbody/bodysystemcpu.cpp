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
    // const auto nb_interactions = nb_bodies_ * nb_bodies_;

    // const auto cycle_start = current_clock_cycle();

    const auto nb_bodies         = nb_bodies_;
    const auto softening_squared = softening_squared_;
    const auto damping           = damping_;

    // could separate 1st particles contribution and initialise dv there but doesnt noticeably improve performance
    fill(dv_.x, T{0});
    fill(dv_.y, T{0});
    fill(dv_.z, T{0});

    if constexpr (std::is_same_v<T, float>) {
        {
            const auto softening_8 = _mm256_set1_ps(softening_squared);

            for (auto j = 0; j < nb_bodies; ++j) {
                const auto pos_j_x = _mm256_set1_ps(positions_.x[j]);
                const auto pos_j_y = _mm256_set1_ps(positions_.y[j]);
                const auto pos_j_z = _mm256_set1_ps(positions_.z[j]);

                const auto m_j = _mm256_set1_ps(masses_[j]);

#pragma omp parallel for
                for (auto i = std::size_t{0}; i < nb_bodies; i += 8) {
                    const auto pos_i_x = _mm256_load_ps(positions_.x.data() + i);
                    const auto pos_i_y = _mm256_load_ps(positions_.y.data() + i);
                    const auto pos_i_z = _mm256_load_ps(positions_.z.data() + i);

                    auto dv_x = _mm256_load_ps(dv_.x.data() + i);
                    auto dv_y = _mm256_load_ps(dv_.y.data() + i);
                    auto dv_z = _mm256_load_ps(dv_.z.data() + i);

                    // dr  [3 FLOPS]
                    const auto dx = _mm256_sub_ps(pos_j_x, pos_i_x);
                    const auto dy = _mm256_sub_ps(pos_j_y, pos_i_y);
                    const auto dz = _mm256_sub_ps(pos_j_z, pos_i_z);

                    // NOTE: sqrt and / (and r2) are long calculations
                    // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

                    // d^2 + e^2 [6 FLOPS]
                    const auto r2 = _mm256_add_ps(_mm256_add_ps(softening_8, _mm256_mul_ps(dx, dx)), _mm256_add_ps(_mm256_mul_ps(dy, dy), _mm256_mul_ps(dz, dz)));

                    // 1 FLOP
                    const auto r = _mm256_sqrt_ps(r2);

                    // 2 FLOPs
                    const auto m_r4 = _mm256_div_ps(m_j, _mm256_mul_ps(r2, r2));

                    // 1 FLOP
                    const auto m_r3 = _mm256_mul_ps(m_r4, r);

                    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
                    const auto f_x = _mm256_mul_ps(m_r3, dx);
                    const auto f_y = _mm256_mul_ps(m_r3, dy);
                    const auto f_z = _mm256_mul_ps(m_r3, dz);

                    dv_x = _mm256_add_ps(dv_x, f_x);
                    dv_y = _mm256_add_ps(dv_y, f_y);
                    dv_z = _mm256_add_ps(dv_z, f_z);

                    _mm256_store_ps(dv_.x.data() + i, dv_x);
                    _mm256_store_ps(dv_.y.data() + i, dv_y);
                    _mm256_store_ps(dv_.z.data() + i, dv_z);
                }
            }
        }

        const auto damping_8 = _mm256_set1_ps(damping);
        const auto dt_8      = _mm256_set1_ps(dt);

        const auto integrate = [&]<auto Dim> {
#pragma omp parallel for
            for (auto i = std::size_t{0}; i < nb_bodies; i += 8) {
                auto dv  = _mm256_load_ps((dv_.*Dim).data() + i);
                auto v   = _mm256_load_ps((velocities_.*Dim).data() + i);
                auto pos = _mm256_load_ps((positions_.*Dim).data() + i);

                dv = _mm256_mul_ps(dv, dt_8);

                v = _mm256_add_ps(v, dv);
                v = _mm256_mul_ps(v, damping_8);

                auto delta_pos = _mm256_mul_ps(v, dt_8);
                pos            = _mm256_add_ps(pos, delta_pos);

                _mm256_store_ps((velocities_.*Dim).data() + i, v);
                _mm256_store_ps((positions_.*Dim).data() + i, pos);
            }
        };

        integrate.operator()<&Coordinates<T>::x>();
        integrate.operator()<&Coordinates<T>::y>();
        integrate.operator()<&Coordinates<T>::z>();
    } else {
        {
            const auto softening_4 = _mm256_set1_pd(softening_squared);

            for (auto j = 0; j < nb_bodies; ++j) {
                const auto pos_j_x = _mm256_set1_pd(positions_.x[j]);
                const auto pos_j_y = _mm256_set1_pd(positions_.y[j]);
                const auto pos_j_z = _mm256_set1_pd(positions_.z[j]);

                const auto m_j = _mm256_set1_pd(masses_[j]);

#pragma omp parallel for
                for (auto i = std::size_t{0}; i < nb_bodies; i += 4) {
                    const auto pos_i_x = _mm256_load_pd(positions_.x.data() + i);
                    const auto pos_i_y = _mm256_load_pd(positions_.y.data() + i);
                    const auto pos_i_z = _mm256_load_pd(positions_.z.data() + i);

                    auto dv_x = _mm256_load_pd(dv_.x.data() + i);
                    auto dv_y = _mm256_load_pd(dv_.y.data() + i);
                    auto dv_z = _mm256_load_pd(dv_.z.data() + i);

                    // dr  [3 FLOPS]
                    const auto dx = _mm256_sub_pd(pos_j_x, pos_i_x);
                    const auto dy = _mm256_sub_pd(pos_j_y, pos_i_y);
                    const auto dz = _mm256_sub_pd(pos_j_z, pos_i_z);

                    // NOTE: sqrt and / (and r2) are long calculations
                    // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

                    // d^2 + e^2 [6 FLOPS]
                    const auto r2 = _mm256_add_pd(_mm256_add_pd(softening_4, _mm256_mul_pd(dx, dx)), _mm256_add_pd(_mm256_mul_pd(dy, dy), _mm256_mul_pd(dz, dz)));

                    // 1 FLOP
                    const auto r = _mm256_sqrt_pd(r2);

                    // 2 FLOPs
                    const auto m_r4 = _mm256_div_pd(m_j, _mm256_mul_pd(r2, r2));

                    // 1 FLOP
                    const auto m_r3 = _mm256_mul_pd(m_r4, r);

                    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
                    const auto f_x = _mm256_mul_pd(m_r3, dx);
                    const auto f_y = _mm256_mul_pd(m_r3, dy);
                    const auto f_z = _mm256_mul_pd(m_r3, dz);

                    dv_x = _mm256_add_pd(dv_x, f_x);
                    dv_y = _mm256_add_pd(dv_y, f_y);
                    dv_z = _mm256_add_pd(dv_z, f_z);

                    _mm256_store_pd(dv_.x.data() + i, dv_x);
                    _mm256_store_pd(dv_.y.data() + i, dv_y);
                    _mm256_store_pd(dv_.z.data() + i, dv_z);
                }
            }
        }

        const auto damping_4 = _mm256_set1_pd(damping);
        const auto dt_4      = _mm256_set1_pd(dt);

        const auto integrate = [&]<auto Dim> {
#pragma omp parallel for
            for (auto i = std::size_t{0}; i < nb_bodies; i += 4) {
                auto dv  = _mm256_load_pd((dv_.*Dim).data() + i);
                auto v   = _mm256_load_pd((velocities_.*Dim).data() + i);
                auto pos = _mm256_load_pd((positions_.*Dim).data() + i);

                dv = _mm256_mul_pd(dv, dt_4);

                v = _mm256_add_pd(v, dv);
                v = _mm256_mul_pd(v, damping_4);

                auto delta_pos = _mm256_mul_pd(v, dt_4);
                pos            = _mm256_add_pd(pos, delta_pos);

                _mm256_store_pd((velocities_.*Dim).data() + i, v);
                _mm256_store_pd((positions_.*Dim).data() + i, pos);
            }
        };

        integrate.operator()<&Coordinates<T>::x>();
        integrate.operator()<&Coordinates<T>::y>();
        integrate.operator()<&Coordinates<T>::z>();
    }

    // std::println("{}", (current_clock_cycle() - cycle_start) / nb_interactions);
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
;
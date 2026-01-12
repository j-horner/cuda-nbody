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

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <immintrin.h>

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

namespace {

auto rdtsc() noexcept {
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

    for (auto& dv : dv_) {
        dv[3] = T{0};
    }
}

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::span<const T> positions, std::span<const T> velocities)
    : nb_bodies_(nb_bodies), softening_squared_(static_cast<T>(params.softening) * params.softening), damping_(params.damping) {
    assert(pos_.size() == nb_bodies_ * 4);
    assert(vel_.size() == pos_.size());

    set_position(positions);
    set_velocity(velocities);
    for (auto& dv : dv_) {
        dv[3] = T{0};
    }
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
    const auto initial_cycle_count = rdtsc();

    const auto nb_bodies         = nb_bodies_;
    const auto softening_squared = softening_squared_;

    if constexpr (std::is_same_v<T, float>) {
        const auto dt_8   = _mm256_setr_ps(dt, dt, dt, dt, dt, dt, dt, dt);
        const auto soft_8 = _mm256_setr_ps(softening_squared, softening_squared, softening_squared, softening_squared, softening_squared, softening_squared, softening_squared, softening_squared);

        auto dr = std::array<std::array<std::array<float, 4>, 4>, 4>{};

        auto s = std::array<float, 8>{};

#pragma omp parallel for
        for (auto i = 0; i < nb_bodies; i += 2) {
            auto acc = _mm256_setzero_ps();
            // auto acc = std::array{0.f, 0.f, 0.f, 0.f};

            const auto pos_i   = pos_[i];
            const auto pos_i_1 = pos_[i + 1];

            // const auto x0 = _mm256_setr_ps(pos_[i][0], pos_[i][1], pos_[i][2], 0.f, 0.f, 0.f, 0.f, 0.f);
            // const auto x1 = _mm256_setr_ps(pos_[i + 1][0], pos_[i + 1][1], pos_[i + 1][2], 0.f, 0.f, 0.f, 0.f, 0.f);

            // const auto x = _mm256_setr_ps(pos_[i][0], pos_[i][1], pos_[i][2], 0.f, pos_[i + 1][0], pos_[i + 1][1], pos_[i + 1][2], 0.f);

            for (auto j = 0; j < nb_bodies; j += 2) {
                // auto dr0_ = _mm256_setr_ps(pos_[j][0], pos_[j][1], pos_[j][2], 0.f, pos_[j][0], pos_[j][1], pos_[j][2], 0.f);
                // auto dr1_ = _mm256_setr_ps(pos_[j + 1][0], pos_[j + 1][1], pos_[j + 1][2], 0.f, pos_[j + 1][0], pos_[j + 1][1], pos_[j + 1][2], 0.f);
                // auto dr2_ = _mm256_setr_ps(pos_[j + 2][0], pos_[j + 2][1], pos_[j + 2][2], 0.f, pos_[j + 2][0], pos_[j + 2][1], pos_[j + 2][2], 0.f);
                // auto dr3_ = _mm256_setr_ps(pos_[j + 3][0], pos_[j + 3][1], pos_[j + 3][2], 0.f, pos_[j + 3][0], pos_[j + 3][1], pos_[j + 3][2], 0.f);

                // dr0_ = _mm256_sub_ps(dr0_, x);
                // dr1_ = _mm256_sub_ps(dr1_, x);
                // dr2_ = _mm256_sub_ps(dr2_, x);
                // dr3_ = _mm256_sub_ps(dr3_, x);

                dr[0][0][0] = pos_[j][0] - pos_i[0];
                dr[0][0][1] = pos_[j][1] - pos_i[1];
                dr[0][0][2] = pos_[j][2] - pos_i[2];
                dr[0][1][0] = pos_[j][0] - pos_i_1[0];
                dr[0][1][1] = pos_[j][1] - pos_i_1[1];
                dr[0][1][2] = pos_[j][2] - pos_i_1[2];

                dr[1][0][0] = pos_[j + 1][0] - pos_i[0];
                dr[1][0][1] = pos_[j + 1][1] - pos_i[1];
                dr[1][0][2] = pos_[j + 1][2] - pos_i[2];
                dr[1][1][0] = pos_[j + 1][0] - pos_i_1[0];
                dr[1][1][1] = pos_[j + 1][1] - pos_i_1[1];
                dr[1][1][2] = pos_[j + 1][2] - pos_i_1[2];

                // dr[2][0][0] = pos_[j + 2][0] - pos_i[0];
                // dr[2][0][1] = pos_[j + 2][1] - pos_i[1];
                // dr[2][0][2] = pos_[j + 2][2] - pos_i[2];
                // dr[2][1][0] = pos_[j + 2][0] - pos_i_1[0];
                // dr[2][1][1] = pos_[j + 2][1] - pos_i_1[1];
                // dr[2][1][2] = pos_[j + 2][2] - pos_i_1[2];

                // dr[3][0][0] = pos_[j + 3][0] - pos_i[0];
                // dr[3][0][1] = pos_[j + 3][1] - pos_i[1];
                // dr[3][0][2] = pos_[j + 3][2] - pos_i[2];
                // dr[3][1][0] = pos_[j + 3][0] - pos_i_1[0];
                // dr[3][1][1] = pos_[j + 3][1] - pos_i_1[1];
                // dr[3][1][2] = pos_[j + 3][2] - pos_i_1[2];

                {
                    auto r2_ = soft_8;

                    {
                        auto drT_0 = _mm256_setr_ps(dr[0][0][0], dr[0][1][0], dr[1][0][0], dr[1][1][0], 0.f, 0.f, 0.f, 0.f);
                        auto drT_1 = _mm256_setr_ps(dr[0][0][1], dr[0][1][1], dr[1][0][1], dr[1][1][1], 0.f, 0.f, 0.f, 0.f);
                        auto drT_2 = _mm256_setr_ps(dr[0][0][2], dr[0][1][2], dr[1][0][2], dr[1][1][2], 0.f, 0.f, 0.f, 0.f);
                        auto drT_3 = _mm256_setr_ps(dr[0][0][3], dr[0][1][3], dr[1][0][3], dr[1][1][3], 0.f, 0.f, 0.f, 0.f);

                        // auto drT_0 = _mm256_setr_ps(dr[0][0][0], dr[0][1][0], dr[1][0][0], dr[1][1][0], dr[2][0][0], dr[2][1][0], dr[3][0][0], dr[3][1][0]);
                        // auto drT_1 = _mm256_setr_ps(dr[0][0][1], dr[0][1][1], dr[1][0][1], dr[1][1][1], dr[2][0][1], dr[2][1][1], dr[3][0][1], dr[3][1][1]);
                        // auto drT_2 = _mm256_setr_ps(dr[0][0][2], dr[0][1][2], dr[1][0][2], dr[1][1][2], dr[2][0][2], dr[2][1][2], dr[3][0][2], dr[3][1][2]);
                        // auto drT_3 = _mm256_setr_ps(dr[0][0][3], dr[0][1][3], dr[1][0][3], dr[1][1][3], dr[2][0][3], dr[2][1][3], dr[3][0][3], dr[3][1][3]);

                        drT_0 = _mm256_mul_ps(drT_0, drT_0);
                        drT_1 = _mm256_mul_ps(drT_1, drT_1);
                        drT_2 = _mm256_mul_ps(drT_2, drT_2);
                        drT_3 = _mm256_mul_ps(drT_3, drT_3);

                        r2_ = _mm256_add_ps(r2_, drT_0);
                        r2_ = _mm256_add_ps(r2_, drT_1);
                        r2_ = _mm256_add_ps(r2_, drT_2);
                        r2_ = _mm256_add_ps(r2_, drT_3);
                    }

                    const auto r_ = _mm256_sqrt_ps(r2_);

                    auto m_r4_ = _mm256_setr_ps(pos_[j][3], pos_[j][3], pos_[j + 1][3], pos_[j + 1][3], 0.f, 0.f, 0.f, 0.f);
                    // auto m_r4_ = _mm256_setr_ps(pos_[j][3], pos_[j][3], pos_[j + 1][3], pos_[j + 1][3], pos_[j + 2][3], pos_[j + 2][3], 0.f, 0.f);
                    // auto m_r4_ = _mm256_setr_ps(pos_[j][3], pos_[j][3], pos_[j + 1][3], pos_[j + 1][3], pos_[j + 2][3], pos_[j + 2][3], pos_[j + 3][3], pos_[j + 3][3]);

                    m_r4_ = _mm256_div_ps(m_r4_, _mm256_mul_ps(r2_, r2_));

                    auto s_ = _mm256_mul_ps(r_, m_r4_);

                    _mm256_store_ps(s.data(), s_);
                }

                auto dr0_ = _mm256_load_ps(dr[0][0].data());
                auto dr1_ = _mm256_load_ps(dr[1][0].data());
                // auto dr2_ = _mm256_load_ps(dr[2][0].data());
                // auto dr3_ = _mm256_load_ps(dr[3][0].data());

                {
                    auto s0_ = _mm256_setr_ps(s[0], s[0], s[0], s[0], s[1], s[1], s[1], s[1]);
                    auto s1_ = _mm256_setr_ps(s[2], s[2], s[2], s[2], s[3], s[3], s[3], s[3]);
                    // auto s2_ = _mm256_setr_ps(s[4], s[4], s[4], s[4], s[5], s[5], s[5], s[5]);
                    // auto s3_ = _mm256_setr_ps(s[6], s[6], s[6], s[6], s[7], s[7], s[7], s[7]);

                    dr0_ = _mm256_mul_ps(dr0_, s0_);
                    dr1_ = _mm256_mul_ps(dr1_, s1_);
                    // dr2_ = _mm256_mul_ps(dr2_, s2_);
                    // dr3_ = _mm256_mul_ps(dr3_, s3_);
                }

                acc = _mm256_add_ps(acc, dr0_);
                acc = _mm256_add_ps(acc, dr1_);
                // acc = _mm256_add_ps(acc, dr2_);
                // acc = _mm256_add_ps(acc, dr3_);

                // r_01  [3 FLOPS]
                // const auto dr = std::array{pos_[j][0] - pos_i[0], pos_[j][1] - pos_i[1], pos_[j][2] - pos_i[2]};

                // NOTE: sqrt and / (and r2) are long calculations
                // breaking dependency chain by allowing them to be calculated separately increases performance ~10%

                // const auto dx2 = dr[0] * dr[0];
                // const auto dy2 = dr[1] * dr[1];
                // const auto dz2 = dr[2] * dr[2];

                // const auto dx2_dy2   = dx2 + dy2;
                // const auto dz2_soft2 = dz2 + softening_squared;

                // d^2 + e^2 [6 FLOPS]
                // const auto r2 = dx2_dy2 + dz2_soft2;

                // 1 FLOP
                // const auto r = std::sqrt(r2);

                // 2 FLOPs
                // const auto m_r4 = pos_[j][3] / (r2 * r2);

                // 1 FLOP
                // const auto s = m_r4 * r;

                // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
                // acc[0] += dr[0] * s;
                // acc[1] += dr[1] * s;
                // acc[2] += dr[2] * s;
            }

            acc = _mm256_mul_ps(acc, dt_8);
            _mm256_store_ps(dv_[i].data(), acc);

            // dv_[i][0] = acc[0] * dt;
            // dv_[i][1] = acc[1] * dt;
            // dv_[i][2] = acc[2] * dt;
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < nb_bodies; i++) {
            auto       acc   = std::array<T, 3>{0, 0, 0};
            const auto pos_i = std::array<T, 3>{pos_[i][0], pos_[i][1], pos_[i][2]};

            // We unroll this loop 4X for a small performance boost.
            for (auto j = 0; j < nb_bodies; j += 4) {
                interaction(acc, pos_i, pos_[j], softening_squared);
                interaction(acc, pos_i, pos_[j + 1], softening_squared);
                interaction(acc, pos_i, pos_[j + 2], softening_squared);
                interaction(acc, pos_i, pos_[j + 3], softening_squared);
            }

            dv_[i][0] = acc[0] * dt;
            dv_[i][1] = acc[1] * dt;
            dv_[i][2] = acc[2] * dt;
        }
    }

    const auto damping = damping_;

    if constexpr (std::is_same_v<T, float>) {
        const auto dt_8      = _mm256_set_ps(dt, dt, dt, dt, dt, dt, dt, dt);
        const auto damping_8 = _mm256_set_ps(damping, damping, damping, damping, damping, damping, damping, damping);

#pragma omp parallel for
        for (auto i = std::size_t{0}; i < nb_bodies; i += 2) {
            // new velocity = old velocity + acceleration * dt
            auto v_2 = _mm256_load_ps(vel_[i].data());

            v_2 = _mm256_add_ps(v_2, _mm256_load_ps(dv_[i].data()));

            v_2 = _mm256_mul_ps(v_2, damping_8);

            _mm256_store_ps(vel_[i].data(), v_2);

            v_2 = _mm256_mul_ps(v_2, dt_8);

            auto p_2 = _mm256_load_ps(pos_[i].data());

            p_2 = _mm256_add_ps(p_2, v_2);

            _mm256_store_ps(pos_[i].data(), p_2);
        }
    } else {
#pragma omp parallel for
        for (auto i = std::size_t{0}; i < nb_bodies; ++i) {
            // new velocity = old velocity + acceleration * dt
            vel_[i][0] = (vel_[i][0] + dv_[i][0]) * damping;
            vel_[i][1] = (vel_[i][1] + dv_[i][1]) * damping;
            vel_[i][2] = (vel_[i][2] + dv_[i][2]) * damping;

            // new position = old position + velocity * dt
            pos_[i][0] += vel_[i][0] * dt;
            pos_[i][1] += vel_[i][1] * dt;
            pos_[i][2] += vel_[i][2] * dt;
        }
    }

    const auto cycles_per_interaction = 2 * (rdtsc() - initial_cycle_count) / (nb_bodies * (nb_bodies - 1));

    std::println("Cycles per interaction: {}", cycles_per_interaction);
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
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

#ifdef OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <span>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using std::ranges::copy;

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params) : m_numBodies(nb_bodies), m_pos(m_numBodies * 4, T{0}), m_vel(m_numBodies * 4, T{0}), m_damping(params.m_damping) {
    setSoftening(params.m_softening);

    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCPU<T>::BodySystemCPU(std::size_t nb_bodies, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : m_numBodies(nb_bodies), m_pos(std::move(positions)), m_vel(std::move(velocities)), m_damping(params.m_damping) {
    assert(m_pos.size() == m_numBodies * 4);
    assert(m_vel.size() == m_pos.size());

    setSoftening(params.m_softening);
}

template <std::floating_point T> auto BodySystemCPU<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, m_pos, m_vel, params.m_clusterScale, params.m_velocityScale);
}

template <std::floating_point T> auto BodySystemCPU<T>::update_params(const NBodyParams& active_params) noexcept -> void {
    setSoftening(active_params.m_softening);
    m_damping = active_params.m_damping;
}

template <std::floating_point T> auto BodySystemCPU<T>::set_position(std::span<const T> data) noexcept -> void {
    assert(data.size() == m_pos.size());
    copy(data, m_pos.begin());
}
template <std::floating_point T> auto BodySystemCPU<T>::set_velocity(std::span<const T> data) noexcept -> void {
    assert(data.size() == m_vel.size());
    copy(data, m_vel.begin());
}

template <std::floating_point T> auto bodyBodyInteraction(T accel[3], const T posMass0[4], const T posMass1[4], T softeningSquared) noexcept -> void {
    T r[3];

    // r_01  [3 FLOPS]
    r[0] = posMass1[0] - posMass0[0];
    r[1] = posMass1[1] - posMass0[1];
    r[2] = posMass1[2] - posMass0[2];

    // d^2 + e^2 [6 FLOPS]
    T distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist     = (T)1.0 / std::sqrt(distSqr);
    T invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = posMass1[3] * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}

template <std::floating_point T> auto BodySystemCPU<T>::_computeNBodyGravitation() noexcept -> void {
#ifdef OPENMP
#pragma omp parallel for
#endif

    for (int i = 0; i < m_numBodies; i++) {
        int indexForce = 3 * i;

        T acc[3] = {0, 0, 0};

        // We unroll this loop 4X for a small performance boost.
        int j = 0;

        while (j < m_numBodies) {
            bodyBodyInteraction<T>(acc, &m_pos[4 * i], &m_pos[4 * j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4 * i], &m_pos[4 * j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4 * i], &m_pos[4 * j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4 * i], &m_pos[4 * j], m_softeningSquared);
            j++;
        }

        m_force[indexForce]     = acc[0];
        m_force[indexForce + 1] = acc[1];
        m_force[indexForce + 2] = acc[2];
    }
}

template <std::floating_point T> auto BodySystemCPU<T>::update(T deltaTime) noexcept -> void {
    _computeNBodyGravitation();

#ifdef OPENMP
#pragma omp parallel for
#endif

    for (int i = 0; i < m_numBodies; ++i) {
        int index      = 4 * i;
        int indexForce = 3 * i;

        T pos[3], vel[3], force[3];
        pos[0]    = m_pos[index + 0];
        pos[1]    = m_pos[index + 1];
        pos[2]    = m_pos[index + 2];
        T invMass = m_pos[index + 3];

        vel[0] = m_vel[index + 0];
        vel[1] = m_vel[index + 1];
        vel[2] = m_vel[index + 2];

        force[0] = m_force[indexForce + 0];
        force[1] = m_force[indexForce + 1];
        force[2] = m_force[indexForce + 2];

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime
        vel[0] += (force[0] * invMass) * deltaTime;
        vel[1] += (force[1] * invMass) * deltaTime;
        vel[2] += (force[2] * invMass) * deltaTime;

        vel[0] *= m_damping;
        vel[1] *= m_damping;
        vel[2] *= m_damping;

        // new position = old position + velocity * deltaTime
        pos[0] += vel[0] * deltaTime;
        pos[1] += vel[1] * deltaTime;
        pos[2] += vel[2] * deltaTime;

        m_pos[index + 0] = pos[0];
        m_pos[index + 1] = pos[1];
        m_pos[index + 2] = pos[2];

        m_vel[index + 0] = vel[0];
        m_vel[index + 1] = vel[1];
        m_vel[index + 2] = vel[2];
    }
}

template BodySystemCPU<float>;
template BodySystemCPU<double>;
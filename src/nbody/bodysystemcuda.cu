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

#include "block_size.hpp"
#include "vec.hpp"

// CUDA standard includes

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <concepts>
#include <span>
#include <stdexcept>
#include <string>

#include <cmath>

namespace cg = cooperative_groups;

template <std::floating_point T> __constant__ T softening_squared;

template <std::floating_point T> auto set_softening_squared(T softening_sq) -> void {
    const auto result = cudaMemcpyToSymbol(softening_squared<T>, &softening_sq, sizeof(T), 0, cudaMemcpyHostToDevice);

    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(result));
    }
}

__device__ auto rsqrt_(float x) -> float {
    return rsqrtf(x);
}

__device__ auto rsqrt_(double x) -> double {
    return rsqrt(x);
}

template <std::floating_point T> __device__ auto body_body_interaction(vec3<T>& ai, const vec4<T>& bi, const vec4<T>& bj) -> void {
    // [3 FLOPS]
    const auto dr = vec3<T>{bj.x - bi.x, bj.y - bi.y, bj.z - bi.z};

    // [6 FLOPS]
    const auto r2 = (softening_squared<T> + dr.x * dr.x) + (dr.y * dr.y + dr.z * dr.z);

    // [4 FLOP]
    const auto m_r3 = (bj.w / (r2 * r2)) * std::sqrt(r2);

    // [6 FLOPS]
    ai.x += dr.x * m_r3;
    ai.y += dr.y * m_r3;
    ai.z += dr.z * m_r3;
}

template <std::floating_point T> __device__ auto compute_body_accel(const vec4<T>& body_pos, const vec4<T>* positions, const cg::thread_block& cta) -> vec3<T> {
    __shared__ vec4<T> shared_pos[block_size];

    auto acc = vec3<T>{0, 0, 0};

    for (auto tile = 0; tile < gridDim.x; ++tile) {
        shared_pos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

        for (auto counter = 0u; counter < blockDim.x; ++counter) {
            body_body_interaction<T>(acc, body_pos, shared_pos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}

template <std::floating_point T> __global__ void integrate_bodies(vec4<T>* __restrict__ new_pos, const vec4<T>* __restrict__ old_pos, vec4<T>* vel, unsigned int nb_bodies, T dt, T damping) {
    // Handle to thread block group
    const auto cta   = cg::this_thread_block();
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nb_bodies) {
        return;
    }

    auto position = old_pos[index];

    const auto accel = compute_body_accel<T>(position, old_pos, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * dt
    // note we factor out the body's mass from the equation, here and in
    // body_body_interaction
    // (because they cancel out).  Thus here force == acceleration
    auto velocity = vel[index];

    velocity.x += accel.x * dt;
    velocity.y += accel.y * dt;
    velocity.z += accel.z * dt;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * dt
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;

    // store new position and velocity
    new_pos[index] = position;
    vel[index]     = velocity;
}

template <std::floating_point T> void integrateNbodySystem(T* new_positions, const T* old_positions, T* velocities, T dt, T damping, unsigned int nb_bodies) {
    const auto nb_blocks = (nb_bodies + (block_size - 1)) / block_size;

    integrate_bodies<T><<<nb_blocks, block_size>>>(reinterpret_cast<vec4<T>*>(new_positions), reinterpret_cast<const vec4<T>*>(old_positions), reinterpret_cast<vec4<T>*>(velocities), nb_bodies, dt, damping);

    const auto err = cudaGetLastError();

    if (cudaSuccess != err) {
        throw std::runtime_error(std::string{"getLastCudaError() CUDA error : Kernel execution failed : ("} + std::to_string(static_cast<int>(err)) + ") " + cudaGetErrorString(err));
    }
}

template void integrateNbodySystem<float>(float* new_positions, const float* old_positions, float* velocities, float dt, float damping, unsigned int nb_bodies);
template void integrateNbodySystem<double>(double* new_positions, const double* old_positions, double* velocities, double dt, double damping, unsigned int nb_bodies);

template auto set_softening_squared<float>(float softening_sq) -> void;
template auto set_softening_squared<double>(double softening_sq) -> void;

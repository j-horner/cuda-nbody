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

#include "helper_cuda.hpp"
#include "vec.hpp"

// CUDA standard includes

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <span>

#include <cmath>

namespace cg = cooperative_groups;

__constant__ float  softeningSquared;
__constant__ double softeningSquared_fp64;

cudaError_t setSofteningSquared(float softeningSq) {
    return cudaMemcpyToSymbol(softeningSquared, &softeningSq, sizeof(float), 0, cudaMemcpyHostToDevice);
}

cudaError_t setSofteningSquared(double softeningSq) {
    return cudaMemcpyToSymbol(softeningSquared_fp64, &softeningSq, sizeof(double), 0, cudaMemcpyHostToDevice);
}

template <class T> struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

template <typename T> __device__ T rsqrt_T(T x) {
    return rsqrt(x);
}

template <> __device__ float rsqrt_T<float>(float x) {
    return rsqrtf(x);
}

template <> __device__ double rsqrt_T<double>(double x) {
    return rsqrt(x);
}

// Macros to simplify shared memory addressing
#define SX(i) sharedPos[i + blockDim.x * threadIdx.y]
// This macro is only used when multithreadBodies is true (below)
#define SX_SUM(i, j) sharedPos[i + blockDim.x * j]

template <typename T> __device__ T getSofteningSquared() {
    return softeningSquared;
}
template <> __device__ double getSofteningSquared<double>() {
    return softeningSquared_fp64;
}

/* template <typename T> struct DeviceData {
    T*           pos[2];    // mapped host pointers
    T*           vel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int nb_bodies;
};*/

template <typename T> __device__ vec3<T> bodyBodyInteraction(vec3<T> ai, vec4<T> bi, vec4<T> bj) {
    vec3<T> r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    T distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += getSofteningSquared<T>();

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist     = rsqrt_T(distSqr);
    T invDistCube = invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

template <typename T> __device__ vec3<T> computeBodyAccel(vec4<T> bodyPos, const vec4<T>* positions, int numTiles, cg::thread_block cta) {
    vec4<T>* sharedPos = SharedMemory<vec4<T>>();

    vec3<T> acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < numTiles; tile++) {
        sharedPos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];

        cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128

        for (unsigned int counter = 0; counter < blockDim.x; counter++) {
            acc = bodyBodyInteraction<T>(acc, bodyPos, sharedPos[counter]);
        }

        cg::sync(cta);
    }

    return acc;
}

template <typename T> __global__ void integrateBodies(vec4<T>* __restrict__ newPos, const vec4<T>* __restrict__ oldPos, vec4<T>* vel, unsigned int deviceNumBodies, float deltaTime, float damping, int numTiles) {
    // Handle to thread block group
    cg::thread_block cta   = cg::this_thread_block();
    int              index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= deviceNumBodies) {
        return;
    }

    vec4<T> position = oldPos[index];

    vec3<T> accel = computeBodyAccel<T>(position, oldPos, numTiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * deltaTime
    // note we factor out the body's mass from the equation, here and in
    // bodyBodyInteraction
    // (because they cancel out).  Thus here force == acceleration
    vec4<T> velocity = vel[index];

    velocity.x += accel.x * deltaTime;
    velocity.y += accel.y * deltaTime;
    velocity.z += accel.z * deltaTime;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * deltaTime
    position.x += velocity.x * deltaTime;
    position.y += velocity.y * deltaTime;
    position.z += velocity.z * deltaTime;

    // store new position and velocity
    newPos[index] = position;
    vel[index]    = velocity;
}

template <typename T> void integrateNbodySystem(T* new_positions, const T* old_positions, T* velocities, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize) {
    {
        const auto numBlocks     = (numBodies + blockSize - 1) / blockSize;
        const auto sharedMemSize = blockSize * 4 * sizeof(T);    // 4 floats for pos

        integrateBodies<T><<<numBlocks, blockSize, sharedMemSize>>>(
            reinterpret_cast<vec4<T>*>(new_positions),
            reinterpret_cast<const vec4<T>*>(old_positions),
            reinterpret_cast<vec4<T>*>(velocities),
            numBodies,
            deltaTime,
            damping,
            numBlocks);
    }

    // check if kernel invocation generated an error
    const auto err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                __FILE__,
                __LINE__,
                "Kernel execution failed",
                static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Explicit specializations needed to generate code
template void integrateNbodySystem<float>(float* new_positions, const float* old_positions, float* velocities, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize);

template void integrateNbodySystem<double>(double* new_positions, const double* old_positions, double* velocities, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize);

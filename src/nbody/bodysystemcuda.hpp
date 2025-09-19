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

#include "nbody_config.hpp"

#include <cuda_runtime.h>

#include <array>
#include <filesystem>
#include <span>
#include <vector>

class ComputeCUDA;
struct NBodyParams;

template <typename T> struct DeviceData {
    T*           dPos[2];    // mapped host pointers
    T*           dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <std::floating_point T> class BodySystemCUDA {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = false;

    BodySystemCUDA(const ComputeCUDA& compute, unsigned int numDevices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params);
    BodySystemCUDA(const ComputeCUDA& compute, unsigned int numDevices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    ~BodySystemCUDA() noexcept;

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto update(T deltaTime) -> void;

    auto update_params(const NBodyParams& active_params) -> void;

    auto get_position() const -> std::span<const T>;
    auto get_velocity() const -> std::span<const T>;

    auto set_position(std::span<const T> data) -> void;
    auto set_velocity(std::span<const T> data) -> void;

    auto getCurrentReadBuffer() const noexcept { return m_pbo[m_currentRead]; }

 private:    // methods
    auto setSoftening(T softening) -> void;

    auto _initialize(int numBodies) -> void;
    auto _finalize() noexcept -> void;

    unsigned int m_numBodies;
    unsigned int m_numDevices;
    bool         m_bInitialized = false;
    int          m_devID;

    // Host data
    std::array<T*, 2> m_hPos{nullptr, nullptr};
    T*                m_hVel = nullptr;

    std::vector<DeviceData<T>> m_deviceData;

    std::vector<T> m_hPos_vec = std::vector(m_numBodies * 4, T{0});
    std::vector<T> m_hVel_vec = std::vector(m_numBodies * 4, T{0});

    bool         m_bUsePBO;
    bool         m_bUseSysMem;
    bool         m_bUseP2P;
    unsigned int m_SMVersion;

    T m_damping = 0.995f;

    unsigned int          m_pbo[2];
    cudaGraphicsResource* m_pGRes[2];
    unsigned int          m_currentRead  = 0u;
    unsigned int          m_currentWrite = 1u;

    unsigned int m_blockSize;
};

extern template BodySystemCUDA<float>;
extern template BodySystemCUDA<double>;
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

#include "device_data.hpp"
#include "nbody_config.hpp"

#include <cuda/api.hpp>
#include <cuda_runtime.h>

#include <array>
#include <filesystem>
#include <span>
#include <vector>

class ComputeCUDA;
struct NBodyParams;

// CUDA BodySystem: runs on the GPU
template <std::floating_point T> class BodySystemCUDA {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = false;

    BodySystemCUDA(const ComputeCUDA& compute, unsigned int nb_devices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params);
    BodySystemCUDA(const ComputeCUDA& compute, unsigned int nb_devices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto update(T deltaTime) -> void;

    auto update_params(const NBodyParams& active_params) -> void;

    auto get_position() const -> std::span<const T>;
    auto get_velocity() const -> std::span<const T>;

    auto set_position(std::span<const T> data) -> void;
    auto set_velocity(std::span<const T> data) -> void;

    auto getCurrentReadBuffer() const noexcept { return pbo_[current_read_]; }

    ~BodySystemCUDA() noexcept;

 private:    // methods
    auto setSoftening(T softening) -> void;

    auto _initialize(unsigned int nb_devices) -> void;

    unsigned int nb_bodies_;
    int          dev_id_;

    // Host data
    std::array<T*, 2> host_pos_{nullptr, nullptr};
    T*                host_vel_ = nullptr;

    std::vector<DeviceData<T>> device_data_;

    std::vector<T> host_pos_vec_ = std::vector(nb_bodies_ * 4, T{0});
    std::vector<T> host_vel_vec_ = std::vector(nb_bodies_ * 4, T{0});

    bool         use_pbo_;
    bool         use_sys_mem_;
    bool         use_p2p_;
    unsigned int sm_version_;

    T damping_ = 0.995f;

    unsigned int          pbo_[2];
    cudaGraphicsResource* graphics_resource_[2];
    unsigned int          current_read_  = 0u;
    unsigned int          current_write_ = 1u;

    unsigned int block_size_;
};

extern template BodySystemCUDA<float>;
extern template BodySystemCUDA<double>;
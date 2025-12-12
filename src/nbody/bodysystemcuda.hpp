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

struct NBodyParams;

template <std::floating_point T> class BodySystemCUDA {
 public:
    using Type                    = T;
    constexpr static auto use_cpu = false;

    BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto virtual getCurrentReadBuffer() const noexcept -> unsigned int { return 0u; }

    auto virtual get_position() const -> std::span<const T> = 0;
    auto virtual get_velocity() const -> std::span<const T> = 0;

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto virtual update(T deltaTime) -> void = 0;

    auto update_params(const NBodyParams& active_params) -> void;

    auto virtual set_position(std::span<const T> data) -> void = 0;
    auto virtual set_velocity(std::span<const T> data) -> void = 0;

    virtual ~BodySystemCUDA() = default;

 protected:
    unsigned int nb_bodies_;

    std::vector<T> host_pos_vec_;
    std::vector<T> host_vel_vec_;

    T damping_ = 0.995f;

    unsigned int current_read_  = 0u;
    unsigned int current_write_ = 1u;

    unsigned int block_size_;

 private:
    auto setSoftening(T softening) -> void;
};

///
/// @brief  The default CUDA implementation. No graphics interop, no host memory.
///
template <std::floating_point T> class BodySystemCUDADefault : public BodySystemCUDA<T> {
 public:
    BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    ~BodySystemCUDADefault() noexcept;

 private:
    auto _initialize() -> void;

    // Host data
    mutable std::vector<T> host_pos_;
    mutable std::vector<T> host_vel_;

    // Device data
    std::array<T*, 2> device_pos_{nullptr, nullptr};
    T*                device_vel_ = nullptr;
};

///
/// @brief  The CUDA implementation using OpenGL interop. Some GPU buffers are allocated by OpenGL.
///
template <std::floating_point T> class BodySystemCUDAGraphics : public BodySystemCUDA<T> {
 public:
    BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    auto getCurrentReadBuffer() const noexcept -> unsigned int final { return pbo_[BodySystemCUDA<T>::current_read_]; }

    ~BodySystemCUDAGraphics() noexcept;

 private:
    auto _initialize() -> void;

    // Host data
    mutable std::vector<T> host_pos_;
    mutable std::vector<T> host_vel_;

    // Device data
    std::array<T*, 2> device_pos_{nullptr, nullptr};
    T*                device_vel_ = nullptr;

    unsigned int          pbo_[2];
    cudaGraphicsResource* graphics_resource_[2];
};

///
/// @brief The CUDA implementation with host memory mapped to device memory.
///
template <std::floating_point T> class BodySystemCUDAHostMemory : public BodySystemCUDA<T> {
 public:
    BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    ~BodySystemCUDAHostMemory() noexcept;

 private:
    auto _initialize() -> void;

    // Host data
    std::array<T*, 2> host_pos_{nullptr, nullptr};
    T*                host_vel_ = nullptr;

    // Device data
    std::array<T*, 2> device_pos_{nullptr, nullptr};
    T*                device_vel_ = nullptr;
};

extern template BodySystemCUDA<float>;
extern template BodySystemCUDA<double>;

extern template BodySystemCUDADefault<float>;
extern template BodySystemCUDADefault<double>;

extern template BodySystemCUDAGraphics<float>;
extern template BodySystemCUDAGraphics<double>;

extern template BodySystemCUDAHostMemory<float>;
extern template BodySystemCUDAHostMemory<double>;
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

#include "compute_cuda.hpp"
#include "gl_includes.hpp"
#include "helper_cuda.hpp"
#include "params.hpp"
#include "randomise_bodies.hpp"
#include "vec.hpp"

#include <cuda_gl_interop.h>

#include <algorithm>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

template <typename T>
void integrateNbodySystem(std::array<T*, 2>& positions, T* velocities, cudaGraphicsResource** pgres, unsigned int currentRead, float deltaTime, float damping, unsigned int numBodies, int blockSize, bool bUsePBO);

cudaError_t setSofteningSquared(float softeningSq);
cudaError_t setSofteningSquared(double softeningSq);

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params)
    : BodySystemCUDA(nb_bodies, blockSize, params, std::vector<T>(nb_bodies * 4, T{0}), std::vector<T>(nb_bodies * 4, T{0})) {}

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : nb_bodies_(nb_bodies), block_size_(blockSize), host_pos_vec_(std::move(positions)), host_vel_vec_(std::move(velocities)), damping_(params.damping) {
    setSoftening(params.softening);
}

template <std::floating_point T> auto BodySystemCUDA<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, host_pos_vec_, host_vel_vec_, params.cluster_scale, params.velocity_scale);
    set_position(host_pos_vec_);
    set_velocity(host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDA<T>::update_params(const NBodyParams& active_params) -> void {
    setSoftening(active_params.softening);
    damping_ = active_params.damping;
}

template <std::floating_point T> auto BodySystemCUDA<T>::setSoftening(T softening) -> void {
    const auto softeningSq = softening * softening;

    checkCudaErrors(setSofteningSquared(softeningSq));
}

template <std::floating_point T> BodySystemCUDADefault<T>::BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    _initialize();

    BodySystemCUDADefault<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDADefault<T>::BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : BodySystemCUDA<T>(nb_bodies, blockSize, params, std::move(positions), std::move(velocities)) {
    assert(this->host_pos_vec_.size() == 4 * this->nb_bodies_);
    assert(this->host_vel_vec_.size() == 4 * this->nb_bodies_);

    _initialize();

    set_position(this->host_pos_vec_);
    set_velocity(this->host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::_initialize() -> void {
    host_pos_.resize(this->nb_bodies_ * 4, 0);
    host_vel_.resize(this->nb_bodies_ * 4, 0);

    const auto memSize = sizeof(T) * 4 * this->nb_bodies_;

    checkCudaErrors(cudaMalloc((void**)&device_pos_[0], memSize));
    checkCudaErrors(cudaMalloc((void**)&device_pos_[1], memSize));
    checkCudaErrors(cudaMalloc((void**)&device_vel_, memSize));
}

template <std::floating_point T> BodySystemCUDADefault<T>::~BodySystemCUDADefault() noexcept {
    checkCudaErrors(cudaFree((void**)device_pos_[0]));
    checkCudaErrors(cudaFree((void**)device_pos_[1]));
    checkCudaErrors(cudaFree((void**)device_vel_));
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::update(T deltaTime) -> void {
    integrateNbodySystem<T>(device_pos_, device_vel_, nullptr, this->current_read_, (float)deltaTime, (float)this->damping_, this->nb_bodies_, this->block_size_, false);

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::get_position() const -> std::span<const T> {
    checkCudaErrors(cudaMemcpy(host_pos_.data(), device_pos_[this->current_read_], this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

    return host_pos_;
}
template <std::floating_point T> auto BodySystemCUDADefault<T>::get_velocity() const -> std::span<const T> {
    checkCudaErrors(cudaMemcpy(host_vel_.data(), device_vel_, this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

    return host_vel_;
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::set_position(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    checkCudaErrors(cudaMemcpy(device_pos_[this->current_read_], data.data(), this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::set_velocity(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    checkCudaErrors(cudaMemcpy(device_vel_, data.data(), this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
}

template <std::floating_point T> BodySystemCUDAGraphics<T>::BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    _initialize();

    BodySystemCUDAGraphics<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDAGraphics<T>::BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : BodySystemCUDA<T>(nb_bodies, blockSize, params, std::move(positions), std::move(velocities)) {
    assert(this->host_pos_vec_.size() == 4 * this->nb_bodies_);
    assert(this->host_vel_vec_.size() == 4 * this->nb_bodies_);

    _initialize();

    set_position(this->host_pos_vec_);
    set_velocity(this->host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::_initialize() -> void {
    const auto memSize = sizeof(T) * 4 * this->nb_bodies_;

    host_pos_.resize(this->nb_bodies_ * 4, 0);
    host_vel_.resize(this->nb_bodies_ * 4, 0);

    // create the position pixel buffer objects for rendering
    // we will actually compute directly from this memory in CUDA too
    glGenBuffers(2, (GLuint*)pbo_);

    for (int i = 0; i < 2; ++i) {
        glBindBuffer(GL_ARRAY_BUFFER, pbo_[i]);
        glBufferData(GL_ARRAY_BUFFER, memSize, host_pos_.data(), GL_DYNAMIC_DRAW);

        int size = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

        if ((unsigned)size != memSize) {
            fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!n");
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&graphics_resource_[i], pbo_[i], cudaGraphicsMapFlagsNone));
    }

    checkCudaErrors(cudaMalloc((void**)&device_vel_, memSize));
}

template <std::floating_point T> BodySystemCUDAGraphics<T>::~BodySystemCUDAGraphics() noexcept {
    checkCudaErrors(cudaFree((void**)device_vel_));

    checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[0]));
    checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[1]));
    glDeleteBuffers(2, (const GLuint*)pbo_);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::update(T deltaTime) -> void {
    integrateNbodySystem<T>(device_pos_, device_vel_, graphics_resource_, this->current_read_, (float)deltaTime, (float)this->damping_, this->nb_bodies_, this->block_size_, true);

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::get_position() const -> std::span<const T> {
    const auto ddata = device_pos_[this->current_read_];

    {
        auto pgres = graphics_resource_[this->current_read_];

        checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
        checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
        size_t bytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ddata, &bytes, pgres));

        checkCudaErrors(cudaMemcpy(host_pos_.data(), ddata, this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
    }

    return host_pos_;
}
template <std::floating_point T> auto BodySystemCUDAGraphics<T>::get_velocity() const -> std::span<const T> {
    checkCudaErrors(cudaMemcpy(host_vel_.data(), device_vel_, this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

    return host_vel_;
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::set_position(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    glBindBuffer(GL_ARRAY_BUFFER, pbo_[this->current_read_]);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(T) * this->nb_bodies_, data.data());

    int size = 0;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

    if ((unsigned)size != 4 * (sizeof(T) * this->nb_bodies_)) {
        fprintf(stderr, "WARNING: Pixel Buffer Object download failed!n");
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::set_velocity(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    checkCudaErrors(cudaMemcpy(device_vel_, data.data(), this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
}

template <std::floating_point T> BodySystemCUDAHostMemory<T>::BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    _initialize();

    BodySystemCUDAHostMemory<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDAHostMemory<T>::BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : BodySystemCUDA<T>(nb_bodies, blockSize, params, std::move(positions), std::move(velocities)) {
    assert(this->host_pos_vec_.size() == 4 * this->nb_bodies_);
    assert(this->host_vel_vec_.size() == 4 * this->nb_bodies_);

    _initialize();

    set_position(this->host_pos_vec_);
    set_velocity(this->host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::_initialize() -> void {
    const auto memSize = sizeof(T) * 4 * this->nb_bodies_;

    checkCudaErrors(cudaHostAlloc((void**)&host_pos_[0], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&host_pos_[1], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
    checkCudaErrors(cudaHostAlloc((void**)&host_vel_, memSize, cudaHostAllocMapped | cudaHostAllocPortable));

    memset(host_pos_[0], 0, memSize);
    memset(host_pos_[1], 0, memSize);
    memset(host_vel_, 0, memSize);

    checkCudaErrors(cudaHostGetDevicePointer((void**)&device_pos_[0], (void*)host_pos_[0], 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&device_pos_[1], (void*)host_pos_[1], 0));
    checkCudaErrors(cudaHostGetDevicePointer((void**)&device_vel_, (void*)host_vel_, 0));
}

template <std::floating_point T> BodySystemCUDAHostMemory<T>::~BodySystemCUDAHostMemory() noexcept {
    checkCudaErrors(cudaFreeHost(host_pos_[0]));
    checkCudaErrors(cudaFreeHost(host_pos_[1]));
    checkCudaErrors(cudaFreeHost(host_vel_));
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::update(T deltaTime) -> void {
    integrateNbodySystem<T>(device_pos_, device_vel_, nullptr, this->current_read_, (float)deltaTime, (float)this->damping_, this->nb_bodies_, this->block_size_, false);

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::get_position() const -> std::span<const T> {
    return {host_pos_[this->current_read_], this->nb_bodies_ * 4};
}
template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::get_velocity() const -> std::span<const T> {
    return {host_vel_, this->nb_bodies_ * 4};
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::set_position(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    memcpy(host_pos_[this->current_read_], data.data(), this->nb_bodies_ * 4 * sizeof(T));
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::set_velocity(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    memcpy(host_vel_, data.data(), this->nb_bodies_ * 4 * sizeof(T));
}

template BodySystemCUDA<float>;
template BodySystemCUDA<double>;

template BodySystemCUDADefault<float>;
template BodySystemCUDADefault<double>;

template BodySystemCUDAGraphics<float>;
template BodySystemCUDAGraphics<double>;

template BodySystemCUDAHostMemory<float>;
template BodySystemCUDAHostMemory<double>;
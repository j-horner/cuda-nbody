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
void integrateNbodySystem(
    std::span<DeviceData<T>> deviceData,
    cudaGraphicsResource**   pgres,
    unsigned int             currentRead,
    float                    deltaTime,
    float                    damping,
    unsigned int             numBodies,
    unsigned int             numDevices,
    int                      blockSize,
    bool                     bUsePBO);

cudaError_t setSofteningSquared(float softeningSq);
cudaError_t setSofteningSquared(double softeningSq);

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(const ComputeCUDA& compute, unsigned int nb_devices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params)
    : nb_bodies_(static_cast<unsigned int>(compute.nb_bodies())), use_pbo_(compute.use_pbo()), use_sys_mem_(compute.use_host_mem()), use_p2p_(useP2P), block_size_(blockSize), dev_id_(deviceId), damping_(params.damping) {
    _initialize(nb_devices);

    setSoftening(params.softening);

    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(const ComputeCUDA& compute, unsigned int nb_devices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : nb_bodies_(static_cast<unsigned int>(compute.nb_bodies())), use_pbo_(compute.use_pbo()), use_sys_mem_(compute.use_host_mem()), use_p2p_(useP2P), block_size_(blockSize), dev_id_(deviceId),
      host_pos_vec_(std::move(positions)), host_vel_vec_(std::move(velocities)), damping_(params.damping) {
    assert(host_pos_vec_.size() == nb_bodies_);
    assert(host_vel_vec_.size() == nb_bodies_);

    _initialize(nb_devices);

    setSoftening(params.softening);

    set_position(host_pos_vec_);
    set_velocity(host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDA<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, host_pos_vec_, host_vel_vec_, params.cluster_scale, params.velocity_scale);
    set_position(host_pos_vec_);
    set_velocity(host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDA<T>::_initialize(unsigned int nb_devices) -> void {
    unsigned int memSize = sizeof(T) * 4 * nb_bodies_;

    device_data_.resize(nb_devices);

    // divide up the workload amongst Devices
    {
        auto weights = std::vector<float>(nb_devices);
        auto nb_sms  = std::vector<int>(nb_devices);
        auto total   = 0.f;

        for (auto i = 0u; i < nb_devices; ++i) {
            cudaDeviceProp props;
            checkCudaErrors(cudaGetDeviceProperties(&props, i));

            // Choose the weight based on the Compute Capability
            // We estimate that a CC2.0 SM is about 4.0x faster than a CC 1.x SM for this application (since a 15-SM GF100 is about 2X faster than a 30-SM GT200).
            nb_sms[i]  = props.multiProcessorCount;
            weights[i] = nb_sms[i] * (props.major >= 2 ? 4.f : 1.f);
            total += weights[i];
        }

        auto offset    = 0u;
        auto remaining = nb_bodies_;

        for (auto i = 0u; i < nb_devices; ++i) {
            auto count = static_cast<unsigned int>((weights[i] / total) * nb_bodies_);
            // Rounding up to nb_sms[i]*256 leads to better GPU utilization _per_ GPU but when using multiple devices, it will lead to the last GPUs not having any work at all which means worse overall performance.
            // unsigned int round = nb_sms[i] * 256;
            constexpr auto round = 256u;

            count = round * ((count + round - 1) / round);
            if (count > remaining) {
                count = remaining;
            }

            remaining -= count;
            device_data_[i].offset    = offset;
            device_data_[i].nb_bodies = count;
            offset += count;
        }

        if (offset < nb_bodies_ - 1) {
            device_data_.back().nb_bodies += nb_bodies_ - offset;
        }
    }

    if (use_sys_mem_) {
        checkCudaErrors(cudaHostAlloc((void**)&host_pos_[0], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc((void**)&host_pos_[1], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc((void**)&host_vel_, memSize, cudaHostAllocMapped | cudaHostAllocPortable));

        memset(host_pos_[0], 0, memSize);
        memset(host_pos_[1], 0, memSize);
        memset(host_vel_, 0, memSize);

        for (unsigned int i = 0; i < nb_devices; i++) {
            if (nb_devices > 1) {
                checkCudaErrors(cudaSetDevice(i));
            }

            checkCudaErrors(cudaEventCreate(&device_data_[i].event));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&device_data_[i].pos[0], (void*)host_pos_[0], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&device_data_[i].pos[1], (void*)host_pos_[1], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&device_data_[i].vel, (void*)host_vel_, 0));
        }
    } else {
        host_pos_[0] = new T[nb_bodies_ * 4];
        host_vel_    = new T[nb_bodies_ * 4];

        memset(host_pos_[0], 0, memSize);
        memset(host_vel_, 0, memSize);

        checkCudaErrors(cudaSetDevice(dev_id_));
        checkCudaErrors(cudaEventCreate(&device_data_[0].event));

        if (use_pbo_) {
            // create the position pixel buffer objects for rendering
            // we will actually compute directly from this memory in CUDA too
            glGenBuffers(2, (GLuint*)pbo_);

            for (int i = 0; i < 2; ++i) {
                glBindBuffer(GL_ARRAY_BUFFER, pbo_[i]);
                glBufferData(GL_ARRAY_BUFFER, memSize, host_pos_[0], GL_DYNAMIC_DRAW);

                int size = 0;
                glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

                if ((unsigned)size != memSize) {
                    fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!n");
                }

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                checkCudaErrors(cudaGraphicsGLRegisterBuffer(&graphics_resource_[i], pbo_[i], cudaGraphicsMapFlagsNone));
            }
        } else {
            checkCudaErrors(cudaMalloc((void**)&device_data_[0].pos[0], memSize));
            checkCudaErrors(cudaMalloc((void**)&device_data_[0].pos[1], memSize));
        }

        checkCudaErrors(cudaMalloc((void**)&device_data_[0].vel, memSize));

        // At this point we already know P2P is supported
        if (use_p2p_) {
            for (unsigned int i = 1; i < nb_devices; i++) {
                cudaError_t error;

                // Enable access for gpu_i to memory owned by gpu0
                checkCudaErrors(cudaSetDevice(i));
                if ((error = cudaDeviceEnablePeerAccess(0, 0)) != cudaErrorPeerAccessAlreadyEnabled) {
                    checkCudaErrors(error);
                } else {
                    // We might have already enabled P2P, so catch this and reset error
                    // code...
                    cudaGetLastError();
                }

                checkCudaErrors(cudaEventCreate(&device_data_[i].event));

                // Point all GPUs to the memory allocated on gpu0
                device_data_[i].pos[0] = device_data_[0].pos[0];
                device_data_[i].pos[1] = device_data_[0].pos[1];
                device_data_[i].vel    = device_data_[0].vel;
            }
        }
    }
}

template <std::floating_point T> BodySystemCUDA<T>::~BodySystemCUDA() noexcept {
    if (use_sys_mem_) {
        checkCudaErrors(cudaFreeHost(host_pos_[0]));
        checkCudaErrors(cudaFreeHost(host_pos_[1]));
        checkCudaErrors(cudaFreeHost(host_vel_));

        for (unsigned int i = 0; i < device_data_.size(); i++) {
            cudaEventDestroy(device_data_[i].event);
        }
    } else {
        delete[] host_pos_[0];
        delete[] host_pos_[1];
        delete[] host_vel_;

        checkCudaErrors(cudaFree((void**)device_data_[0].vel));

        if (use_pbo_) {
            checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[0]));
            checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[1]));
            glDeleteBuffers(2, (const GLuint*)pbo_);
        } else {
            checkCudaErrors(cudaFree((void**)device_data_[0].pos[0]));
            checkCudaErrors(cudaFree((void**)device_data_[0].pos[1]));

            checkCudaErrors(cudaEventDestroy(device_data_[0].event));

            if (use_p2p_) {
                for (unsigned int i = 1; i < device_data_.size(); i++) {
                    checkCudaErrors(cudaEventDestroy(device_data_[i].event));
                }
            }
        }
    }
}

template <std::floating_point T> auto BodySystemCUDA<T>::setSoftening(T softening) -> void {
    T softeningSq = softening * softening;

    for (unsigned int i = 0; i < device_data_.size(); i++) {
        if (device_data_.size() > 1) {
            checkCudaErrors(cudaSetDevice(i));
        }

        checkCudaErrors(setSofteningSquared(softeningSq));
    }
}

template <std::floating_point T> auto BodySystemCUDA<T>::update(T deltaTime) -> void {
    integrateNbodySystem<T>(device_data_, graphics_resource_, current_read_, (float)deltaTime, (float)damping_, nb_bodies_, static_cast<unsigned int>(device_data_.size()), block_size_, use_pbo_);

    std::swap(current_read_, current_write_);
}

template <std::floating_point T> auto BodySystemCUDA<T>::update_params(const NBodyParams& active_params) -> void {
    setSoftening(active_params.softening);
    damping_ = active_params.damping;
}

template <std::floating_point T> auto BodySystemCUDA<T>::get_position() const -> std::span<const T> {
    T* hdata = 0;
    T* ddata = 0;

    cudaGraphicsResource* pgres = NULL;

    int currentReadHost = use_sys_mem_ ? current_read_ : 0;

    hdata = host_pos_[currentReadHost];
    ddata = device_data_[0].pos[current_read_];

    if (use_pbo_) {
        pgres = graphics_resource_[current_read_];
    }

    if (!use_sys_mem_) {
        if (pgres) {
            checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
            checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
            size_t bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ddata, &bytes, pgres));
        }

        checkCudaErrors(cudaMemcpy(hdata, ddata, nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        if (pgres) {
            checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
        }
    }

    return {hdata, nb_bodies_ * 4};
}
template <std::floating_point T> auto BodySystemCUDA<T>::get_velocity() const -> std::span<const T> {
    T* hdata = 0;
    T* ddata = 0;

    cudaGraphicsResource* pgres = NULL;

    hdata = host_vel_;
    ddata = device_data_[0].vel;

    if (!use_sys_mem_) {
        if (pgres) {
            checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
            checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
            size_t bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ddata, &bytes, pgres));
        }

        checkCudaErrors(cudaMemcpy(hdata, ddata, nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        if (pgres) {
            checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
        }
    }

    return {hdata, nb_bodies_ * 4};
}

template <std::floating_point T> auto BodySystemCUDA<T>::set_position(std::span<const T> data) -> void {
    current_read_  = 0;
    current_write_ = 1;

    if (use_pbo_) {
        glBindBuffer(GL_ARRAY_BUFFER, pbo_[current_read_]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(T) * nb_bodies_, data.data());

        int size = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

        if ((unsigned)size != 4 * (sizeof(T) * nb_bodies_)) {
            fprintf(stderr, "WARNING: Pixel Buffer Object download failed!n");
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        if (use_sys_mem_) {
            memcpy(host_pos_[current_read_], data.data(), nb_bodies_ * 4 * sizeof(T));
        } else
            checkCudaErrors(cudaMemcpy(device_data_[0].pos[current_read_], data.data(), nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <std::floating_point T> auto BodySystemCUDA<T>::set_velocity(std::span<const T> data) -> void {
    current_read_  = 0;
    current_write_ = 1;

    if (use_sys_mem_) {
        memcpy(host_vel_, data.data(), nb_bodies_ * 4 * sizeof(T));
    } else {
        checkCudaErrors(cudaMemcpy(device_data_[0].vel, data.data(), nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template BodySystemCUDA<float>;
template BodySystemCUDA<double>;
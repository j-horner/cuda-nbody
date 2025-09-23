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
BodySystemCUDA<T>::BodySystemCUDA(const ComputeCUDA& compute, unsigned int numDevices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params)
    : m_numBodies(static_cast<unsigned int>(compute.nb_bodies())), m_numDevices(numDevices), m_bUsePBO(compute.use_pbo()), m_bUseSysMem(compute.use_host_mem()), m_bUseP2P(useP2P), m_blockSize(blockSize), m_devID(deviceId),
      m_damping(params.m_damping) {
    _initialize(m_numBodies);

    setSoftening(params.m_softening);

    reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDA<T>::BodySystemCUDA(const ComputeCUDA& compute, unsigned int numDevices, unsigned int blockSize, bool useP2P, int deviceId, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : m_numBodies(static_cast<unsigned int>(compute.nb_bodies())), m_numDevices(numDevices), m_bUsePBO(compute.use_pbo()), m_bUseSysMem(compute.use_host_mem()), m_bUseP2P(useP2P), m_blockSize(blockSize), m_devID(deviceId),
      m_hPos_vec(std::move(positions)), m_hVel_vec(std::move(velocities)), m_damping(params.m_damping) {
    assert(m_hPos_vec.size() == m_numBodies);
    assert(m_hVel_vec.size() == m_numBodies);

    _initialize(m_numBodies);

    setSoftening(params.m_softening);

    set_position(m_hPos_vec);
    set_velocity(m_hVel_vec);
}

template <std::floating_point T> BodySystemCUDA<T>::~BodySystemCUDA() noexcept {
    _finalize();
    m_numBodies = 0;
}

template <std::floating_point T> auto BodySystemCUDA<T>::reset(const NBodyParams& params, NBodyConfig config) -> void {
    randomise_bodies<T>(config, m_hPos_vec, m_hVel_vec, params.m_clusterScale, params.m_velocityScale);
    set_position(m_hPos_vec);
    set_velocity(m_hVel_vec);
}

template <std::floating_point T> auto BodySystemCUDA<T>::_initialize(int num_bodies) -> void {
    assert(!m_bInitialized);

    m_numBodies = num_bodies;

    unsigned int memSize = sizeof(T) * 4 * num_bodies;

    m_deviceData.resize(m_numDevices);

    // divide up the workload amongst Devices
    float* weights = new float[m_numDevices];
    int*   numSms  = new int[m_numDevices];
    float  total   = 0;

    for (unsigned int i = 0; i < m_numDevices; i++) {
        cudaDeviceProp props;
        checkCudaErrors(cudaGetDeviceProperties(&props, i));

        // Choose the weight based on the Compute Capability
        // We estimate that a CC2.0 SM is about 4.0x faster than a CC 1.x SM for
        // this application (since a 15-SM GF100 is about 2X faster than a 30-SM
        // GT200).
        numSms[i]  = props.multiProcessorCount;
        weights[i] = numSms[i] * (props.major >= 2 ? 4.f : 1.f);
        total += weights[i];
    }

    unsigned int offset    = 0;
    unsigned int remaining = m_numBodies;

    for (unsigned int i = 0; i < m_numDevices; i++) {
        unsigned int count = (int)((weights[i] / total) * m_numBodies);
        // Rounding up to numSms[i]*256 leads to better GPU utilization _per_ GPU
        // but when using multiple devices, it will lead to the last GPUs not having
        // any work at all
        // which means worse overall performance
        // unsigned int round = numSms[i] * 256;
        unsigned int round = 256;

        count = round * ((count + round - 1) / round);
        if (count > remaining) {
            count = remaining;
        }

        remaining -= count;
        m_deviceData[i].offset    = offset;
        m_deviceData[i].numBodies = count;
        offset += count;

        if ((i == m_numDevices - 1) && (offset < m_numBodies - 1)) {
            m_deviceData[i].numBodies += m_numBodies - offset;
        }
    }

    delete[] weights;
    delete[] numSms;

    if (m_bUseSysMem) {
        checkCudaErrors(cudaHostAlloc((void**)&m_hPos[0], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc((void**)&m_hPos[1], memSize, cudaHostAllocMapped | cudaHostAllocPortable));
        checkCudaErrors(cudaHostAlloc((void**)&m_hVel, memSize, cudaHostAllocMapped | cudaHostAllocPortable));

        memset(m_hPos[0], 0, memSize);
        memset(m_hPos[1], 0, memSize);
        memset(m_hVel, 0, memSize);

        for (unsigned int i = 0; i < m_numDevices; i++) {
            if (m_numDevices > 1) {
                checkCudaErrors(cudaSetDevice(i));
            }

            checkCudaErrors(cudaEventCreate(&m_deviceData[i].event));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&m_deviceData[i].dPos[0], (void*)m_hPos[0], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&m_deviceData[i].dPos[1], (void*)m_hPos[1], 0));
            checkCudaErrors(cudaHostGetDevicePointer((void**)&m_deviceData[i].dVel, (void*)m_hVel, 0));
        }
    } else {
        m_hPos[0] = new T[m_numBodies * 4];
        m_hVel    = new T[m_numBodies * 4];

        memset(m_hPos[0], 0, memSize);
        memset(m_hVel, 0, memSize);

        checkCudaErrors(cudaSetDevice(m_devID));
        checkCudaErrors(cudaEventCreate(&m_deviceData[0].event));

        if (m_bUsePBO) {
            // create the position pixel buffer objects for rendering
            // we will actually compute directly from this memory in CUDA too
            glGenBuffers(2, (GLuint*)m_pbo);

            for (int i = 0; i < 2; ++i) {
                glBindBuffer(GL_ARRAY_BUFFER, m_pbo[i]);
                glBufferData(GL_ARRAY_BUFFER, memSize, m_hPos[0], GL_DYNAMIC_DRAW);

                int size = 0;
                glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

                if ((unsigned)size != memSize) {
                    fprintf(stderr, "WARNING: Pixel Buffer Object allocation failed!n");
                }

                glBindBuffer(GL_ARRAY_BUFFER, 0);
                checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_pGRes[i], m_pbo[i], cudaGraphicsMapFlagsNone));
            }
        } else {
            checkCudaErrors(cudaMalloc((void**)&m_deviceData[0].dPos[0], memSize));
            checkCudaErrors(cudaMalloc((void**)&m_deviceData[0].dPos[1], memSize));
        }

        checkCudaErrors(cudaMalloc((void**)&m_deviceData[0].dVel, memSize));

        // At this point we already know P2P is supported
        if (m_bUseP2P) {
            for (unsigned int i = 1; i < m_numDevices; i++) {
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

                checkCudaErrors(cudaEventCreate(&m_deviceData[i].event));

                // Point all GPUs to the memory allocated on gpu0
                m_deviceData[i].dPos[0] = m_deviceData[0].dPos[0];
                m_deviceData[i].dPos[1] = m_deviceData[0].dPos[1];
                m_deviceData[i].dVel    = m_deviceData[0].dVel;
            }
        }
    }

    m_bInitialized = true;
}

template <std::floating_point T> auto BodySystemCUDA<T>::_finalize() noexcept -> void {
    assert(m_bInitialized);

    if (m_bUseSysMem) {
        checkCudaErrors(cudaFreeHost(m_hPos[0]));
        checkCudaErrors(cudaFreeHost(m_hPos[1]));
        checkCudaErrors(cudaFreeHost(m_hVel));

        for (unsigned int i = 0; i < m_numDevices; i++) {
            cudaEventDestroy(m_deviceData[i].event);
        }
    } else {
        delete[] m_hPos[0];
        delete[] m_hPos[1];
        delete[] m_hVel;

        checkCudaErrors(cudaFree((void**)m_deviceData[0].dVel));

        if (m_bUsePBO) {
            checkCudaErrors(cudaGraphicsUnregisterResource(m_pGRes[0]));
            checkCudaErrors(cudaGraphicsUnregisterResource(m_pGRes[1]));
            glDeleteBuffers(2, (const GLuint*)m_pbo);
        } else {
            checkCudaErrors(cudaFree((void**)m_deviceData[0].dPos[0]));
            checkCudaErrors(cudaFree((void**)m_deviceData[0].dPos[1]));

            checkCudaErrors(cudaEventDestroy(m_deviceData[0].event));

            if (m_bUseP2P) {
                for (unsigned int i = 1; i < m_numDevices; i++) {
                    checkCudaErrors(cudaEventDestroy(m_deviceData[i].event));
                }
            }
        }
    }

    m_bInitialized = false;
}

template <std::floating_point T> auto BodySystemCUDA<T>::setSoftening(T softening) -> void {
    T softeningSq = softening * softening;

    for (unsigned int i = 0; i < m_numDevices; i++) {
        if (m_numDevices > 1) {
            checkCudaErrors(cudaSetDevice(i));
        }

        checkCudaErrors(setSofteningSquared(softeningSq));
    }
}

template <std::floating_point T> auto BodySystemCUDA<T>::update(T deltaTime) -> void {
    assert(m_bInitialized);

    integrateNbodySystem<T>(m_deviceData, m_pGRes, m_currentRead, (float)deltaTime, (float)m_damping, m_numBodies, m_numDevices, m_blockSize, m_bUsePBO);

    std::swap(m_currentRead, m_currentWrite);
}

template <std::floating_point T> auto BodySystemCUDA<T>::update_params(const NBodyParams& active_params) -> void {
    setSoftening(active_params.m_softening);
    m_damping = active_params.m_damping;
}

template <std::floating_point T> auto BodySystemCUDA<T>::get_position() const -> std::span<const T> {
    assert(m_bInitialized);

    T* hdata = 0;
    T* ddata = 0;

    cudaGraphicsResource* pgres = NULL;

    int currentReadHost = m_bUseSysMem ? m_currentRead : 0;

    hdata = m_hPos[currentReadHost];
    ddata = m_deviceData[0].dPos[m_currentRead];

    if (m_bUsePBO) {
        pgres = m_pGRes[m_currentRead];
    }

    if (!m_bUseSysMem) {
        if (pgres) {
            checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
            checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
            size_t bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ddata, &bytes, pgres));
        }

        checkCudaErrors(cudaMemcpy(hdata, ddata, m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        if (pgres) {
            checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
        }
    }

    return {hdata, m_numBodies * 4};
}
template <std::floating_point T> auto BodySystemCUDA<T>::get_velocity() const -> std::span<const T> {
    assert(m_bInitialized);

    T* hdata = 0;
    T* ddata = 0;

    cudaGraphicsResource* pgres = NULL;

    hdata = m_hVel;
    ddata = m_deviceData[0].dVel;

    if (!m_bUseSysMem) {
        if (pgres) {
            checkCudaErrors(cudaGraphicsResourceSetMapFlags(pgres, cudaGraphicsMapFlagsReadOnly));
            checkCudaErrors(cudaGraphicsMapResources(1, &pgres, 0));
            size_t bytes;
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&ddata, &bytes, pgres));
        }

        checkCudaErrors(cudaMemcpy(hdata, ddata, m_numBodies * 4 * sizeof(T), cudaMemcpyDeviceToHost));

        if (pgres) {
            checkCudaErrors(cudaGraphicsUnmapResources(1, &pgres, 0));
        }
    }

    return {hdata, m_numBodies * 4};
}

template <std::floating_point T> auto BodySystemCUDA<T>::set_position(std::span<const T> data) -> void {
    assert(m_bInitialized);

    m_currentRead  = 0;
    m_currentWrite = 1;

    if (m_bUsePBO) {
        glBindBuffer(GL_ARRAY_BUFFER, m_pbo[m_currentRead]);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * sizeof(T) * m_numBodies, data.data());

        int size = 0;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, (GLint*)&size);

        if ((unsigned)size != 4 * (sizeof(T) * m_numBodies)) {
            fprintf(stderr, "WARNING: Pixel Buffer Object download failed!n");
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    } else {
        if (m_bUseSysMem) {
            memcpy(m_hPos[m_currentRead], data.data(), m_numBodies * 4 * sizeof(T));
        } else
            checkCudaErrors(cudaMemcpy(m_deviceData[0].dPos[m_currentRead], data.data(), m_numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <std::floating_point T> auto BodySystemCUDA<T>::set_velocity(std::span<const T> data) -> void {
    assert(m_bInitialized);

    m_currentRead  = 0;
    m_currentWrite = 1;

    if (m_bUseSysMem) {
        memcpy(m_hVel, data.data(), m_numBodies * 4 * sizeof(T));
    } else {
        checkCudaErrors(cudaMemcpy(m_deviceData[0].dVel, data.data(), m_numBodies * 4 * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template BodySystemCUDA<float>;
template BodySystemCUDA<double>;
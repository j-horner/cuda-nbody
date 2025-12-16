#include "bodysystemcuda_graphics.hpp"

#include "gl_includes.hpp"
#include "helper_cuda.hpp"
#include "integrate_nbody_cuda.hpp"

#include <cuda_gl_interop.h>

#include <cassert>

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
    checkCudaErrors(cudaGraphicsResourceSetMapFlags(graphics_resource_[this->current_read_], cudaGraphicsMapFlagsReadOnly));
    checkCudaErrors(cudaGraphicsResourceSetMapFlags(graphics_resource_[1 - this->current_read_], cudaGraphicsMapFlagsWriteDiscard));
    checkCudaErrors(cudaGraphicsMapResources(2, graphics_resource_, 0));
    size_t bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&(device_pos_[this->current_read_]), &bytes, graphics_resource_[this->current_read_]));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&(device_pos_[1 - this->current_read_]), &bytes, graphics_resource_[1 - this->current_read_]));

    integrateNbodySystem<T>(device_pos_[1 - this->current_read_], device_pos_[this->current_read_], device_vel_, this->current_read_, deltaTime, this->damping_, this->nb_bodies_, this->block_size_);

    checkCudaErrors(cudaGraphicsUnmapResources(2, graphics_resource_, 0));

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

template BodySystemCUDAGraphics<float>;
template BodySystemCUDAGraphics<double>;

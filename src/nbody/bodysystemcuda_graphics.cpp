#include "bodysystemcuda_graphics.hpp"

#include "gl_includes.hpp"
#include "helper_cuda.hpp"
#include "integrate_nbody_cuda.hpp"

#include <cuda_gl_interop.h>

#include <cassert>

template <std::floating_point T> BodySystemCUDAGraphics<T>::BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    initialize();

    BodySystemCUDAGraphics<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDAGraphics<T>::BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : BodySystemCUDA<T>(nb_bodies, blockSize, params, std::move(positions), std::move(velocities)) {
    assert(this->host_pos_vec_.size() == 4 * this->nb_bodies_);
    assert(this->host_vel_vec_.size() == 4 * this->nb_bodies_);

    initialize();

    set_position(this->host_pos_vec_);
    set_velocity(this->host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::initialize() -> void {
    const auto memSize = sizeof(T) * 4 * this->nb_bodies_;

    host_pos_.resize(this->nb_bodies_ * 4, 0);
    host_vel_.resize(this->nb_bodies_ * 4, 0);

    // create the position pixel buffer objects for rendering
    // we will actually compute directly from this memory in CUDA too
    {
        const auto host_positions = std::span<const T>{host_pos_};

        pbos_ = BufferObjects<2>::create_dynamic(std::array{host_positions, host_positions});
    }

    for (int i = 0; i < 2; ++i) {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&graphics_resource_[i], pbos_.buffer(i), cudaGraphicsMapFlagsNone));
    }

    checkCudaErrors(cudaMalloc((void**)&device_vel_, memSize));
}

template <std::floating_point T> BodySystemCUDAGraphics<T>::~BodySystemCUDAGraphics() noexcept {
    checkCudaErrors(cudaFree((void**)device_vel_));

    checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[0]));
    checkCudaErrors(cudaGraphicsUnregisterResource(graphics_resource_[1]));
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

    pbos_.bind_data(this->current_read_, data);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::set_velocity(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    checkCudaErrors(cudaMemcpy(device_vel_, data.data(), this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyHostToDevice));
}

template BodySystemCUDAGraphics<float>;
template BodySystemCUDAGraphics<double>;

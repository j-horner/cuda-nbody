#include "bodysystemcuda_graphics.hpp"
#include "gl_includes.hpp"
#include "integrate_nbody_cuda.hpp"

#include <cuda_gl_interop.h>
#include <thrust/copy.h>

#include <stdexcept>

#include <cassert>

template <std::floating_point T> BodySystemCUDAGraphics<T>::BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    BodySystemCUDAGraphics<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::update(T deltaTime) -> void {
    {
        const auto device_mapping = graphics_resources_.map<T>(std::array{CUDAGraphicsFlag{this->current_read_, cudaGraphicsMapFlagsReadOnly}, CUDAGraphicsFlag{1 - this->current_read_, cudaGraphicsMapFlagsWriteDiscard}});

        const auto& position_ptrs = device_mapping.pointers();

        integrateNbodySystem<T>(position_ptrs[1 - this->current_read_], position_ptrs[this->current_read_], device_vel_.data().get(), this->current_read_, deltaTime, this->damping_, this->nb_bodies_, this->block_size_);
    }

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::get_position() const -> std::span<const T> {
    {
        const auto device_mapping = graphics_resources_.map<T>(CUDAGraphicsFlag{this->current_read_, cudaGraphicsMapFlagsReadOnly});

        const auto& device_data = device_mapping.pointers()[0];

        const auto result = cudaMemcpy(host_pos_.data(), device_data, this->nb_bodies_ * 4 * sizeof(T), cudaMemcpyDeviceToHost);

        if (result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorName(result));
        }
    }

    return host_pos_;
}
template <std::floating_point T> auto BodySystemCUDAGraphics<T>::get_velocity() const -> std::span<const T> {
    thrust::copy(device_vel_.begin(), device_vel_.end(), host_vel_.begin());

    return host_vel_;
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::set_position(std::span<const T> data) -> void {
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    pbos_.bind_data(this->current_read_, data);
}

template <std::floating_point T> auto BodySystemCUDAGraphics<T>::set_velocity(std::span<const T> data) -> void {
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    thrust::copy(data.begin(), data.end(), device_vel_.begin());
}

template BodySystemCUDAGraphics<float>;
template BodySystemCUDAGraphics<double>;

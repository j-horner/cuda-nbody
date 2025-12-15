#include "bodysystemcuda_host_memory.hpp"
#include "helper_cuda.hpp"
#include "integrate_nbody_cuda.hpp"

#include <cassert>

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
    integrateNbodySystem<T>(device_pos_[1 - this->current_read_], device_pos_[this->current_read_], device_vel_, this->current_read_, deltaTime, this->damping_, this->nb_bodies_, this->block_size_);

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

template BodySystemCUDAHostMemory<float>;
template BodySystemCUDAHostMemory<double>;
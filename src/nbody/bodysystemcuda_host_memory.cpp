#include "bodysystemcuda_host_memory.hpp"

#include "helper_cuda.hpp"
#include "integrate_nbody_cuda.hpp"

#include <algorithm>

#include <cassert>

template <std::floating_point T> BodySystemCUDAHostMemory<T>::BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, blockSize, params) {
    BodySystemCUDAHostMemory<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T>
BodySystemCUDAHostMemory<T>::BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities)
    : BodySystemCUDA<T>(nb_bodies, blockSize, params, std::move(positions), std::move(velocities)) {
    set_position(this->host_pos_vec_);
    set_velocity(this->host_vel_vec_);
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::update(T deltaTime) -> void {
    integrateNbodySystem<
        T>(positions_[1 - this->current_read_].device_ptr(), positions_[this->current_read_].device_ptr(), velocities_.device_ptr(), this->current_read_, deltaTime, this->damping_, this->nb_bodies_, this->block_size_);

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::get_position() const -> std::span<const T> {
    return {positions_[this->current_read_].host_ptr(), this->nb_bodies_ * 4};
}
template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::get_velocity() const -> std::span<const T> {
    return {velocities_.host_ptr(), this->nb_bodies_ * 4};
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::set_position(std::span<const T> data) -> void {
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    std::ranges::copy(data, positions_[this->current_read_].host_ptr());
}

template <std::floating_point T> auto BodySystemCUDAHostMemory<T>::set_velocity(std::span<const T> data) -> void {
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    std::ranges::copy(data, velocities_.host_ptr());
}

template BodySystemCUDAHostMemory<float>;
template BodySystemCUDAHostMemory<double>;
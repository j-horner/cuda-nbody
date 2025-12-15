#include "bodysystemcuda_default.hpp"
#include "helper_cuda.hpp"
#include "integrate_nbody_cuda.hpp"

#include <thrust/copy.h>

#include <cassert>

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
    const auto nb_values = this->nb_bodies_ * 4;

    host_pos_.resize(nb_values, 0);
    host_vel_.resize(nb_values, 0);

    device_pos_[0].resize(nb_values, 0);
    device_pos_[1].resize(nb_values, 0);
    device_vel_.resize(nb_values, 0);
}

template <std::floating_point T> BodySystemCUDADefault<T>::~BodySystemCUDADefault() noexcept = default;

template <std::floating_point T> auto BodySystemCUDADefault<T>::update(T deltaTime) -> void {
    integrateNbodySystem<
        T>(device_pos_[1 - this->current_read_].data().get(), device_pos_[this->current_read_].data().get(), device_vel_.data().get(), this->current_read_, deltaTime, this->damping_, this->nb_bodies_, this->block_size_);

    std::swap(this->current_read_, this->current_write_);
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::get_position() const -> std::span<const T> {
    const auto& positions = device_pos_[this->current_read_];

    thrust::copy(positions.begin(), positions.end(), host_pos_.begin());

    return host_pos_;
}
template <std::floating_point T> auto BodySystemCUDADefault<T>::get_velocity() const -> std::span<const T> {
    thrust::copy(device_vel_.begin(), device_vel_.end(), host_vel_.begin());

    return host_vel_;
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::set_position(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    thrust::copy(data.begin(), data.end(), device_pos_[this->current_read_].begin());
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::set_velocity(std::span<const T> data) -> void {
    this->current_read_  = 0;
    this->current_write_ = 1;

    thrust::copy(data.begin(), data.end(), device_vel_.begin());
}

template BodySystemCUDADefault<float>;
template BodySystemCUDADefault<double>;
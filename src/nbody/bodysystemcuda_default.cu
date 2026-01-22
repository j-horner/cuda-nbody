#include "bodysystemcuda_default.hpp"
#include "integrate_nbody_cuda.hpp"

#include <thrust/copy.h>

#include <cassert>

template <std::floating_point T> BodySystemCUDADefault<T>::BodySystemCUDADefault(unsigned int nb_bodies, const NBodyParams& params) : BodySystemCUDA<T>(nb_bodies, params) {
    BodySystemCUDADefault<T>::reset(params, NBodyConfig::NBODY_CONFIG_SHELL);
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::update(T deltaTime) -> void {
    integrateNbodySystem<T>(device_pos_[1 - this->current_read_].data().get(), device_pos_[this->current_read_].data().get(), device_vel_.data().get(), this->current_read_, deltaTime, this->damping_, this->nb_bodies_);

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
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    thrust::copy(data.begin(), data.end(), device_pos_[this->current_read_].begin());
}

template <std::floating_point T> auto BodySystemCUDADefault<T>::set_velocity(std::span<const T> data) -> void {
    assert(data.size() == 4 * this->nb_bodies_);

    this->current_read_  = 0;
    this->current_write_ = 1;

    thrust::copy(data.begin(), data.end(), device_vel_.begin());
}

template BodySystemCUDADefault<float>;
template BodySystemCUDADefault<double>;
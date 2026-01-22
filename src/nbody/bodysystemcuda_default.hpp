#pragma once

#include "bodysystemcuda.hpp"

#include <thrust/device_vector.h>

#include <array>
#include <concepts>

///
/// @brief  The default CUDA implementation. No graphics interop, no host memory.
///
template <std::floating_point T> class BodySystemCUDADefault : public BodySystemCUDA<T> {
 public:
    BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    ~BodySystemCUDADefault() noexcept final = default;

 private:
    // Host data
    mutable std::vector<T> host_pos_ = std::vector<T>(this->nb_bodies_ * 4, 0);
    mutable std::vector<T> host_vel_ = std::vector<T>(this->nb_bodies_ * 4, 0);

    // Device data
    std::array<thrust::device_vector<T>, 2> device_pos_ = std::array{thrust::device_vector<T>(this->nb_bodies_ * 4, 0), thrust::device_vector<T>(this->nb_bodies_ * 4, 0)};
    thrust::device_vector<T>                device_vel_ = thrust::device_vector<T>(this->nb_bodies_ * 4, 0);
};

extern template BodySystemCUDADefault<float>;
extern template BodySystemCUDADefault<double>;
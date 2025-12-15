#pragma once

#include "bodysystemcuda.hpp"

#include <thrust/device_vector.h>

///
/// @brief  The default CUDA implementation. No graphics interop, no host memory.
///
template <std::floating_point T> class BodySystemCUDADefault : public BodySystemCUDA<T> {
 public:
    BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDADefault(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    ~BodySystemCUDADefault() noexcept;

 private:
    auto _initialize() -> void;

    // Host data
    mutable std::vector<T> host_pos_;
    mutable std::vector<T> host_vel_;

    // Device data
    std::array<thrust::device_vector<T>, 2> device_pos_;
    thrust::device_vector<T>                device_vel_;
};

extern template BodySystemCUDADefault<float>;
extern template BodySystemCUDADefault<double>;
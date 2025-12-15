#pragma once

#include "bodysystemcuda.hpp"

///
/// @brief The CUDA implementation with host memory mapped to device memory.
///
template <std::floating_point T> class BodySystemCUDAHostMemory : public BodySystemCUDA<T> {
 public:
    BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDAHostMemory(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    ~BodySystemCUDAHostMemory() noexcept;

 private:
    auto _initialize() -> void;

    // Host data
    std::array<T*, 2> host_pos_{nullptr, nullptr};
    T*                host_vel_ = nullptr;

    // Device data
    std::array<T*, 2> device_pos_{nullptr, nullptr};
    T*                device_vel_ = nullptr;
};

extern template BodySystemCUDAHostMemory<float>;
extern template BodySystemCUDAHostMemory<double>;
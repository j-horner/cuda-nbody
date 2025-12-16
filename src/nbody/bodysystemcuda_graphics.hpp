#pragma once

#include "bodysystemcuda.hpp"

#include <array>

struct cudaGraphicsResource;

///
/// @brief  The CUDA implementation using OpenGL interop. Some GPU buffers are allocated by OpenGL.
///
template <std::floating_point T> class BodySystemCUDAGraphics : public BodySystemCUDA<T> {
 public:
    BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params);
    BodySystemCUDAGraphics(unsigned int nb_bodies, unsigned int blockSize, const NBodyParams& params, std::vector<T> positions, std::vector<T> velocities);

    auto update(T deltaTime) -> void final;

    auto get_position() const -> std::span<const T> final;
    auto get_velocity() const -> std::span<const T> final;

    auto set_position(std::span<const T> data) -> void final;
    auto set_velocity(std::span<const T> data) -> void final;

    auto getCurrentReadBuffer() const noexcept -> unsigned int { return pbo_[BodySystemCUDA<T>::current_read_]; }

    ~BodySystemCUDAGraphics() noexcept;

 private:
    auto initialize() -> void;

    // Host data
    mutable std::vector<T> host_pos_;
    mutable std::vector<T> host_vel_;

    // Device data
    std::array<T*, 2> device_pos_{nullptr, nullptr};
    T*                device_vel_ = nullptr;

    unsigned int          pbo_[2];
    cudaGraphicsResource* graphics_resource_[2];
};

extern template BodySystemCUDAGraphics<float>;
extern template BodySystemCUDAGraphics<double>;

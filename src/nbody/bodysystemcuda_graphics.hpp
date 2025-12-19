#pragma once

#include "bodysystemcuda.hpp"
#include "buffer_objects.hpp"
#include "cuda_opengl_buffers.hpp"

#include <thrust/device_vector.h>

#include <array>

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

    auto getCurrentReadBuffer() const noexcept { return pbos_.use(BodySystemCUDA<T>::current_read_); }

    ~BodySystemCUDAGraphics() noexcept final = default;

 private:
    // Host data
    mutable std::vector<T> host_pos_ = std::vector<T>(4 * this->nb_bodies_, T{0});
    mutable std::vector<T> host_vel_ = std::vector<T>(4 * this->nb_bodies_, T{0});

    // Device data
    BufferObjects<2>             pbos_               = BufferObjects<2>::create_dynamic(std::array<std::span<const T>, 2>{host_pos_, host_pos_});
    mutable CUDAOpenGLBuffers<2> graphics_resources_ = CUDAOpenGLBuffers<2>{pbos_};

    thrust::device_vector<T> device_vel_ = thrust::device_vector<T>(4 * this->nb_bodies_, T{0});
};

extern template BodySystemCUDAGraphics<float>;
extern template BodySystemCUDAGraphics<double>;

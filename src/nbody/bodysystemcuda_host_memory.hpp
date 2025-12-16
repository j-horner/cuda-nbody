#pragma once

#include "bodysystemcuda.hpp"
#include "unique_mapped_span.hpp"

#include <array>
#include <concepts>

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
    auto initialize() -> void;

    std::array<UniqueMappedSpan<T>, 2> positions_;
    UniqueMappedSpan<T>                velocities_;
};

extern template BodySystemCUDAHostMemory<float>;
extern template BodySystemCUDAHostMemory<double>;
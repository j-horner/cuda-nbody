#pragma once

#include "nbody_config.hpp"

#include <cuda/api/event.hpp>

#include <chrono>
#include <concepts>
#include <memory>
#include <span>
#include <vector>

struct NBodyParams;
class Interface;
template <std::floating_point T> class BodySystemCUDA;
template <std::floating_point T> class BodySystemCUDAGraphics;

namespace cuda {
class device_t;
}    // namespace cuda

class ComputeCUDA {
 public:
    ComputeCUDA(bool enable_host_mem, bool use_pbo, double fp64_enabled, std::size_t num_bodies, const NBodyParams& params);

    auto nb_bodies() const noexcept { return nb_bodies_; }
    auto use_pbo() const noexcept { return use_pbo_; }
    auto use_host_mem() const noexcept { return use_host_mem_; }

    auto get_position_fp32() const noexcept -> std::span<const float>;
    auto get_position_fp64() const noexcept -> std::span<const double>;

    auto display(Interface& interface) const -> void;

    auto switch_precision() -> void;

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto set_values(std::span<const float> positions, std::span<const float> velocities) -> void;
    auto set_values(std::span<const double> positions, std::span<const double> velocities) -> void;

    auto update(float dt) -> void;

    auto update_params(const NBodyParams& params) -> void;

    auto compare_results(const NBodyParams& params) -> bool;

    using Milliseconds = std::chrono::duration<float, std::milli>;

    auto get_milliseconds_passed() -> Milliseconds;

    auto run_benchmark(int nb_iterations, float dt) -> Milliseconds;

    ~ComputeCUDA() noexcept;

 private:
    template <std::floating_point TNew, std::floating_point TOld> auto switch_precision(BodySystemCUDA<TNew>& new_nbody, const BodySystemCUDA<TOld>& old_nbody) -> void;

    template <std::floating_point T> auto run_benchmark(int nb_iterations, float dt, BodySystemCUDA<T>& nbody) -> Milliseconds;

    template <std::floating_point T> auto compare_results(const NBodyParams& params, BodySystemCUDA<T>& nbodyCuda) const -> bool;

    std::size_t nb_bodies_;

    bool fp64_enabled_;

    bool use_host_mem_;
    bool use_pbo_;

    bool double_supported_ = true;

    std::unique_ptr<BodySystemCUDA<float>>  nbody_fp32_;
    std::unique_ptr<BodySystemCUDA<double>> nbody_fp64_;

    BodySystemCUDAGraphics<float>*  nbody_fp32_pbo_ = nullptr;
    BodySystemCUDAGraphics<double>* nbody_fp64_pbo_ = nullptr;

    cuda::event_t host_mem_sync_event_;
    cuda::event_t start_event_;
    cuda::event_t stop_event_;
};
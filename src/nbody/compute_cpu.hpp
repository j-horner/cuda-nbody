#pragma once

#include "nbody_config.hpp"

#include <chrono>
#include <concepts>
#include <memory>
#include <span>
#include <vector>

struct NBodyParams;
class Interface;

template <std::floating_point T> class BodySystemCPU;

class ComputeCPU {
 public:
    ComputeCPU(double fp64_enabled, std::size_t num_bodies, const NBodyParams& params);
    ComputeCPU(double              fp64_enabled,
               std::size_t         num_bodies,
               const NBodyParams&  params,
               std::vector<float>  positions_fp32,
               std::vector<float>  velocities_fp32,
               std::vector<double> positions_fp64,
               std::vector<double> velocities_fp64);

    auto nb_bodies() const noexcept { return nb_bodies_; }

    auto switch_precision() -> void;

    auto run_benchmark(int nb_iterations, float dt) -> float;

    auto reset(const NBodyParams& params, NBodyConfig config) -> void;

    auto set_values(std::span<const float> positions, std::span<const float> velocities) -> void;
    auto set_values(std::span<const double> positions, std::span<const double> velocities) -> void;

    auto update(float dt) -> void;

    auto get_position_fp32() const noexcept -> std::span<const float>;
    auto get_position_fp64() const noexcept -> std::span<const double>;

    auto update_params(const NBodyParams& params) -> void;

    auto get_milliseconds_passed() -> float;

    auto display(Interface& interface) const -> void;

    ~ComputeCPU() noexcept;

 private:
    template <std::floating_point TNew, std::floating_point TOld> auto switch_precision(BodySystemCPU<TNew>& new_nbody, const BodySystemCPU<TOld>& old_nbody) -> void;

    template <std::floating_point T> auto run_benchmark(int nb_iterations, float dt, BodySystemCPU<T>& nbody) -> float;

    std::size_t nb_bodies_;

    bool fp64_enabled_;

    std::unique_ptr<BodySystemCPU<float>>  nbody_fp32_;
    std::unique_ptr<BodySystemCPU<double>> nbody_fp64_;

    using Clock        = std::chrono::steady_clock;
    using TimePoint    = std::chrono::time_point<Clock>;
    using MilliSeconds = std::chrono::duration<float, std::milli>;

    TimePoint reset_time_;
};
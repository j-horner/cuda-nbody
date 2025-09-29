#pragma once

#include "nbody_config.hpp"

#include <cuda/api/event.hpp>

#include <concepts>
#include <memory>
#include <span>
#include <vector>

struct NBodyParams;
class Interface;
template <std::floating_point T> class BodySystemCUDA;

namespace cuda {
class device_t;
}    // namespace cuda

class ComputeCUDA {
 public:
    ComputeCUDA(int nb_requested_devices, bool enable_host_mem, bool use_pbo, int device, int block_size, double fp64_enabled, std::size_t num_bodies, const NBodyParams& params);

    ComputeCUDA(int                 nb_requested_devices,
                bool                enable_host_mem,
                bool                use_pbo,
                int                 device,
                int                 block_size,
                double              fp64_enabled,
                std::size_t         num_bodies,
                const NBodyParams&  params,
                std::vector<float>  positions_fp32,
                std::vector<float>  velocities_fp32,
                std::vector<double> positions_fp64,
                std::vector<double> velocities_fp64);

    auto nb_bodies() const noexcept { return nb_bodies_; }
    auto use_pbo() const noexcept { return use_pbo_; }
    auto use_host_mem() const noexcept { return use_host_mem_; }

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

    auto compare_results(const NBodyParams& params) -> bool;

    ~ComputeCUDA() noexcept;

 private:
    ComputeCUDA(int                  nb_requested_devices,
                bool                 enable_host_mem,
                bool                 use_pbo,
                const cuda::device_t main_device,
                int                  block_size,
                double               fp64_enabled,
                std::size_t          num_bodies,
                const NBodyParams&   params,
                std::vector<float>   positions_fp32,
                std::vector<float>   velocities_fp32,
                std::vector<double>  positions_fp64,
                std::vector<double>  velocities_fp64);

    template <std::floating_point TNew, std::floating_point TOld> auto switch_precision(BodySystemCUDA<TNew>& new_nbody, const BodySystemCUDA<TOld>& old_nbody) -> void;

    template <std::floating_point T> auto run_benchmark(int nb_iterations, float dt, BodySystemCUDA<T>& nbody) -> float;

    template <std::floating_point T> auto compare_results(const NBodyParams& params, BodySystemCUDA<T>& nbodyCuda) const -> bool;

    std::size_t nb_bodies_;

    bool fp64_enabled_;

    bool use_host_mem_;
    bool use_pbo_;

    bool double_supported_ = true;

    std::unique_ptr<BodySystemCUDA<float>>  nbody_fp32_;
    std::unique_ptr<BodySystemCUDA<double>> nbody_fp64_;

    cuda::event_t host_mem_sync_event_;
    cuda::event_t start_event_;
    cuda::event_t stop_event_;
};
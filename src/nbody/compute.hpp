#pragma once

#include "nbody_config.hpp"
#include "params.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>

class ComputeCPU;
class ComputeCUDA;
class Camera;
class Interface;
class ParamListGL;

class ComputeConfig {
 public:
    ComputeConfig(bool                         enable_fp64,
                  bool                         enable_cycle_demo,
                  bool                         enable_cpu,
                  bool                         enable_compare_to_cpu,
                  bool                         enable_benchmark,
                  bool                         enable_host_memory,
                  int                          device,
                  std::size_t                  nb_requested_devices,
                  std::size_t                  block_size,
                  std::size_t                  nb_bodies,
                  const std::filesystem::path& tipsy_file);

    auto nb_bodies() const noexcept { return num_bodies_; }

    auto& active_params() const noexcept { return active_params_; }

    auto interactions_per_second() const noexcept { return interactions_per_second_; }

    auto gflops() const noexcept { return g_flops_; }

    auto fps() const noexcept { return fps_; }

    auto fp64_enabled() const noexcept { return fp64_enabled_; }

    auto paused() const noexcept { return paused_; }

    auto add_modifiable_parameters(ParamListGL& param_list) -> void;

    auto run_benchmark(int nb_iterations) -> void;

    auto compare_results() -> bool;

    auto pause() noexcept -> void { paused_ = !paused_; }

    auto switch_precision() -> void;

    auto toggle_cycle_demo() -> void;

    auto previous_demo(Camera& camera) -> void;

    auto next_demo(Camera& camera) -> void;

    auto update_simulation(Camera& camera) -> void;

    auto display_NBody_system(Interface& interface) -> void;

    auto reset(NBodyConfig initial_configuration) -> void;

    auto update_params() -> void;

    auto calculate_fps(int frame_count) -> void;

    ~ComputeConfig() noexcept;

 private:
    auto print_benchmark_results(int nb_iterations, float milliseconds) -> void;

    auto select_demo(Camera& camera) -> void;

    constexpr auto compute_perf_stats(float frequency) -> void;

    auto get_milliseconds_passed() -> float;

    constexpr static auto demoParams = std::array{
        NBodyParams{0.016f, 1.54f, 8.0f, 0.1f, 1.0f, 0, -2, -100},
        NBodyParams{0.016f, 0.68f, 20.0f, 0.1f, 1.0f, 0, -2, -30},
        NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0, 0, -15},
        NBodyParams{0.0006f, 0.16f, 1000.0f, 1.0f, 1.0f, 0, 0, -15},
        NBodyParams{0.0019f, 0.32f, 276.0f, 1.0f, 1.0f, 0, 0, -50},
        NBodyParams{0.0016f, 0.32f, 272.0f, 0.145f, 1.0f, 0, 0, -50},
        NBodyParams{0.016f, 6.04f, 0.0f, 1.0f, 1.0f, 0, 0, -50}};

    constexpr static auto nb_demos = demoParams.size();

    constexpr static auto demo_time = std::chrono::seconds(10);

    bool        paused_ = false;
    bool        fp64_enabled_;
    bool        cycle_demo_;
    int         active_demo_ = 0;
    bool        use_cpu_;
    std::size_t num_bodies_              = 16384;
    bool        double_supported_        = true;
    float       g_flops_                 = 0.f;
    float       fps_                     = 0.f;
    float       interactions_per_second_ = 0.f;
    NBodyParams active_params_           = demoParams[0];

    std::unique_ptr<ComputeCPU>  compute_cpu_;
    std::unique_ptr<ComputeCUDA> compute_cuda_;

    template <std::floating_point T> struct TipsyData {
        std::vector<T> positions;
        std::vector<T> velocities;
    };

    TipsyData<float>  tipsy_data_fp32_;
    TipsyData<double> tipsy_data_fp64_;

    using Clock     = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    TimePoint demo_reset_time_;
};

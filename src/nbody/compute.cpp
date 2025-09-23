#include "compute.hpp"

#include "camera.hpp"
#include "compute_cpu.hpp"
#include "compute_cuda.hpp"
#include "interface.hpp"
#include "param.hpp"
#include "paramgl.hpp"
#include "tipsy.hpp"

#include <chrono>
#include <memory>
#include <print>

namespace {
constexpr auto flops_per_interaction(bool fp64_enabled) {
    return fp64_enabled ? 30 : 20;
}
}    // namespace

ComputeConfig::~ComputeConfig() noexcept = default;

ComputeConfig::ComputeConfig(
    bool                         enable_fp64,
    bool                         enable_cycle_demo,
    bool                         enable_cpu,
    bool                         enable_compare_to_cpu,
    bool                         enable_benchmark,
    bool                         enable_host_memory,
    int                          device,
    std::size_t                  nb_requested_devices,
    std::size_t                  block_size,
    std::size_t                  nb_bodies,
    const std::filesystem::path& tipsy_file)
    : fp64_enabled_(enable_fp64), cycle_demo_(enable_cycle_demo), use_cpu_(enable_cpu) {
    const auto use_pbo = !(enable_benchmark || enable_compare_to_cpu || enable_host_memory || enable_cpu);
    if (!tipsy_file.empty()) {
        auto [positions, velocities] = read_tipsy_file(tipsy_file);

        tipsy_data_fp32_.positions.resize(positions.size());
        tipsy_data_fp32_.velocities.resize(velocities.size());

        using std::ranges::transform;

        constexpr auto to_float = [](double x) noexcept { return static_cast<float>(x); };

        transform(positions, tipsy_data_fp32_.positions.begin(), to_float);
        transform(velocities, tipsy_data_fp32_.velocities.begin(), to_float);

        tipsy_data_fp64_.positions  = std::move(positions);
        tipsy_data_fp64_.velocities = std::move(velocities);

        if (use_cpu_) {
            compute_cpu_ = std::make_unique<ComputeCPU>(enable_fp64, nb_bodies, active_params_, tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities, tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
        } else {
            compute_cuda_ = std::make_unique<ComputeCUDA>(
                nb_requested_devices,
                enable_host_memory,
                use_pbo,
                device,
                block_size,
                enable_fp64,
                nb_bodies,
                active_params_,
                tipsy_data_fp32_.positions,
                tipsy_data_fp32_.velocities,
                tipsy_data_fp64_.positions,
                tipsy_data_fp64_.velocities);
        }
    } else {
        if (use_cpu_) {
            compute_cpu_ = std::make_unique<ComputeCPU>(enable_fp64, nb_bodies, active_params_);
        } else {
            compute_cuda_ = std::make_unique<ComputeCUDA>(nb_requested_devices, enable_host_memory, use_pbo, device, block_size, enable_fp64, nb_bodies, active_params_);
        }
    }

    if (use_cpu_) {
        num_bodies_ = compute_cpu_->nb_bodies();
    } else {
        num_bodies_ = compute_cuda_->nb_bodies();
    }

    if (num_bodies_ <= 1024) {
        active_params_.m_clusterScale  = 1.52f;
        active_params_.m_velocityScale = 2.f;
    } else if (num_bodies_ <= 2048) {
        active_params_.m_clusterScale  = 1.56f;
        active_params_.m_velocityScale = 2.64f;
    } else if (num_bodies_ <= 4096) {
        active_params_.m_clusterScale  = 1.68f;
        active_params_.m_velocityScale = 2.98f;
    } else if (num_bodies_ <= 8192) {
        active_params_.m_clusterScale  = 1.98f;
        active_params_.m_velocityScale = 2.9f;
    } else if (num_bodies_ <= 16384) {
        active_params_.m_clusterScale  = 1.54f;
        active_params_.m_velocityScale = 8.f;
    } else if (num_bodies_ <= 32768) {
        active_params_.m_clusterScale  = 1.44f;
        active_params_.m_velocityScale = 11.f;
    }

    if (tipsy_file.empty()) {
        if (use_cpu_) {
            compute_cpu_->reset(active_params_, NBodyConfig::NBODY_CONFIG_SHELL);
        } else {
            compute_cuda_->reset(active_params_, NBodyConfig::NBODY_CONFIG_SHELL);
        }
    }

    demo_reset_time_ = Clock::now();
}

auto ComputeConfig::print_benchmark_results(int nb_iterations, float milliseconds) -> void {
    compute_perf_stats(nb_iterations * (1000.0f / milliseconds));

    std::println("{} bodies, total time for {} iterations: {:3} ms", num_bodies_, nb_iterations, milliseconds);
    std::println("= {:3} billion interactions per second", interactions_per_second_);

    std::println("= {:3} {}-precision GFLOP/s at {} flops per interaction", g_flops_, fp64_enabled_ ? "double" : "single", flops_per_interaction(fp64_enabled_));
}

constexpr auto ComputeConfig::compute_perf_stats(float frequency) -> void {
    // double precision uses intrinsic operation followed by refinement, resulting in higher operation count per interaction.
    // Note: Astrophysicists use 38 flops per interaction no matter what, based on "historical precedent", but they are using FLOP/s as a measure of "science throughput".
    // We are using it as a measure of hardware throughput.  They should really use interactions/s...
    interactions_per_second_ = (static_cast<float>(num_bodies_ * num_bodies_) * 1e-9f) * frequency;

    g_flops_ = interactions_per_second_ * static_cast<float>(flops_per_interaction(fp64_enabled_));
}

auto ComputeConfig::switch_precision() -> void {
    if (use_cpu_) {
        compute_cpu_->switch_precision();
    } else {
        compute_cuda_->switch_precision();
    }

    fp64_enabled_ = !fp64_enabled_;
}

auto ComputeConfig::toggle_cycle_demo() -> void {
    cycle_demo_ = !cycle_demo_;
    std::println("Cycle Demo Parameters: {}\n", cycle_demo_ ? "ON" : "OFF");
}

auto ComputeConfig::previous_demo(Camera& camera) -> void {
    if (active_demo_ == 0) {
        active_demo_ = nb_demos - 1;
    } else {
        --active_demo_;
    }
    select_demo(camera);
}

auto ComputeConfig::next_demo(Camera& camera) -> void {
    if (active_demo_ == (nb_demos - 1)) {
        active_demo_ = 0;
    } else {
        ++active_demo_;
    }
    select_demo(camera);
}

auto ComputeConfig::select_demo(Camera& camera) -> void {
    using enum NBodyConfig;

    assert(active_demo_ < nb_demos);

    active_params_ = demoParams[active_demo_];

    camera.reset(active_params_.camera_origin);

    if (tipsy_data_fp32_.positions.empty()) {
        if (use_cpu_) {
            compute_cpu_->reset(active_params_, NBODY_CONFIG_SHELL);
        } else {
            compute_cuda_->reset(active_params_, NBODY_CONFIG_SHELL);
        }
    } else {
        if (use_cpu_) {
            if (fp64_enabled_) {
                compute_cpu_->set_values(tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            } else {
                compute_cpu_->set_values(tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities);
            }
        } else {
            if (fp64_enabled_) {
                compute_cuda_->set_values(tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            } else {
                compute_cuda_->set_values(tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            }
        }
    }
    demo_reset_time_ = Clock::now();
}

auto ComputeConfig::update_simulation(Camera& camera) -> void {
    if (!paused_) {
        const auto current_demo_time = Clock::now() - demo_reset_time_;

        if (cycle_demo_ && (current_demo_time > demo_time)) {
            next_demo(camera);
        }

        if (use_cpu_) {
            compute_cpu_->update(active_params_.m_timestep);
        } else {
            compute_cuda_->update(active_params_.m_timestep);
        }
    }
}

auto ComputeConfig::display_NBody_system(Interface& interface) -> void {
    if (use_cpu_) {
        compute_cpu_->display(interface);
    } else {
        compute_cuda_->display(interface);
    }
}

auto ComputeConfig::reset(NBodyConfig initial_configuration) -> void {
    if (tipsy_data_fp32_.positions.empty()) {
        if (use_cpu_) {
            compute_cpu_->reset(active_params_, initial_configuration);
        } else {
            compute_cuda_->reset(active_params_, initial_configuration);
        }
    } else {
        if (use_cpu_) {
            if (fp64_enabled_) {
                compute_cpu_->set_values(tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            } else {
                compute_cpu_->set_values(tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities);
            }
        } else {
            if (fp64_enabled_) {
                compute_cuda_->set_values(tipsy_data_fp64_.positions, tipsy_data_fp64_.velocities);
            } else {
                compute_cuda_->set_values(tipsy_data_fp32_.positions, tipsy_data_fp32_.velocities);
            }
        }
    }
}

auto ComputeConfig::update_params() -> void {
    if (use_cpu_) {
        compute_cpu_->update_params(active_params_);
    } else {
        compute_cuda_->update_params(active_params_);
    }
}

auto ComputeConfig::get_milliseconds_passed() -> float {
    // stop timer
    if (use_cpu_) {
        return compute_cpu_->get_milliseconds_passed();
    }
    return compute_cuda_->get_milliseconds_passed();
}

auto ComputeConfig::calculate_fps(int frame_count) -> void {
    const auto milliseconds_passed = get_milliseconds_passed();

    const auto frequency = (1000.f / milliseconds_passed);
    fps_                 = static_cast<float>(frame_count) * frequency;

    compute_perf_stats(fps_);
}

auto ComputeConfig::run_benchmark(int nb_iterations) -> void {
    const auto milliseconds = use_cpu_ ? compute_cpu_->run_benchmark(nb_iterations, active_params_.m_timestep) : compute_cuda_->run_benchmark(nb_iterations, active_params_.m_timestep);

    print_benchmark_results(nb_iterations, milliseconds);
}

auto ComputeConfig::compare_results() -> bool {
    assert(compute_cuda_);
    return compute_cuda_->compare_results(active_params_);
}

auto ComputeConfig::add_modifiable_parameters(ParamListGL& param_list) -> void {
    // Velocity Damping
    param_list.add_param(std::make_unique<Param<float>>("Velocity Damping", active_params_.m_damping, 0.5f, 1.0f, .0001f, &active_params_.m_damping));
    // Softening Factor
    param_list.add_param(std::make_unique<Param<float>>("Softening Factor", active_params_.m_softening, 0.001f, 1.0f, .0001f, &active_params_.m_softening));
    // Time step size
    param_list.add_param(std::make_unique<Param<float>>("Time Step", active_params_.m_timestep, 0.0f, 1.0f, .0001f, &active_params_.m_timestep));
    // Cluster scale (only affects starting configuration
    param_list.add_param(std::make_unique<Param<float>>("Cluster Scale", active_params_.m_clusterScale, 0.0f, 10.0f, 0.01f, &active_params_.m_clusterScale));

    // Velocity scale (only affects starting configuration)
    param_list.add_param(std::make_unique<Param<float>>("Velocity Scale", active_params_.m_velocityScale, 0.0f, 1000.0f, 0.1f, &active_params_.m_velocityScale));
}

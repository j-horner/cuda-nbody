#include "compute_cuda.hpp"

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "bodysystemcuda_default.hpp"
#include "bodysystemcuda_graphics.hpp"
#include "bodysystemcuda_host_memory.hpp"
#include "helper_cuda.hpp"
#include "interface.hpp"

#include <cuda/api.hpp>

#include <print>

namespace {
// General GPU Device CUDA Initialization
auto initialise_gpu() -> cuda::device_t {
    auto device = cuda::device::current::get();

    using enum cuda::device::attribute_t;

    const auto compute_mode = device.get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);

    if (compute_mode == CU_COMPUTEMODE_PROHIBITED) {
        throw std::runtime_error("Error: device is running in <Compute Mode Prohibited>, no threads can use cudaSetDevice().\n");
    }

    const auto major = device.compute_capability().major();

    if (major < 1) {
        throw std::runtime_error("GPU device does not support CUDA.\n");
    }

    device.architecture().name();

    std::println("CUDA Device: \"{}\"\n", device.architecture().name());

    return device;
}

auto get_main_device() -> cuda::device_t {
    const auto nb_devices_available = cuda::device::count();

    if (nb_devices_available == 0) {
        throw std::runtime_error("gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
    }

    return initialise_gpu();
}

}    // namespace

ComputeCUDA::ComputeCUDA(bool enable_host_mem, bool use_pbo, int block_size, double fp64_enabled, std::size_t num_bodies, const NBodyParams& params)
    : ComputeCUDA(enable_host_mem, use_pbo, block_size, fp64_enabled, num_bodies, params, {}, {}, {}, {}) {}

ComputeCUDA::ComputeCUDA(
    bool                enable_host_mem,
    bool                use_pbo,
    int                 block_size,
    double              fp64_enabled,
    std::size_t         num_bodies,
    const NBodyParams&  params,
    std::vector<float>  positions_fp32,
    std::vector<float>  velocities_fp32,
    std::vector<double> positions_fp64,
    std::vector<double> velocities_fp64)
    : fp64_enabled_(fp64_enabled), use_host_mem_(enable_host_mem), use_pbo_(use_pbo), host_mem_sync_event_(cuda::event::create(cuda::device::current::get())), start_event_(cuda::event::create(cuda::device::current::get())),
      stop_event_(cuda::event::create(cuda::device::current::get())) {
    const auto main_device = cuda::device::current::get();

    const auto compute_capability = main_device.compute_capability();

    const auto major = compute_capability.major();
    const auto minor = compute_capability.minor();

    std::println("> Compute {}.{} CUDA device: [{}]", major, minor, main_device.name());

    if (use_host_mem_) {
        if (!main_device.properties().can_map_host_memory()) {
            throw std::invalid_argument(std::format("Device {} cannot map host memory!", main_device.name()));
        }

        const auto result = cudaSetDeviceFlags(cudaDeviceMapHost);

        if (result) {
            throw std::runtime_error(std::format("CUDA error. Could not set device flag cudaDeviceMapHost for device {}: {} - {}", main_device.name(), static_cast<unsigned int>(result), cudaGetErrorName(result)));
        }
    }

    // CC 1.2 and earlier do not support double precision
    if (major * 10 + minor <= 12) {
        double_supported_ = false;
    }

    if (fp64_enabled_ && (!double_supported_)) {
        throw std::invalid_argument("One or more of the requested devices does not support double precision floating-point");
    }

    if (num_bodies != 0u) {
        nb_bodies_ = num_bodies;

        assert(nb_bodies_ >= 1);

        if (nb_bodies_ % block_size) {
            auto new_nb_bodies = ((nb_bodies_ / block_size) + 1) * block_size;
            std::println(R"(Warning: "number of bodies" specified {} is not a multiple of {}.)", nb_bodies_, block_size);
            std::println("Rounding up to the nearest multiple: {}.", new_nb_bodies);
            nb_bodies_ = new_nb_bodies;
        } else {
            std::println("number of bodies = {}", nb_bodies_);
        }
    } else {
        // default number of bodies is #SMs * 4 * CTA size
        nb_bodies_ = num_bodies != 0 ? num_bodies : block_size * 4 * main_device.multiprocessor_count();
    }

    std::println("> Simulation data stored in {} memory", use_host_mem_ ? "system" : "video");
    std::println("> {} precision floating point simulation", fp64_enabled_ ? "Double" : "Single");

    const auto allocate_nbody = [&]<template <std::floating_point> typename BodySystem>() {
        const auto n_bodies = static_cast<unsigned int>(nb_bodies_);

        if (!positions_fp32.empty()) {
            nbody_fp32_ = std::make_unique<BodySystem<float>>(n_bodies, block_size, params, std::move(positions_fp32), std::move(velocities_fp32));

            if (double_supported_) {
                nbody_fp64_ = std::make_unique<BodySystem<double>>(n_bodies, block_size, params, std::move(positions_fp64), std::move(velocities_fp64));
            }
        } else {
            nbody_fp32_ = std::make_unique<BodySystem<float>>(n_bodies, block_size, params);

            if (double_supported_) {
                nbody_fp64_ = std::make_unique<BodySystem<double>>(n_bodies, block_size, params);
            }
        }
    };

    if (use_pbo_) {
        allocate_nbody.template operator()<BodySystemCUDAGraphics>();

        nbody_fp32_pbo_ = dynamic_cast<BodySystemCUDAGraphics<float>*>(nbody_fp32_.get());
        nbody_fp64_pbo_ = dynamic_cast<BodySystemCUDAGraphics<double>*>(nbody_fp64_.get());

    } else if (use_host_mem_) {
        allocate_nbody.template operator()<BodySystemCUDAHostMemory>();
    } else {
        allocate_nbody.template operator()<BodySystemCUDADefault>();
    }

    start_event_.record();
}

template <std::floating_point TNew, std::floating_point TOld> auto ComputeCUDA::switch_precision(BodySystemCUDA<TNew>& new_nbody, const BodySystemCUDA<TOld>& old_nbody) -> void {
    static_assert(!std::is_same_v<TNew, TOld>);

    cudaDeviceSynchronize();

    fp64_enabled_ = std::is_same_v<TNew, double>;

    const auto nb_bodies_4 = static_cast<std::size_t>(nb_bodies_ * 4);

    auto oldPos = std::vector<TOld>(nb_bodies_4);
    auto oldVel = std::vector<TOld>(nb_bodies_4);

    using std::ranges::copy;
    copy(old_nbody.get_position(), oldPos.begin());
    copy(old_nbody.get_velocity(), oldVel.begin());

    // convert float to double
    auto newPos = std::vector<TNew>(nb_bodies_4);
    auto newVel = std::vector<TNew>(nb_bodies_4);

    for (int i = 0; i < nb_bodies_4; i++) {
        newPos[i] = static_cast<TNew>(oldPos[i]);
        newVel[i] = static_cast<TNew>(oldVel[i]);
    }

    new_nbody.set_position(newPos);
    new_nbody.set_velocity(newVel);

    cudaDeviceSynchronize();
}

template <std::floating_point T> auto ComputeCUDA::run_benchmark(int nb_iterations, float dt, BodySystemCUDA<T>& nbody) -> Milliseconds {
    // once without timing to prime the device

    nbody.update(dt);

    start_event_.record();

    for (int i = 0; i < nb_iterations; ++i) {
        nbody.update(dt);
    }

    return get_milliseconds_passed();
}

auto ComputeCUDA::run_benchmark(int nb_iterations, float dt) -> Milliseconds {
    if (fp64_enabled_) {
        return run_benchmark(nb_iterations, dt, *nbody_fp64_);
    } else {
        return run_benchmark(nb_iterations, dt, *nbody_fp32_);
    }
}

auto ComputeCUDA::switch_precision() -> void {
    if (!double_supported_) {
        std::println(stderr, "WARNING: Attempted to switch precision but double precision is not supported.");
        return;
    }

    if (fp64_enabled_) {
        switch_precision(*nbody_fp32_, *nbody_fp64_);
        std::println("> Double precision floating point simulation");
    } else {
        switch_precision(*nbody_fp64_, *nbody_fp32_);
        std::println("> Single precision floating point simulation");
    }
}

auto ComputeCUDA::reset(const NBodyParams& params, NBodyConfig config) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->reset(params, config);
    } else {
        nbody_fp32_->reset(params, config);
    }
}

auto ComputeCUDA::set_values(std::span<const float> positions, std::span<const float> velocities) -> void {
    nbody_fp32_->set_position(positions);
    nbody_fp32_->set_velocity(velocities);
}
auto ComputeCUDA::set_values(std::span<const double> positions, std::span<const double> velocities) -> void {
    nbody_fp64_->set_position(positions);
    nbody_fp64_->set_velocity(velocities);
}

auto ComputeCUDA::update(float dt) -> void {
    // insert an event to wait on before rendering
    host_mem_sync_event_.record();

    if (fp64_enabled_) {
        nbody_fp64_->update(dt);
    } else {
        nbody_fp32_->update(dt);
    }
}

auto ComputeCUDA::get_position_fp32() const noexcept -> std::span<const float> {
    return nbody_fp32_->get_position();
}
auto ComputeCUDA::get_position_fp64() const noexcept -> std::span<const double> {
    return nbody_fp64_->get_position();
}

auto ComputeCUDA::update_params(const NBodyParams& params) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->update_params(params);
    } else {
        nbody_fp32_->update_params(params);
    }
}

auto ComputeCUDA::get_milliseconds_passed() -> Milliseconds {
    stop_event_.record();
    stop_event_.synchronize();

    const auto milliseconds = cuda::event::time_elapsed_between(start_event_, stop_event_);

    start_event_.record();

    return milliseconds;
}

auto ComputeCUDA::display(Interface& interface) const -> void {
    if (use_pbo_) {
        if (fp64_enabled_) {
            interface.display_nbody_system_fp64(nbody_fp64_pbo_->getCurrentReadBuffer());
        } else {
            interface.display_nbody_system_fp32(nbody_fp32_pbo_->getCurrentReadBuffer());
        }
    } else {
        // This event sync is required because we are rendering from the host memory that CUDA is writing.
        // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
        host_mem_sync_event_.synchronize();

        if (fp64_enabled_) {
            interface.display_nbody_system(nbody_fp64_->get_position());
        } else {
            interface.display_nbody_system(nbody_fp32_->get_position());
        }
    }
}

template <std::floating_point T> auto ComputeCUDA::compare_results(const NBodyParams& params, BodySystemCUDA<T>& nbodyCuda) const -> bool {
    auto passed = true;

    nbodyCuda.update(0.001f);

    {
        auto nbodyCpu = BodySystemCPU<T>(nb_bodies_, params);

        nbodyCpu.set_position(nbodyCuda.get_position());
        nbodyCpu.set_velocity(nbodyCuda.get_velocity());

        nbodyCpu.update(0.001f);

        const auto cudaPos = nbodyCuda.get_position();
        const auto cpuPos  = nbodyCpu.get_position();

        constexpr auto tolerance = T{0.0005f};

        for (int i = 0; i < nb_bodies_; i++) {
            if (std::abs(cpuPos[i] - cudaPos[i]) > tolerance) {
                passed = false;
                std::println("Error: (host){} != (device){}", cpuPos[i], cudaPos[i]);
            }
        }
    }
    if (passed) {
        std::println("  OK");
    }
    return passed;
}

auto ComputeCUDA::compare_results(const NBodyParams& params) -> bool {
    return fp64_enabled_ ? compare_results(params, *nbody_fp64_) : compare_results(params, *nbody_fp32_);
}

ComputeCUDA::~ComputeCUDA() noexcept = default;
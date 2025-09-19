#include "compute_cuda.hpp"

#include "bodysystemcpu.hpp"
#include "bodysystemcuda.hpp"
#include "helper_cuda.hpp"
#include "interface.hpp"

#include <print>

ComputeCUDA::ComputeCUDA(std::size_t nb_requested_devices, bool enable_host_mem, bool use_pbo, int device, std::size_t block_size, double fp64_enabled, std::size_t num_bodies, const NBodyParams& params)
    : ComputeCUDA(nb_requested_devices, enable_host_mem, use_pbo, device, block_size, fp64_enabled, num_bodies, params, {}, {}, {}, {}) {}

ComputeCUDA::ComputeCUDA(
    std::size_t         nb_requested_devices,
    bool                enable_host_mem,
    bool                use_pbo,
    int                 device,
    std::size_t         block_size,
    double              fp64_enabled,
    std::size_t         num_bodies,
    const NBodyParams&  params,
    std::vector<float>  positions_fp32,
    std::vector<float>  velocities_fp32,
    std::vector<double> positions_fp64,
    std::vector<double> velocities_fp64)
    : fp64_enabled_(fp64_enabled), use_host_mem_(enable_host_mem), use_pbo_(use_pbo) {
    assert(!(use_host_mem_ && use_pbo_));

    auto nb_devices_requested = 1;

    if (nb_requested_devices > 0) {
        nb_devices_requested = static_cast<int>(nb_requested_devices);
        std::println("number of CUDA devices  = {}", nb_devices_requested);
    }

    {
        auto nb_devices_available = 0;
        cudaGetDeviceCount(&nb_devices_available);

        if (nb_devices_available < nb_devices_requested) {
            throw std::invalid_argument(std::format("Error: only {} Devices available, {} requested.", nb_devices_available, nb_devices_requested));
        }
    }

    auto use_p2p = true;    // this is always optimal to use P2P path when available

    if (nb_devices_requested > 1) {
        // If user did not explicitly request host memory to be used (false by default), we default to P2P.
        // We fallback to host memory, if any of GPUs does not support P2P.
        if (!enable_host_mem) {
            auto all_gpus_support_p2p = true;
            // Enable P2P only in one direction, as every peer will access gpu0
            for (auto i = 1; i < nb_devices_requested; ++i) {
                auto canAccessPeer = 0;
                checkCudaErrors(cudaDeviceCanAccessPeer(&canAccessPeer, i, 0));

                if (canAccessPeer != 1) {
                    all_gpus_support_p2p = false;
                }
            }

            if (!all_gpus_support_p2p) {
                use_host_mem_ = true;
                use_p2p       = false;
            }
        }
    }

    std::println("> Simulation data stored in {} memory", use_host_mem_ ? "system" : "video");
    std::println("> {} precision floating point simulation", fp64_enabled_ ? "Double" : "Single");
    std::println("> {} Devices used for simulation", nb_devices_requested);

    auto dev_id = 0;

    auto custom_gpu = false;

    auto cuda_properties = cudaDeviceProp{};

    if (device != -1) {
        custom_gpu = true;
    }

    // If the command-line has a device number specified, use it
    if (custom_gpu) {
        dev_id = device;
        assert(dev_id >= 0);

        const auto new_dev_ID = gpuDeviceInit(dev_id);

        if (new_dev_ID < 0) {
            throw std::invalid_argument(std::format("Could not use custom CUDA device: {}", dev_id));
        }

        dev_id = new_dev_ID;

    } else {
        // Otherwise pick the device with highest Gflops/s
        dev_id = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(dev_id));
        int major = 0, minor = 0;
        checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id));
        checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev_id));
        std::println(R"(GPU Device {}: "{}" with compute capability {}.{}\n)", dev_id, _ConvertSMVer2ArchName(major, minor), major, minor);
    }

    checkCudaErrors(cudaGetDevice(&dev_id));
    checkCudaErrors(cudaGetDeviceProperties(&cuda_properties, dev_id));

    // Initialize devices
    assert(!(custom_gpu && (nb_devices_requested > 1)));

    if (custom_gpu || nb_devices_requested == 1) {
        auto properties = cudaDeviceProp{};
        checkCudaErrors(cudaGetDeviceProperties(&properties, dev_id));
        std::println("> Compute {}.{} CUDA device: [{}]", properties.major, properties.minor, properties.name);
        // CC 1.2 and earlier do not support double precision
        if (properties.major * 10 + properties.minor <= 12) {
            double_supported_ = false;
        }

    } else {
        for (int i = 0; i < nb_devices_requested; i++) {
            auto properties = cudaDeviceProp{};
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));

            std::println("> Compute {}.{} CUDA device: [{}]", properties.major, properties.minor, properties.name);

            if (use_host_mem_) {
                if (!properties.canMapHostMemory) {
                    throw std::invalid_argument(std::format("Device {} cannot map host memory!", i));
                }

                if (nb_devices_requested > 1) {
                    checkCudaErrors(cudaSetDevice(i));
                }

                checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
            }

            // CC 1.2 and earlier do not support double precision
            if (properties.major * 10 + properties.minor <= 12) {
                double_supported_ = false;
            }
        }
    }

    if (fp64_enabled_ && !double_supported_) {
        throw std::invalid_argument("One or more of the requested devices does not support double precision floating-point");
    }

    auto blockSize = static_cast<int>(block_size);

    // default number of bodies is #SMs * 4 * CTA size
    if (nb_devices_requested == 1) {
        num_bodies_ = num_bodies != 0 ? num_bodies : blockSize * 4 * cuda_properties.multiProcessorCount;
    } else {
        num_bodies_ = 0;
        for (auto i = 0; i < nb_devices_requested; ++i) {
            auto properties = cudaDeviceProp{};
            checkCudaErrors(cudaGetDeviceProperties(&properties, i));
            num_bodies_ += blockSize * (properties.major >= 2 ? 4 : 1) * properties.multiProcessorCount;
        }
    }

    if (num_bodies != 0u) {
        num_bodies_ = num_bodies;

        assert(num_bodies_ >= 1);

        if (num_bodies_ % blockSize) {
            auto new_nb_bodies = ((num_bodies_ / blockSize) + 1) * blockSize;
            std::println(R"(Warning: "number of bodies" specified {} is not a multiple of {}.)", num_bodies_, blockSize);
            std::println("Rounding up to the nearest multiple: {}.", new_nb_bodies);
            num_bodies_ = new_nb_bodies;
        } else {
            std::println("number of bodies = {}", num_bodies_);
        }
    }

    if (!positions_fp32.empty()) {
        nbody_fp32_ = std::make_unique<BodySystemCUDA<float>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, params, std::move(positions_fp32), std::move(velocities_fp32));

        if (double_supported_) {
            nbody_fp64_ = std::make_unique<BodySystemCUDA<double>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, params, std::move(positions_fp64), std::move(velocities_fp64));
        }
    } else {
        nbody_fp32_ = std::make_unique<BodySystemCUDA<float>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, params);

        if (double_supported_) {
            nbody_fp64_ = std::make_unique<BodySystemCUDA<double>>(*this, nb_devices_requested, blockSize, use_p2p, dev_id, params);
        }
    }
    checkCudaErrors(cudaEventCreate(&start_event_));
    checkCudaErrors(cudaEventCreate(&stop_event_));
    checkCudaErrors(cudaEventCreate(&host_mem_sync_event_));
    checkCudaErrors(cudaEventRecord(start_event_, 0));
}

template <std::floating_point TNew, std::floating_point TOld> auto ComputeCUDA::switch_precision(BodySystemCUDA<TNew>& new_nbody, const BodySystemCUDA<TOld>& old_nbody) -> void {
    static_assert(!std::is_same_v<TNew, TOld>);

    cudaDeviceSynchronize();

    fp64_enabled_ = std::is_same_v<TNew, double>;

    const auto nb_bodies_4 = static_cast<std::size_t>(num_bodies_ * 4);

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

template <std::floating_point T> auto ComputeCUDA::run_benchmark(int nb_iterations, float dt, BodySystemCUDA<T>& nbody) -> float {
    // once without timing to prime the device

    nbody.update(dt);

    auto milliseconds = 0.f;

    checkCudaErrors(cudaEventRecord(start_event_, 0));

    for (int i = 0; i < nb_iterations; ++i) {
        nbody.update(dt);
    }

    checkCudaErrors(cudaEventRecord(stop_event_, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event_));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));

    return milliseconds;
}

auto ComputeCUDA::run_benchmark(int nb_iterations, float dt) -> float {
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
    cudaEventRecord(host_mem_sync_event_);    // insert an event to wait on before rendering

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

auto ComputeCUDA::get_milliseconds_passed() -> float {
    auto milliseconds = 0.f;

    checkCudaErrors(cudaEventRecord(stop_event_, 0));
    checkCudaErrors(cudaEventSynchronize(stop_event_));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    checkCudaErrors(cudaEventRecord(start_event_, 0));

    return milliseconds;
}

auto ComputeCUDA::display(Interface& interface) const -> void {
    if (use_pbo_) {
        if (fp64_enabled_) {
            interface.display_nbody_system_fp64(nbody_fp64_->getCurrentReadBuffer());
        } else {
            interface.display_nbody_system_fp32(nbody_fp32_->getCurrentReadBuffer());
        }
    } else {
        // This event sync is required because we are rendering from the host memory that CUDA is writing.
        // If we don't wait until CUDA is done updating it, we will render partially updated data, resulting in a jerky frame rate.
        cudaEventSynchronize(host_mem_sync_event_);

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
        auto nbodyCpu = BodySystemCPU<T>(num_bodies_, params);

        nbodyCpu.set_position(nbodyCuda.get_position());
        nbodyCpu.set_velocity(nbodyCuda.get_velocity());

        nbodyCpu.update(0.001f);

        const auto cudaPos = nbodyCuda.get_position();
        const auto cpuPos  = nbodyCpu.get_position();

        constexpr auto tolerance = T{0.0005f};

        for (int i = 0; i < num_bodies_; i++) {
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

ComputeCUDA::~ComputeCUDA() noexcept {
    checkCudaErrors(cudaEventDestroy(start_event_));
    checkCudaErrors(cudaEventDestroy(stop_event_));
    checkCudaErrors(cudaEventDestroy(host_mem_sync_event_));
}
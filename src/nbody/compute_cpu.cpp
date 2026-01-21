#include "compute_cpu.hpp"

#include "bodysystemcpu.hpp"
#include "interface.hpp"

#include <print>

ComputeCPU::ComputeCPU(double fp64_enabled, std::size_t num_bodies, const NBodyParams& params) : ComputeCPU(fp64_enabled, num_bodies, params, {}, {}, {}, {}) {}

ComputeCPU::ComputeCPU(double              fp64_enabled,
                       std::size_t         num_bodies,
                       const NBodyParams&  params,
                       std::vector<float>  positions_fp32,
                       std::vector<float>  velocities_fp32,
                       std::vector<double> positions_fp64,
                       std::vector<double> velocities_fp64)
    : fp64_enabled_(fp64_enabled) {
#ifdef OPENMP
    std::println("> Simulation with CPU using OpenMP");
#else
    std::println("> Simulation with CPU");
#endif

    std::println("> Simulation data stored in system memory");
    std::println("> {} precision floating point simulation", fp64_enabled ? "Double" : "Single");
    std::println("> 0 Devices used for simulation");

    nb_bodies_ = 8192;

    if (num_bodies != 0u) {
        nb_bodies_ = num_bodies;
        std::println("number of bodies = {}", nb_bodies_);
    }

    if (!positions_fp64.empty()) {
        nbody_fp32_ = std::make_unique<BodySystemCPU<float>>(nb_bodies_, params, std::move(positions_fp32), std::move(velocities_fp32));
        nbody_fp64_ = std::make_unique<BodySystemCPU<double>>(nb_bodies_, params, std::move(positions_fp64), std::move(velocities_fp64));
    } else {
        nbody_fp32_ = std::make_unique<BodySystemCPU<float>>(nb_bodies_, params);
        nbody_fp64_ = std::make_unique<BodySystemCPU<double>>(nb_bodies_, params);
    }

    reset_time_ = Clock::now();
}

template <std::floating_point TNew, std::floating_point TOld> auto ComputeCPU::switch_precision(BodySystemCPU<TNew>& new_nbody, const BodySystemCPU<TOld>& old_nbody) -> void {
    static_assert(!std::is_same_v<TNew, TOld>);

    fp64_enabled_ = std::is_same_v<TNew, double>;

    const auto& old_pos = old_nbody.positions();
    const auto& old_vel = old_nbody.velocities();

    // convert float to double
    auto& new_pos = new_nbody.positions();
    auto& new_vel = new_nbody.velocities();

    for (auto i = std::size_t{0u}; i < nb_bodies_; ++i) {
        new_pos.x[i] = static_cast<TNew>(old_pos.x[i]);
        new_pos.y[i] = static_cast<TNew>(old_pos.y[i]);
        new_pos.z[i] = static_cast<TNew>(old_pos.z[i]);
        new_vel.x[i] = static_cast<TNew>(old_vel.x[i]);
        new_vel.y[i] = static_cast<TNew>(old_vel.y[i]);
        new_vel.z[i] = static_cast<TNew>(old_vel.z[i]);
    }
}

template <std::floating_point T> auto ComputeCPU::run_benchmark(int nb_iterations, float dt, BodySystemCPU<T>& nbody) -> Milliseconds {
    const auto start = Clock::now();

    for (int i = 0; i < nb_iterations; ++i) {
        nbody.update(dt);
    }

    return Milliseconds{Clock::now() - start};
}

auto ComputeCPU::run_benchmark(int nb_iterations, float dt) -> Milliseconds {
    if (fp64_enabled_) {
        return run_benchmark(nb_iterations, dt, *nbody_fp64_);
    } else {
        return run_benchmark(nb_iterations, dt, *nbody_fp32_);
    }
}

auto ComputeCPU::switch_precision() -> void {
    if (fp64_enabled_) {
        switch_precision(*nbody_fp32_, *nbody_fp64_);
        std::println("> Double precision floating point simulation");
        return;
    }
    switch_precision(*nbody_fp64_, *nbody_fp32_);
    std::println("> Single precision floating point simulation");
}

auto ComputeCPU::reset(const NBodyParams& params, NBodyConfig config) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->reset(params, config);
    } else {
        nbody_fp32_->reset(params, config);
    }
}

auto ComputeCPU::set_values(std::span<const float> positions, std::span<const float> velocities) -> void {
    nbody_fp32_->set_position(positions);
    nbody_fp32_->set_velocity(velocities);
}
auto ComputeCPU::set_values(std::span<const double> positions, std::span<const double> velocities) -> void {
    nbody_fp64_->set_position(positions);
    nbody_fp64_->set_velocity(velocities);
}

auto ComputeCPU::update(float dt) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->update(dt);
    } else {
        nbody_fp32_->update(dt);
    }
}

auto ComputeCPU::update_params(const NBodyParams& params) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->update_params(params);
    } else {
        nbody_fp32_->update_params(params);
    }
}

auto ComputeCPU::get_milliseconds_passed() -> Milliseconds {
    const auto now          = Clock::now();
    const auto milliseconds = Milliseconds{now - reset_time_};

    reset_time_ = now;

    return milliseconds;
}

auto ComputeCPU::display(Interface& interface) const -> void {
    if (fp64_enabled_) {
        interface.display_nbody_system(const_cast<const BodySystemCPU<double>&>(*nbody_fp64_).positions());
    } else {
        interface.display_nbody_system(const_cast<const BodySystemCPU<float>&>(*nbody_fp32_).positions());
    }
}

ComputeCPU::~ComputeCPU() noexcept = default;
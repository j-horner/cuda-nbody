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

#ifdef OPENMP
    num_bodies_ = 8192;
#else
    num_bodies_ = 4096;
#endif

    if (num_bodies != 0u) {
        num_bodies_ = num_bodies;
        std::println("number of bodies = {}", num_bodies_);
    }

    if (!positions_fp64.empty()) {
        nbody_fp32_ = std::make_unique<BodySystemCPU<float>>(num_bodies_, params, std::move(positions_fp32), std::move(velocities_fp32));
        nbody_fp64_ = std::make_unique<BodySystemCPU<double>>(num_bodies_, params, std::move(positions_fp64), std::move(velocities_fp64));
    } else {
        nbody_fp32_ = std::make_unique<BodySystemCPU<float>>(num_bodies_, params);
        nbody_fp64_ = std::make_unique<BodySystemCPU<double>>(num_bodies_, params);
    }

    reset_time_ = Clock::now();
}

template <std::floating_point TNew, std::floating_point TOld> auto ComputeCPU::switch_precision(BodySystemCPU<TNew>& new_nbody, const BodySystemCPU<TOld>& old_nbody) -> void {
    static_assert(!std::is_same_v<TNew, TOld>);

    fp64_enabled_ = std::is_same_v<TNew, double>;

    const auto nb_bodies_4 = static_cast<std::size_t>(num_bodies_ * 4);

    const auto old_pos = old_nbody.get_position();
    const auto old_vel = old_nbody.get_velocity();

    // convert float to double
    const auto new_pos = new_nbody.get_position();
    const auto new_vel = new_nbody.get_velocity();

    for (int i = 0; i < nb_bodies_4; i++) {
        new_pos[i] = static_cast<TNew>(old_pos[i]);
        new_vel[i] = static_cast<TNew>(old_vel[i]);
    }
}

template <std::floating_point T> auto ComputeCPU::run_benchmark(int nb_iterations, float dt, BodySystemCPU<T>& nbody) -> float {
    const auto start = Clock::now();

    for (int i = 0; i < nb_iterations; ++i) {
        nbody.update(dt);
    }

    return MilliSeconds{Clock::now() - start}.count();
}

auto ComputeCPU::run_benchmark(int nb_iterations, float dt) -> float {
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

auto ComputeCPU::get_position_fp32() const noexcept -> std::span<const float> {
    return nbody_fp32_->get_position();
}
auto ComputeCPU::get_position_fp64() const noexcept -> std::span<const double> {
    return nbody_fp64_->get_position();
}

auto ComputeCPU::update_params(const NBodyParams& params) -> void {
    if (fp64_enabled_) {
        nbody_fp64_->update_params(params);
    } else {
        nbody_fp32_->update_params(params);
    }
}

auto ComputeCPU::get_milliseconds_passed() -> float {
    const auto now          = Clock::now();
    const auto milliseconds = MilliSeconds{now - reset_time_}.count();

    reset_time_ = now;

    return milliseconds;
}

auto ComputeCPU::display(Interface& interface) const -> void {
    if (fp64_enabled_) {
        interface.display_nbody_system(nbody_fp64_->get_position());
    } else {
        interface.display_nbody_system(nbody_fp32_->get_position());
    }
}

ComputeCPU::~ComputeCPU() noexcept = default;
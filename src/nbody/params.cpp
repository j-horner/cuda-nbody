#include "params.hpp"

#include <print>

auto NBodyParams::print() const -> void {
    std::println("{{ {}, {}, {}, {}, {}, {}, {}, {} }},", time_step, cluster_scale, velocity_scale, softening, damping, camera_origin[0], camera_origin[1], camera_origin[2]);
}

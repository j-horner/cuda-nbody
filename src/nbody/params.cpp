#include "params.hpp"

#include <print>

auto NBodyParams::print() const -> void {
    std::println("{{ {}, {}, {}, {}, {}, {}, {}, {} }},", m_timestep, m_clusterScale, m_velocityScale, m_softening, m_damping, camera_origin[0], camera_origin[1], camera_origin[2]);
}

#include "params.hpp"

#include "param.hpp"
#include "paramgl.hpp"

#include <memory>
#include <print>

auto NBodyParams::print() const -> void {
    std::println("{{ {}, {}, {}, {}, {}, {}, {}, {} }},", m_timestep, m_clusterScale, m_velocityScale, m_softening, m_damping, camera_origin[0], camera_origin[1], camera_origin[2]);
}

auto NBodyParams::create_sliders() -> ParamListGL {
    // create a new parameter list
    auto paramlist = ParamListGL{};

    // add some parameters to the list

    // Velocity Damping
    paramlist.add_param(std::make_unique<Param<float>>("Velocity Damping", m_damping, 0.5f, 1.0f, .0001f, &m_damping));
    // Softening Factor
    paramlist.add_param(std::make_unique<Param<float>>("Softening Factor", m_softening, 0.001f, 1.0f, .0001f, &m_softening));
    // Time step size
    paramlist.add_param(std::make_unique<Param<float>>("Time Step", m_timestep, 0.0f, 1.0f, .0001f, &m_timestep));
    // Cluster scale (only affects starting configuration
    paramlist.add_param(std::make_unique<Param<float>>("Cluster Scale", m_clusterScale, 0.0f, 10.0f, 0.01f, &m_clusterScale));

    // Velocity scale (only affects starting configuration)
    paramlist.add_param(std::make_unique<Param<float>>("Velocity Scale", m_velocityScale, 0.0f, 1000.0f, 0.1f, &m_velocityScale));

    return paramlist;
}
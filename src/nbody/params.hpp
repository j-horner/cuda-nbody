#pragma once

#include <array>

////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams {
    float                m_timestep;
    float                m_clusterScale;
    float                m_velocityScale;
    float                m_softening;
    float                m_damping;
    std::array<float, 3> camera_origin;

    auto print() const -> void;
};
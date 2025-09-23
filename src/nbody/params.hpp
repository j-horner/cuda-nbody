#pragma once

#include <array>

////////////////////////////////////////
// Demo Parameters
////////////////////////////////////////
struct NBodyParams {
    float                time_step;
    float                cluster_scale;
    float                velocity_scale;
    float                softening;
    float                damping;
    std::array<float, 3> camera_origin;

    auto print() const -> void;
};
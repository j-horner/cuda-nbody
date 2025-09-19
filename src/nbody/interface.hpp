#pragma once

#include "paramgl.hpp"
#include "render_particles.hpp"

#include <span>

class ParamListGL;
class Camera;
class ComputeConfig;

class Interface {
 public:
    Interface(bool display_sliders, ParamListGL parameters, bool enable_fullscreen, ParticleRenderer renderer) noexcept;

    auto toggle_sliders() noexcept -> void { show_sliders_ = !show_sliders_; }
    auto toggle_interactions() noexcept -> void { display_interactions = !display_interactions; }
    auto cycle_display_mode() noexcept -> void { display_mode = (ParticleRenderer::DisplayMode)((display_mode + 1) % ParticleRenderer::PARTICLE_NUM_MODES); }
    auto togle_display() noexcept -> void { display_enabled = !display_enabled; }

    auto display(ComputeConfig& compute, Camera& camera) -> void;

    auto is_mouse_over_sliders(int x, int y) noexcept -> bool { return show_sliders_ && param_list.is_mouse_over(x, y); }

    auto modify_sliders(int button, int state, int x, int y) -> void { param_list.modify_sliders(button, state, x, y); }

    auto motion(int x, int y) const -> bool { return param_list.motion(x, y); }

    auto show_sliders() const noexcept { return show_sliders_; }

    // The special keyboard callback is triggered when keyboard function or directional keys are pressed.
    auto special(int key, int x, int y) -> void;

    auto display_nbody_system(std::span<const float> positions) -> void;
    auto display_nbody_system(std::span<const double> positions) -> void;

    auto display_nbody_system_fp32(unsigned int pbo) -> void;
    auto display_nbody_system_fp64(unsigned int pbo) -> void;

 private:
    bool                          display_enabled = true;
    bool                          show_sliders_;
    ParamListGL                   param_list;
    bool                          full_screen;
    bool                          display_interactions = false;
    ParticleRenderer::DisplayMode display_mode         = ParticleRenderer::PARTICLE_SPRITES_COLOR;
    int                           frame_count          = 0;
    int                           fps_limit            = 5;
    float                         point_size_          = 1.0f;
    ParticleRenderer              renderer_;
};
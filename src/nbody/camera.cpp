#include "camera.hpp"

#include "gl_includes.hpp"

#include <cmath>

auto Camera::zoom(float dy) noexcept -> void {
    translation_[2] += (dy / 100.0f) * 0.5f * std::abs(translation_[2]);
}

auto Camera::view_transform() noexcept -> void {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    static auto           camera_rot_lag = std::array{0.f, 0.f, 0.f};
    constexpr static auto inertia        = 0.1f;

    for (int c = 0; c < 3; ++c) {
        translation_lag_[c] += (translation_[c] - translation_lag_[c]) * inertia;
        camera_rot_lag[c] += (rotation_[c] - camera_rot_lag[c]) * inertia;
    }

    glTranslatef(translation_lag_[0], translation_lag_[1], translation_lag_[2]);
    glRotatef(camera_rot_lag[0], 1.0, 0.0, 0.0);
    glRotatef(camera_rot_lag[1], 0.0, 1.0, 0.0);
}

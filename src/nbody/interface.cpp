#include "interface.hpp"

#include "camera.hpp"
#include "compute.hpp"
#include "gl_includes.hpp"
#include "gl_print.hpp"
#include "helper_cuda.hpp"
#include "paramgl.hpp"
#include "render_particles.hpp"
#include "win_coords.hpp"

#include <GL/freeglut.h>

#include <format>
#include <memory>

Interface::Interface(bool display_sliders, ParamListGL parameters, bool enable_fullscreen, ParticleRenderer renderer) noexcept
    : show_sliders_(display_sliders), param_list(std::move(parameters)), full_screen(enable_fullscreen), renderer_(std::move(renderer)) {
    param_list.add_param(std::make_unique<Param<float>>("Point Size", point_size_, 0.001f, 10.0f, 0.01f, &point_size_));
}

auto Interface::display(ComputeConfig& compute, Camera& camera) -> void {
    compute.update_simulation(camera);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (display_enabled) {
        camera.view_transform();

        compute.display_NBody_system(*this);

        // display user interface
        if (show_sliders_) {
            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            param_list.render();
            glDisable(GL_BLEND);
        }

        if (full_screen) {
            auto dev_id = -1;

            checkCudaErrors(cudaGetDevice(&dev_id));

            auto properties = cudaDeviceProp{};
            checkCudaErrors(cudaGetDeviceProperties(&properties, dev_id));

            const auto win_coords = WinCoords{};

            const auto msg1 = display_interactions ? std::format("{:.2f} billion interactions per second", compute.interactions_per_second()) : std::format("{:.2f} GFLOP/s", compute.gflops());

            const auto msg2 = std::format("{:.2f} FPS [{} | {} bodies]", compute.fps(), compute.fp64_enabled() ? "double precision" : "single precision", compute.nb_bodies());

            glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ZERO);    // invert color
            glEnable(GL_BLEND);
            glColor3f(0.46f, 0.73f, 0.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 122, properties.name, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 96, msg2, GLUT_BITMAP_TIMES_ROMAN_24);
            glColor3f(1.0f, 1.0f, 1.0f);
            glPrint(80, glutGet(GLUT_WINDOW_HEIGHT) - 70, msg1, GLUT_BITMAP_TIMES_ROMAN_24);
            glDisable(GL_BLEND);
        }

        glutSwapBuffers();
    }

    ++frame_count;

    // this displays the frame rate updated every second (independent of frame rate)
    if (frame_count >= fps_limit) {
        compute.calculate_fps(frame_count);

        const auto fps_str = std::format(
            "CUDA N-Body ({} bodies): {:.1f} fps | {:.1f} BIPS | {:.1f} GFLOP/s | {}",
            compute.nb_bodies(),
            compute.fps(),
            compute.interactions_per_second(),
            compute.gflops(),
            compute.fp64_enabled() ? "double precision" : "single precision");

        glutSetWindowTitle(fps_str.c_str());
        frame_count = 0;

        if (compute.paused()) {
            fps_limit = 0;
        } else if (compute.fps() > 1.f) {
            // setting the refresh limit (in number of frames) to be the FPS value obviously refreshes this message every second...
            fps_limit = static_cast<int>(compute.fps());
        } else {
            fps_limit = 1;
        }
    }

    glutReportErrors();
}

auto Interface::special(int key, int x, int y) -> void {
    param_list.special(key, x, y);
    glutPostRedisplay();
}

auto Interface::display_nbody_system(std::span<const float> positions) -> void {
    renderer_.display(display_mode, point_size_, positions);
}
auto Interface::display_nbody_system(std::span<const double> positions) -> void {
    renderer_.display(display_mode, point_size_, positions);
}

auto Interface::display_nbody_system_fp32(unsigned int pbo) -> void {
    renderer_.display<float>(display_mode, point_size_, pbo);
}
auto Interface::display_nbody_system_fp64(unsigned int pbo) -> void {
    renderer_.display<double>(display_mode, point_size_, pbo);
}
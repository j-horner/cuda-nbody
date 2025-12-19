#include "controls.hpp"

#include "camera.hpp"
#include "compute.hpp"
#include "interface.hpp"

#include <GL/freeglut.h>

auto Controls::set_state(int button, int state, int x, int y) noexcept -> void {
    if (state == GLUT_DOWN) {
        button_state_ |= 1 << button;
    } else if (state == GLUT_UP) {
        button_state_ = 0;
    }

    const auto mods = glutGetModifiers();

    if (mods & GLUT_ACTIVE_SHIFT) {
        button_state_ = 2;
    } else if (mods & GLUT_ACTIVE_CTRL) {
        button_state_ = 3;
    }

    old_x_ = x;
    old_y_ = y;
}

auto Controls::move_camera(Camera& camera, int x, int y) noexcept -> void {
    const auto dx = static_cast<float>(x - old_x_);
    const auto dy = static_cast<float>(y - old_y_);

    if (button_state_ == 3) {
        // left+middle = zoom
        camera.zoom(dy);
    } else if (button_state_ & 2) {
        // middle = translate
        camera.translate(dx, dy);
    } else if (button_state_ & 1) {
        // left = rotate
        camera.rotate(dx, dy);
    }

    old_x_ = x;
    old_y_ = y;
}

auto Controls::mouse(int button, int state, int x, int y, Interface& interface, Compute& compute) -> void {
    if (interface.is_mouse_over_sliders(x, y)) {
        // call list mouse function
        interface.modify_sliders(button, state, x, y);
        compute.update_params();
    }

    set_state(button, state, x, y);

    glutPostRedisplay();
}

auto Controls::motion(int x, int y, const Interface& interface, Camera& camera, Compute& compute) -> void {
    if (interface.show_sliders()) {
        // call parameter list motion function
        if (interface.motion(x, y)) {
            // by definition of this function, a mouse function is pressed so we need to update the parameters
            compute.update_params();
            glutPostRedisplay();
            return;
        }
    }

    move_camera(camera, x, y);

    glutPostRedisplay();
}

auto Controls::keyboard(unsigned char key, [[maybe_unused]] int x, [[maybe_unused]] int y, Compute& compute, Interface& interface, Camera& camera) -> bool {
    using enum NBodyConfig;

    switch (key) {
        case ' ':
            compute.pause();
            break;

        case 27:    // escape
        case 'q':
        case 'Q':
            {
                assert(glGetError() == GL_NO_ERROR);
                glutLeaveMainLoop();
                return true;
            }

        case 13:    // return
            compute.switch_precision();
            break;

        case '`':
            interface.toggle_sliders();
            break;

        case 'g':
        case 'G':
            interface.toggle_interactions();
            break;

        case 'p':
        case 'P':
            interface.cycle_display_mode();
            break;

        case 'c':
        case 'C':
            compute.toggle_cycle_demo();
            break;

        case '[':
            compute.previous_demo(camera);
            break;

        case ']':
            compute.next_demo(camera);
            break;

        case 'd':
        case 'D':
            interface.togle_display();
            break;

        case 'o':
        case 'O':
            compute.active_params().print();
            break;

        case '1':
            compute.reset(NBODY_CONFIG_SHELL);
            break;

        case '2':
            compute.reset(NBODY_CONFIG_RANDOM);
            break;

        case '3':
            compute.reset(NBODY_CONFIG_EXPAND);
            break;
    }

    glutPostRedisplay();

    return false;
}

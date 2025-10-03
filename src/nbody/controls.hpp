#pragma once

class Camera;
class Interface;
class Compute;

class Controls {
 public:
    ///
    /// @brief When a user presses and releases mouse buttons in the window, each press and each release generates a mouse callback.
    ///
    /// @param button   One of GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, or GLUT_RIGHT_BUTTON
    /// @param state    Either GLUT_UP or GLUT_DOWN indicating whether the callback was due to a release or press respectively.
    /// @param x        Window relative x coordinate of the mouse.
    /// @param y        Window relative y coordinate of the mouse.
    ///
    auto mouse(int button, int state, int x, int y, Interface& interface, Compute& compute) -> void;

    ///
    ///  @brief     The motion callback for a window is called when the mouse moves within the window while one or more mouse buttons are pressed.
    ///             "passive_motion" would be the relevant function to use if no mouse button is pressed.
    ///
    auto motion(int x, int y, const Interface& interface, Camera& camera, Compute& compute) -> void;

    ///
    /// @brief  When a user types into the window, each key press generating an ASCII character will generate a keyboard callback.
    ///         During a keyboard callback, glutGetModifiers may be called to determine the state of modifier keys when the keystroke generating the callback occurred.
    ///
    /// @param key      The generated ASCII character.
    /// @param x
    /// @param y
    /// @param compute
    /// @param
    /// @param camera
    /// @return
    static auto keyboard(unsigned char key, int x, int y, Compute& compute, Interface& interface, Camera& camera) -> void;

 private:
    auto move_camera(Camera& camera, int x, int y) noexcept -> void;

    auto set_state(int button, int state, int x, int y) noexcept -> void;

    int button_state_ = 0;
    int old_x_        = 0;
    int old_y_        = 0;
};
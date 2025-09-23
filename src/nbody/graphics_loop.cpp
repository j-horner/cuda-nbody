#include "graphics_loop.hpp"

#include "camera.hpp"
#include "compute.hpp"
#include "controls.hpp"
#include "gl_includes.hpp"
#include "interface.hpp"

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define NOMINMAX
#include <GL/wglew.h>
#endif

#include <GL/freeglut.h>

#include <concepts>
#include <tuple>
#include <type_traits>

// get the parameter list of a lambda (with some minor fixes): https://stackoverflow.com/a/70954691
template <typename T> struct Signature;
template <typename C, typename... Args> struct Signature<void (C::*)(Args...) const> {
    using type = typename std::tuple<Args...>;
};
template <typename C> struct Signature<void (C::*)() const> {
    using type = void;
};

template <typename F>
concept is_functor = std::is_class_v<std::decay_t<F>> && requires(F&& t) { &std::decay_t<F>::operator(); };

template <is_functor T> auto arguments(T&& t) -> Signature<decltype(&std::decay_t<T>::operator())>::type;

template <auto GLUTFunction, typename T> struct RegisterCallback;

template <auto GLUTFunction> struct RegisterCallback<GLUTFunction, void> {
    template <is_functor F> static auto callback(void* f_data) -> void {
        const auto* obj = static_cast<F*>(f_data);

        return obj->operator()();
    }

    template <typename F> static auto register_callback(F& func) -> void { GLUTFunction(callback<F>, static_cast<void*>(&func)); }
};

template <auto GLUTFunction, typename... Args> struct RegisterCallback<GLUTFunction, std::tuple<Args...>> {
    template <is_functor F> static auto callback(Args... args, void* f_data) -> void {
        const auto* obj = static_cast<F*>(f_data);

        return obj->operator()(args...);
    }

    template <typename F> static auto register_callback(F& func) -> void { GLUTFunction(callback<F>, static_cast<void*>(&func)); }
};

template <auto GLUTFunction, typename F> auto register_callback(F& func) -> void {
    using Args = std::decay_t<decltype(arguments(func))>;
    RegisterCallback<GLUTFunction, Args>::register_callback(func);
}

auto execute_graphics_loop(ComputeConfig& compute, Interface& interface, Camera& camera, Controls& controls) -> void {
    auto display_ = [&]() { interface.display(compute, camera); };

    auto reshape_ = [](int w, int h) {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(60.0, static_cast<float>(w) / static_cast<float>(h), 0.1, 1000.0);

        glMatrixMode(GL_MODELVIEW);
        glViewport(0, 0, w, h);
    };

    auto mouse_    = [&](int button, int state, int x, int y) { controls.mouse(button, state, x, y, interface, compute); };
    auto motion_   = [&](int x, int y) { controls.motion(x, y, interface, camera, compute); };
    auto keyboard_ = [&](unsigned char k, int x, int y) { Controls::keyboard(k, x, y, compute, interface, camera); };

    // The special keyboard callback is triggered when keyboard function or directional keys are pressed.
    auto special_ = [&](int key, int x, int y) { interface.special(key, x, y); };
    auto idle_    = []() { glutPostRedisplay(); };

    static_assert(std::is_same_v<decltype(arguments(display_)), void>);
    static_assert(std::is_same_v<decltype(arguments(reshape_)), std::tuple<int, int>>);

    register_callback<glutDisplayFuncUcall>(display_);
    register_callback<glutReshapeFuncUcall>(reshape_);
    register_callback<glutMotionFuncUcall>(motion_);
    register_callback<glutMouseFuncUcall>(mouse_);
    register_callback<glutKeyboardFuncUcall>(keyboard_);
    register_callback<glutSpecialFuncUcall>(special_);
    register_callback<glutIdleFuncUcall>(idle_);

    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

    glutMainLoop();
    // TODO: something is triggering an error once the main loop exits
    // if (false == sdkCheckErrorGL(__FILE__, __LINE__)) {
    //     std::exit(EXIT_FAILURE);
    // }
}
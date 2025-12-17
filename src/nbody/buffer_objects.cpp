#include "buffer_objects.hpp"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

#include "gl_includes.hpp"

#include <cassert>

BufferObject::BufferObject(unsigned int buffer) : current_buffer_(current_buffer()) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    // check_OpenGL_error();
}

BufferObject ::~BufferObject() noexcept {
    glBindBuffer(GL_ARRAY_BUFFER, current_buffer_);
    // check_OpenGL_error();
}

auto BufferObject::current_buffer() noexcept -> unsigned int {
    // check_OpenGL_error();

    auto buffer = GLint{-1};

    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &buffer);

    // check_OpenGL_error();

    return static_cast<unsigned int>(buffer);
}

auto BufferObject::bind(unsigned int buffer) noexcept -> void {
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
}

BufferObjects::BufferObjects() noexcept {
    glGenBuffers(1, reinterpret_cast<GLuint*>(&buffer_));
    // check_OpenGL_error();
    assert(buffer_ != 0u);
}

BufferObjects::BufferObjects(BufferObjects&& other) noexcept {
    *this = std::move(other);
}

auto BufferObjects::operator=(BufferObjects&& other) noexcept -> BufferObjects& {
    if (&other != this) {
        buffer_       = other.buffer_;
        other.buffer_ = 0u;
    }
    return *this;
}

BufferObjects::~BufferObjects() noexcept {
    if (buffer_ != 0u) {
        assert(BufferObject::current_buffer() != buffer_);

        glDeleteBuffers(1, reinterpret_cast<GLuint*>(&buffer_));
        // check_OpenGL_error();
    }
}

template <std::floating_point T> auto BufferObjects::bind_static_data(std::span<const T> data) noexcept -> void {
    [[maybe_unused]] const auto buffer = use();

    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(), GL_STATIC_DRAW);
    // check_OpenGL_error();
}
template <std::floating_point T> auto BufferObjects::bind_dynamic_data(std::span<const T> data) noexcept -> void {
    [[maybe_unused]] const auto buffer = use();

    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(), GL_DYNAMIC_DRAW);
    // check_OpenGL_error();
}

template auto BufferObjects::bind_static_data<float>(std::span<const float> data) noexcept -> void;
template auto BufferObjects::bind_static_data<double>(std::span<const double> data) noexcept -> void;

template auto BufferObjects::bind_dynamic_data<float>(std::span<const float> data) noexcept -> void;
template auto BufferObjects::bind_dynamic_data<double>(std::span<const double> data) noexcept -> void;
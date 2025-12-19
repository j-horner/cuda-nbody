#include "buffer_objects.hpp"

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION

#include "gl_includes.hpp"

#include <algorithm>

#include <cassert>

BufferObject::BufferObject(unsigned int buffer_idx) : buffer_(buffer_idx), current_buffer_(current_buffer()) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_);
    assert(glGetError() == GL_NO_ERROR);
}

BufferObject ::~BufferObject() noexcept {
    glBindBuffer(GL_ARRAY_BUFFER, current_buffer_);
    assert(glGetError() == GL_NO_ERROR);
}

auto BufferObject::current_buffer() noexcept -> unsigned int {
    auto buffer = GLint{-1};

    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &buffer);
    assert(glGetError() == GL_NO_ERROR);

    return static_cast<unsigned int>(buffer);
}

auto BufferObject::bind(unsigned int buffer) noexcept -> void {
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    assert(glGetError() == GL_NO_ERROR);
}

template <std::size_t N> BufferObjects<N>::BufferObjects() noexcept {
    glGenBuffers(N, reinterpret_cast<GLuint*>(buffers_.data()));
    assert(glGetError() == GL_NO_ERROR);
    assert(!std::ranges::contains(buffers_, 0u));
}

template <std::size_t N> BufferObjects<N>::BufferObjects(BufferObjects&& other) noexcept {
    *this = std::move(other);
}

template <std::size_t N> auto BufferObjects<N>::operator=(BufferObjects&& other) noexcept -> BufferObjects& {
    if (&other != this) {
        buffers_ = other.buffers_;
        std::ranges::fill(other.buffers_, 0u);
    }
    return *this;
}

template <std::size_t N> BufferObjects<N>::~BufferObjects() noexcept {
    if (!std::ranges::contains(buffers_, 0u)) {
        assert(!std::ranges::contains(buffers_, BufferObject::current_buffer()));

        glDeleteBuffers(N, reinterpret_cast<GLuint*>(buffers_.data()));
        assert(glGetError() == GL_NO_ERROR);
    } else {
        constexpr auto zeros = std::array<unsigned int, N>{{{}}};
        assert(buffers_ == zeros);
    }
}

template <std::size_t N> template <std::floating_point T> auto BufferObjects<N>::allocate_and_bind_static_data(std::size_t k, std::span<const T> data) -> void {
    [[maybe_unused]] const auto buffer = use(k);

    const auto memory_size = data.size() * sizeof(T);

    glBufferData(GL_ARRAY_BUFFER, memory_size, data.data(), GL_STATIC_DRAW);
    assert(glGetError() == GL_NO_ERROR);

    auto size = GLint{0};
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    assert(glGetError() == GL_NO_ERROR);
    assert(static_cast<std::size_t>(size) == memory_size);
}
template <std::size_t N> template <std::floating_point T> auto BufferObjects<N>::allocate_and_bind_dynamic_data(std::size_t k, std::span<const T> data) -> void {
    [[maybe_unused]] const auto buffer = use(k);

    const auto memory_size = data.size() * sizeof(T);

    glBufferData(GL_ARRAY_BUFFER, memory_size, data.data(), GL_DYNAMIC_DRAW);
    assert(glGetError() == GL_NO_ERROR);

    auto size = GLint{0};
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    assert(glGetError() == GL_NO_ERROR);
    assert(static_cast<std::size_t>(size) == memory_size);
}

template <std::size_t N> template <std::floating_point T> auto BufferObjects<N>::bind_data(std::size_t k, std::span<const T> data) -> void {
    [[maybe_unused]] const auto buffer = use(k);

    const auto memory_size = data.size() * sizeof(T);

    glBufferSubData(GL_ARRAY_BUFFER, 0, memory_size, data.data());
    assert(glGetError() == GL_NO_ERROR);

    auto size = GLint{0};
    glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);
    assert(glGetError() == GL_NO_ERROR);
    assert(static_cast<std::size_t>(size) == memory_size);
}

template BufferObjects<1>;
template BufferObjects<2>;

template auto BufferObjects<1>::bind_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<1>::bind_data<double>(std::size_t k, std::span<const double> data) -> void;

template auto BufferObjects<1>::allocate_and_bind_static_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<1>::allocate_and_bind_static_data<double>(std::size_t k, std::span<const double> data) -> void;
template auto BufferObjects<1>::allocate_and_bind_dynamic_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<1>::allocate_and_bind_dynamic_data<double>(std::size_t k, std::span<const double> data) -> void;

template auto BufferObjects<2>::bind_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<2>::bind_data<double>(std::size_t k, std::span<const double> data) -> void;

template auto BufferObjects<2>::allocate_and_bind_static_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<2>::allocate_and_bind_static_data<double>(std::size_t k, std::span<const double> data) -> void;
template auto BufferObjects<2>::allocate_and_bind_dynamic_data<float>(std::size_t k, std::span<const float> data) -> void;
template auto BufferObjects<2>::allocate_and_bind_dynamic_data<double>(std::size_t k, std::span<const double> data) -> void;

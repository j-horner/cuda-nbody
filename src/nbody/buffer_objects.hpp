#pragma once

#include <concepts>
#include <span>

class BufferObjects;

class BufferObject {
 public:
    BufferObject(unsigned int buffer);

    BufferObject(const BufferObject&) = delete;
    BufferObject(BufferObject&&)      = delete;

    auto operator=(const BufferObject&) -> BufferObject&     = delete;
    auto operator=(BufferObject&&) noexcept -> BufferObject& = delete;

    ~BufferObject() noexcept;

 private:
    friend class BufferObjects;

    static auto current_buffer() noexcept -> unsigned int;
    static auto bind(unsigned int buffer) noexcept -> void;

    unsigned int current_buffer_;
};

class BufferObjects {
 public:
    template <std::floating_point T> static auto create_static(std::span<const T> data) noexcept -> BufferObjects {
        auto bo = BufferObjects{};
        bo.bind_static_data<T>(data);
        return bo;
    }
    template <std::floating_point T> static auto create_dynamic(std::span<const T> data) noexcept -> BufferObjects {
        auto bo = BufferObjects{};
        bo.bind_dynamic_data<T>(data);
        return bo;
    }

    BufferObjects() noexcept;

    BufferObjects(const BufferObjects&) = delete;
    BufferObjects(BufferObjects&&) noexcept;

    auto operator=(const BufferObjects&) -> BufferObjects& = delete;
    auto operator=(BufferObjects&&) noexcept -> BufferObjects&;

    template <std::floating_point T> auto bind_static_data(std::span<const T> data) noexcept -> void;
    template <std::floating_point T> auto bind_dynamic_data(std::span<const T> data) noexcept -> void;

    auto use() const -> BufferObject { return {buffer_}; }

    ~BufferObjects() noexcept;

    auto buffer() const noexcept { return buffer_; }

 private:
    unsigned int buffer_;
};

extern template auto BufferObjects::bind_static_data<float>(std::span<const float> data) noexcept -> void;
extern template auto BufferObjects::bind_static_data<double>(std::span<const double> data) noexcept -> void;

extern template auto BufferObjects::bind_dynamic_data<float>(std::span<const float> data) noexcept -> void;
extern template auto BufferObjects::bind_dynamic_data<double>(std::span<const double> data) noexcept -> void;
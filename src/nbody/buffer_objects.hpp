#pragma once

#include <array>
#include <concepts>
#include <span>

#include <cstddef>

template <std::size_t N> class BufferObjects;

class BufferObject {
 public:
    explicit BufferObject(unsigned int buffer_idx);

    BufferObject(const BufferObject&) = delete;
    BufferObject(BufferObject&&)      = delete;

    auto operator=(const BufferObject&) -> BufferObject&     = delete;
    auto operator=(BufferObject&&) noexcept -> BufferObject& = delete;

    auto buffer() const noexcept { return buffer_; }

    ~BufferObject() noexcept;

 private:
    template <std::size_t N> friend class BufferObjects;

    static auto current_buffer() noexcept -> unsigned int;
    static auto bind(unsigned int buffer) noexcept -> void;

    unsigned int buffer_;
    unsigned int current_buffer_;
};

template <std::size_t N> class BufferObjects {
 public:
    template <std::floating_point T> static auto create_static(const std::array<std::span<const T>, N>& data) -> BufferObjects {
        auto bo = BufferObjects{};
        for (auto k = std::size_t{0}; k < N; ++k) {
            bo.allocate_and_bind_static_data<T>(k, data[k]);
        }
        return bo;
    }
    template <std::floating_point T> static auto create_dynamic(const std::array<std::span<const T>, N>& data) -> BufferObjects {
        auto bo = BufferObjects{};
        for (auto k = std::size_t{0}; k < N; ++k) {
            bo.allocate_and_bind_dynamic_data<T>(k, data[k]);
        }
        return bo;
    }

    BufferObjects() noexcept;

    BufferObjects(const BufferObjects&) = delete;
    BufferObjects(BufferObjects&&) noexcept;

    auto operator=(const BufferObjects&) -> BufferObjects& = delete;
    auto operator=(BufferObjects&&) noexcept -> BufferObjects&;

    template <std::floating_point T> auto bind_data(std::size_t k, std::span<const T> data) -> void;

    auto use(std::size_t k) const -> BufferObject { return BufferObject{buffers_[k]}; }

    auto buffer(std::size_t k) const noexcept { return buffers_[k]; }

    ~BufferObjects() noexcept;

 private:
    template <std::floating_point T> auto allocate_and_bind_static_data(std::size_t k, std::span<const T> data) -> void;
    template <std::floating_point T> auto allocate_and_bind_dynamic_data(std::size_t k, std::span<const T> data) -> void;

    std::array<unsigned int, N> buffers_;
};

extern template BufferObjects<1>;
extern template BufferObjects<2>;

extern template auto BufferObjects<1>::bind_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<1>::bind_data<double>(std::size_t k, std::span<const double> data) -> void;

extern template auto BufferObjects<1>::allocate_and_bind_static_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<1>::allocate_and_bind_static_data<double>(std::size_t k, std::span<const double> data) -> void;
extern template auto BufferObjects<1>::allocate_and_bind_dynamic_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<1>::allocate_and_bind_dynamic_data<double>(std::size_t k, std::span<const double> data) -> void;

extern template auto BufferObjects<2>::bind_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<2>::bind_data<double>(std::size_t k, std::span<const double> data) -> void;

extern template auto BufferObjects<2>::allocate_and_bind_static_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<2>::allocate_and_bind_static_data<double>(std::size_t k, std::span<const double> data) -> void;
extern template auto BufferObjects<2>::allocate_and_bind_dynamic_data<float>(std::size_t k, std::span<const float> data) -> void;
extern template auto BufferObjects<2>::allocate_and_bind_dynamic_data<double>(std::size_t k, std::span<const double> data) -> void;
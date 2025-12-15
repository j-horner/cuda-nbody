#pragma once

#include <utility>

#include <cstddef>

template <typename T> class UniqueMappedSpan {
 public:
    UniqueMappedSpan() = default;

    UniqueMappedSpan(std::size_t n, const T& val);

    UniqueMappedSpan(const UniqueMappedSpan&) = delete;
    UniqueMappedSpan(UniqueMappedSpan&& other) noexcept { *this = std::move(other); }

    auto operator=(const UniqueMappedSpan&) = delete;
    auto operator=(UniqueMappedSpan&& other) noexcept -> UniqueMappedSpan& {
        if (this != &other) {
            deallocate();
            host_ptr_   = other.host_ptr_;
            device_ptr_ = other.device_ptr_;

            other.host_ptr_   = nullptr;
            other.device_ptr_ = nullptr;
        }

        return *this;
    }

    auto host_ptr() const noexcept { return host_ptr_; }
    auto device_ptr() const noexcept { return device_ptr_; }

    ~UniqueMappedSpan() noexcept { deallocate(); }

 private:
    auto deallocate() noexcept -> void;

    T* host_ptr_   = nullptr;
    T* device_ptr_ = nullptr;
};

extern template UniqueMappedSpan<float>;
extern template UniqueMappedSpan<double>;
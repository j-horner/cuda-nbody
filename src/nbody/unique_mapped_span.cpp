#include "unique_mapped_span.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include <cassert>

template <typename T> UniqueMappedSpan<T>::UniqueMappedSpan(std::size_t n, const T& val) {
    auto result = cudaHostAlloc(reinterpret_cast<void**>(&host_ptr_), n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable);

    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(result));
    }

    std::fill(host_ptr_, host_ptr_ + n, val);
    result = cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_ptr_), host_ptr_, 0);

    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorName(result));
    }
}

template <typename T> auto UniqueMappedSpan<T>::deallocate() noexcept -> void {
    if (host_ptr_ != nullptr) {
        [[maybe_unused]] const auto result = cudaFreeHost(host_ptr_);
        assert(result == cudaSuccess);
    }
}

template UniqueMappedSpan<float>;
template UniqueMappedSpan<double>;
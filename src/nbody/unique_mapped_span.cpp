#include "unique_mapped_span.hpp"

#include "helper_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>

template <typename T> UniqueMappedSpan<T>::UniqueMappedSpan(std::size_t n, const T& val) {
    checkCudaErrors(cudaHostAlloc((void**)&host_ptr_, n * sizeof(T), cudaHostAllocMapped | cudaHostAllocPortable));

    std::fill(host_ptr_, host_ptr_ + n, val);
    checkCudaErrors(cudaHostGetDevicePointer((void**)&device_ptr_, host_ptr_, 0));
}

template <typename T> auto UniqueMappedSpan<T>::deallocate() noexcept -> void {
    if (host_ptr_ != nullptr) {
        checkCudaErrors(cudaFreeHost(host_ptr_));
    }
}

template UniqueMappedSpan<float>;
template UniqueMappedSpan<double>;
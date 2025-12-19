#include "cuda_opengl_buffers.hpp"

#include "buffer_objects.hpp"
#include "gl_includes.hpp"

#include <cuda_gl_interop.h>

#include <algorithm>
#include <stdexcept>

#include <cassert>

template <std::size_t N> CUDAOpenGLBuffers<N>::CUDAOpenGLBuffers(const BufferObjects<N>& buffers) {
    for (auto i = 0; i < N; ++i) {
        const auto result = cudaGraphicsGLRegisterBuffer(&(resources_[i]), buffers.buffer(i), cudaGraphicsMapFlagsNone);

        if (result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorName(result));
        }

        assert(resources_[i] != nullptr);
    }
}

template <std::size_t N> auto CUDAOpenGLBuffers<N>::operator=(CUDAOpenGLBuffers&& other) noexcept -> CUDAOpenGLBuffers& {
    if (&other != this) {
        unregister();
        resources_ = other.resources_;
        std::ranges::fill(other.resources_, nullptr);
    }
    return *this;
}

template <std::size_t N> auto CUDAOpenGLBuffers<N>::unregister() noexcept -> void {
    if (resources_[0] != nullptr) {
        for (auto i = 0; i < N; ++i) {
            assert(resources_[i] != nullptr);
            [[maybe_unused]] const auto result = cudaGraphicsUnregisterResource(resources_[i]);
            assert(result == cudaSuccess);
        }
    }
}

template <std::size_t N> template <typename T, std::size_t K> CUDAOpenGLBuffers<N>::MappedPointers<T, K>::MappedPointers(const std::array<CUDAGraphicsFlag, K>& flags, std::array<cudaGraphicsResource*, N>& resources) {
    auto result = cudaSuccess;

    auto check_cuda_errors = [&] {
        if (result != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorName(result));
        }
    };

    for (const auto& [idx, flag] : flags) {
        result = cudaGraphicsResourceSetMapFlags(resources[idx], flag);

        check_cuda_errors();
    }

    if constexpr (K == 1) {
        resources_begin_ = resources.data() + flags[0].idx;
    } else {
        resources_begin_ = resources.data();
    }

    result = cudaGraphicsMapResources(K, resources_begin_, 0);

    check_cuda_errors();

    size_t bytes;

    if constexpr (K == 1) {
        result = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(ptrs_.data()), &bytes, *resources_begin_);
        check_cuda_errors();
    } else {
        for (auto k = 0u; k < K; ++k) {
            result = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(ptrs_.data() + k), &bytes, resources[k]);
            check_cuda_errors();
        }
    }
}

template <std::size_t N> template <typename T, std::size_t K> CUDAOpenGLBuffers<N>::MappedPointers<T, K>::~MappedPointers() noexcept {
    [[maybe_unused]] const auto result = cudaGraphicsUnmapResources(K, resources_begin_, 0);
    assert(result == cudaSuccess);
}

template class CUDAOpenGLBuffers<2>;

template class CUDAOpenGLBuffers<2>::MappedPointers<float, 1>;
template class CUDAOpenGLBuffers<2>::MappedPointers<double, 1>;

template class CUDAOpenGLBuffers<2>::MappedPointers<float, 2>;
template class CUDAOpenGLBuffers<2>::MappedPointers<double, 2>;
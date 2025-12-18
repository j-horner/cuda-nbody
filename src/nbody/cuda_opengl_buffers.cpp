#include "cuda_opengl_buffers.hpp"

#include "buffer_objects.hpp"
#include "gl_includes.hpp"
#include "helper_cuda.hpp"

#include <cuda_gl_interop.h>

#include <algorithm>

template <std::size_t N> CUDAOpenGLBuffers<N>::CUDAOpenGLBuffers(const BufferObjects<N>& buffers) {
    for (auto i = 0; i < N; ++i) {
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&(resources_[i]), buffers.buffer(i), cudaGraphicsMapFlagsNone));
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
            checkCudaErrors(cudaGraphicsUnregisterResource(resources_[i]));
        }
    }
}

template <std::size_t N> template <typename T, std::size_t K> CUDAOpenGLBuffers<N>::MappedPointers<T, K>::MappedPointers(const std::array<CUDAGraphicsFlag, K>& flags, std::array<cudaGraphicsResource*, N>& resources) {
    for (const auto& [idx, flag] : flags) {
        checkCudaErrors(cudaGraphicsResourceSetMapFlags(resources[idx], flag));
    }

    if constexpr (K == 1) {
        resources_begin_ = resources.data() + flags[0].idx;
    } else {
        resources_begin_ = resources.data();
    }

    checkCudaErrors(cudaGraphicsMapResources(K, resources_begin_, 0));

    size_t bytes;

    if constexpr (K == 1) {
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(ptrs_.data()), &bytes, *resources_begin_));
    } else {
        for (auto k = 0u; k < K; ++k) {
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(ptrs_.data() + k), &bytes, resources[k]));
        }
    }
}

template <std::size_t N> template <typename T, std::size_t K> CUDAOpenGLBuffers<N>::MappedPointers<T, K>::~MappedPointers() noexcept {
    checkCudaErrors(cudaGraphicsUnmapResources(K, resources_begin_, 0));
}

template class CUDAOpenGLBuffers<2>;

template class CUDAOpenGLBuffers<2>::MappedPointers<float, 1>;
template class CUDAOpenGLBuffers<2>::MappedPointers<double, 1>;

template class CUDAOpenGLBuffers<2>::MappedPointers<float, 2>;
template class CUDAOpenGLBuffers<2>::MappedPointers<double, 2>;
#pragma once

#include "gl_includes.hpp"

#include <cuda_gl_interop.h>

#include <array>
#include <memory>

template <std::size_t N> class BufferObjects;

struct CUDAGraphicsFlag {
    std::size_t          idx;
    cudaGraphicsMapFlags flag;
};

template <std::size_t N> class CUDAOpenGLBuffers {
 public:
    explicit CUDAOpenGLBuffers(const BufferObjects<N>& buffers);

    CUDAOpenGLBuffers(const CUDAOpenGLBuffers&) = delete;
    CUDAOpenGLBuffers(CUDAOpenGLBuffers&& other) noexcept { *this = std::move(other); }

    auto operator=(const CUDAOpenGLBuffers&) -> CUDAOpenGLBuffers& = delete;
    auto operator=(CUDAOpenGLBuffers&& other) noexcept -> CUDAOpenGLBuffers&;

    template <typename T, std::size_t K> class MappedPointers {
        static_assert(K == 1 || K == N, "Either one of the resources are mapped, or all of them are");

     public:
        auto& pointers() const noexcept { return ptrs_; }

        MappedPointers(const MappedPointers&) = delete;
        MappedPointers(MappedPointers&&)      = delete;

        auto operator=(const MappedPointers&) -> MappedPointers& = delete;
        auto operator=(MappedPointers&&) -> MappedPointers&      = delete;

        ~MappedPointers() noexcept;

     private:
        friend class CUDAOpenGLBuffers;

        explicit MappedPointers(const std::array<CUDAGraphicsFlag, K>& flags, std::array<cudaGraphicsResource*, N>& resources);

        std::array<T*, K>      ptrs_;
        cudaGraphicsResource** resources_begin_;
    };

    template <typename T> auto map(const std::array<CUDAGraphicsFlag, N>& flags) -> MappedPointers<T, N> { return MappedPointers<T, N>(flags, resources_); }
    template <typename T> auto map(const CUDAGraphicsFlag& flag) -> MappedPointers<T, 1> { return MappedPointers<T, 1>({flag}, resources_); }

    ~CUDAOpenGLBuffers() noexcept { unregister(); }

 private:
    auto unregister() noexcept -> void;

    std::array<cudaGraphicsResource*, N> resources_{{}};
};

extern template class CUDAOpenGLBuffers<2>;

extern template class CUDAOpenGLBuffers<2>::MappedPointers<float, 1>;
extern template class CUDAOpenGLBuffers<2>::MappedPointers<double, 1>;

extern template class CUDAOpenGLBuffers<2>::MappedPointers<float, 2>;
extern template class CUDAOpenGLBuffers<2>::MappedPointers<double, 2>;
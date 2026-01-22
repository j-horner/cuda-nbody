#pragma once

#include <xsimd/xsimd.hpp>

#include <concepts>
#include <vector>

#include <cstddef>

template <std::floating_point T> struct Coordinates {
    using Vector = std::vector<T, xsimd::aligned_allocator<T>>;

    explicit Coordinates(std::size_t nb_bodies) : x(nb_bodies), y(nb_bodies), z(nb_bodies) {}

    Vector x;
    Vector y;
    Vector z;
};
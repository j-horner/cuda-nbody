#pragma once

#include <concepts>
#include <vector>

#include <cstddef>

template <std::floating_point T> struct Coordinates {
    explicit Coordinates(std::size_t nb_bodies) : x(nb_bodies), y(nb_bodies), z(nb_bodies) {}

    std::vector<T> x;
    std::vector<T> y;
    std::vector<T> z;
};
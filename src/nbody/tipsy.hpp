#pragma once

#include "coordinates.hpp"

#include <array>
#include <filesystem>
#include <vector>

auto read_tipsy_file(const std::filesystem::path& fileName) -> std::array<std::vector<double>, 2>;

struct TipsyData {
    Coordinates<double>         positions;
    Coordinates<double>         velocities;
    Coordinates<double>::Vector masses;
};

auto read_tipsy_file_coordinates(const std::filesystem::path& fileName) -> TipsyData;

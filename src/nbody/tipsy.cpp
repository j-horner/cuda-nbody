#include "tipsy.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <ios>
#include <print>
#include <vector>

#include <cassert>

using Vector3D = std::array<float, 3>;

struct GasParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    rho;
    float    temp;
    float    hsmooth;
    float    metals;
    float    phi;
};

struct DarkParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    eps;
    int      phi;
};

struct StarParticle {
    float    mass;
    Vector3D pos;
    Vector3D vel;
    float    metals;
    float    tform;
    float    eps;
    int      phi;
};

struct Dump {
    double time;
    int    nbodies;
    int    ndim;
    int    nsph;
    int    ndark;
    int    nstar;
};

auto read_tipsy_file(const std::filesystem::path& fileName) -> std::array<std::vector<double>, 2> {
    // Read in our custom version of the tipsy file format written by Jeroen Bedorf.
    // Most important change is that we store particle id on the location where previously the potential was stored.

    std::println("Trying to read file: {}", fileName.string());

    auto inputFile = std::ifstream(fileName, std::ios::in | std::ios::binary);

    if (!inputFile.is_open()) {
        throw std::runtime_error("Can't open input file");
    }

    auto read_data = [&](auto& data) { inputFile.read(reinterpret_cast<char*>(&data), sizeof(data)); };

    auto h = Dump{};
    read_data(h);

    int  idummy;
    auto positions = std::array<double, 4>{};
    auto velocity  = std::array<double, 4>{};

    // Read tipsy header
    auto NTotal = h.nbodies;
    auto NFirst = h.ndark;

    auto d = DarkParticle{};
    auto s = StarParticle{};

    auto bodyPositions  = std::vector<double>{};
    auto bodyVelocities = std::vector<double>{};

    for (int i = 0; i < NTotal; i++) {
        if (i < NFirst) {
            read_data(d);
            velocity[3]  = d.eps;
            positions[3] = d.mass;
            positions[0] = d.pos[0];
            positions[1] = d.pos[1];
            positions[2] = d.pos[2];
            velocity[0]  = d.vel[0];
            velocity[1]  = d.vel[1];
            velocity[2]  = d.vel[2];
            idummy       = d.phi;
        } else {
            read_data(s);
            velocity[3]  = s.eps;
            positions[3] = s.mass;
            positions[0] = s.pos[0];
            positions[1] = s.pos[1];
            positions[2] = s.pos[2];
            velocity[0]  = s.vel[0];
            velocity[1]  = s.vel[1];
            velocity[2]  = s.vel[2];
            idummy       = s.phi;
        }
        bodyPositions.insert(bodyPositions.end(), {positions[0], positions[1], positions[2], positions[3]});
        bodyVelocities.insert(bodyVelocities.end(), {velocity[0], velocity[1], velocity[2], velocity[3]});
    }

    // round up to a multiple of 256 bodies since our kernel only supports that...
    auto newTotal = NTotal;

    if (NTotal % 256) {
        newTotal = ((NTotal / 256) + 1) * 256;
    }

    bodyPositions.insert(bodyPositions.end(), 4u * (newTotal - NTotal), 0.0);
    bodyVelocities.insert(bodyVelocities.end(), 4u * (newTotal - NTotal), 0.0);

    assert(bodyPositions.size() == newTotal * 4);
    assert(bodyVelocities.size() == newTotal * 4);

    std::println("Read {} bodies", newTotal);

    return {std::move(bodyPositions), std::move(bodyVelocities)};
}

auto read_tipsy_file_coordinates(const std::filesystem::path& fileName) -> TipsyData {
    std::println("Trying to read file: {}", fileName.string());

    auto inputFile = std::ifstream(fileName, std::ios::in | std::ios::binary);

    if (!inputFile.is_open()) {
        throw std::runtime_error("Can't open input file");
    }

    auto read_data = [&](auto& data) { inputFile.read(reinterpret_cast<char*>(&data), sizeof(data)); };

    auto h = Dump{};
    read_data(h);

    // Read tipsy header
    auto NTotal = static_cast<std::size_t>(h.nbodies);
    auto NFirst = h.ndark;

    auto d = DarkParticle{};
    auto s = StarParticle{};

    auto masses = Coordinates<double>::Vector(NTotal);

    auto positions  = Coordinates<double>{NTotal};
    auto velocities = Coordinates<double>{NTotal};

    for (int i = 0; i < NTotal; i++) {
        if (i < NFirst) {
            read_data(d);
            masses[i]       = d.mass;
            positions.x[i]  = d.pos[0];
            positions.y[i]  = d.pos[1];
            positions.z[i]  = d.pos[2];
            velocities.x[i] = d.vel[0];
            velocities.y[i] = d.vel[1];
            velocities.z[i] = d.vel[2];
        } else {
            read_data(s);
            masses[i]       = s.mass;
            positions.x[i]  = s.pos[0];
            positions.y[i]  = s.pos[1];
            positions.z[i]  = s.pos[2];
            velocities.x[i] = s.vel[0];
            velocities.y[i] = s.vel[1];
            velocities.z[i] = s.vel[2];
        }
    }

    // round up to a multiple of 256 bodies since our kernel only supports that...
    auto newTotal = NTotal;

    if (NTotal % 256) {
        newTotal = ((NTotal / 256) + 1) * 256;
    }

    auto pad_with_0 = [&](Coordinates<double>::Vector& v) { v.insert(v.end(), 4u * (newTotal - NTotal), 0.0); };

    pad_with_0(positions.x);
    pad_with_0(positions.y);
    pad_with_0(positions.z);
    pad_with_0(velocities.x);
    pad_with_0(velocities.y);
    pad_with_0(velocities.z);
    pad_with_0(masses);

    std::println("Read {} bodies", newTotal);

    return {std::move(positions), std::move(velocities), std::move(masses)};
}
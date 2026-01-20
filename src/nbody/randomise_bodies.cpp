#include "randomise_bodies.hpp"

#include "vec.hpp"

#include <vector_types.h>

#include <algorithm>

#include <cassert>
#include <cmath>

namespace {

template <std::floating_point T> auto normalize(vec3<T>& vector) noexcept -> T {
    const auto dist = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);

    if (dist > 1e-6) {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}
template <std::floating_point T> constexpr auto dot(vec3<T> v0, vec3<T> v1) noexcept -> T {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

template <std::floating_point T> constexpr auto cross(vec3<T> v0, vec3<T> v1) noexcept -> vec3<T> {
    auto rt = vec3<T>{};
    rt.x    = v0.y * v1.z - v0.z * v1.y;
    rt.y    = v0.z * v1.x - v0.x * v1.z;
    rt.z    = v0.x * v1.y - v0.y * v1.x;
    return rt;
}

template <std::floating_point T> auto rng() noexcept {
    return rand() / static_cast<T>(RAND_MAX);    // [0, 1]
};

template <std::floating_point T> auto rng_2() noexcept {
    return rand() * (T{2.0f} / static_cast<T>(RAND_MAX)) - T{1.0f};    // [-1, 1]
};

}    // namespace

template <std::floating_point T> auto randomise_bodies(NBodyConfig config, std::span<T> pos, std::span<T> vel, float clusterScale, float velocityScale) noexcept -> void {
    using enum NBodyConfig;

    assert(pos.size() % 4 == 0);
    assert(vel.size() == pos.size());

    const auto nb_bodies = pos.size() / 4;

    switch (config) {
        default:
        case NBODY_CONFIG_RANDOM:
            {
                const auto scale  = clusterScale * std::max(T{1}, nb_bodies / T{1024});
                const auto vscale = velocityScale * scale;

                auto p = std::size_t{0};
                auto v = std::size_t{0};
                auto i = std::size_t{0};

                while (i < nb_bodies) {
                    auto point = vec3<T>{};
                    point.x    = rng_2<T>();
                    point.y    = rng_2<T>();
                    point.z    = rng_2<T>();
                    auto r2    = dot<T>(point, point);

                    if (r2 > 1)
                        continue;

                    auto velocity = vec3<T>{};
                    velocity.x    = rng_2<T>();
                    velocity.y    = rng_2<T>();
                    velocity.z    = rng_2<T>();
                    r2            = dot<T>(velocity, velocity);

                    if (r2 > 1)
                        continue;

                    pos[p++] = point.x * scale;    // pos.x
                    pos[p++] = point.y * scale;    // pos.y
                    pos[p++] = point.z * scale;    // pos.z
                    pos[p++] = 1.0f;               // mass

                    vel[v++] = velocity.x * vscale;    // pos.x
                    vel[v++] = velocity.y * vscale;    // pos.x
                    vel[v++] = velocity.z * vscale;    // pos.x

                    vel[v++] = 0.0f;    // inverse mass

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_SHELL:
            {
                const auto scale  = clusterScale;
                const auto vscale = scale * velocityScale;
                const auto inner  = T{2.5f} * scale;
                const auto outer  = T{4} * scale;

                auto i = std::size_t{0};
                auto p = std::size_t{0};
                auto v = std::size_t{0};

                while (i < nb_bodies) {
                    auto x = rng_2<T>();
                    auto y = rng_2<T>();
                    auto z = rng_2<T>();

                    auto       point = vec3<T>{x, y, z};
                    const auto len   = normalize<T>(point);

                    if (len > 1)
                        continue;

                    pos[p++] = point.x * (inner + (outer - inner) * rng<T>());
                    pos[p++] = point.y * (inner + (outer - inner) * rng<T>());
                    pos[p++] = point.z * (inner + (outer - inner) * rng<T>());
                    pos[p++] = 1.0f;

                    auto axis = vec3<T>{0, 0, 1};

                    if (1 - point.z < 1e-6) {
                        axis.x = point.y;
                        axis.y = point.x;
                        normalize<T>(axis);
                    }

                    auto vv  = vec3<T>{pos[4 * i], pos[4 * i + 1], pos[4 * i + 2]};
                    vv       = cross<T>(vv, axis);
                    vel[v++] = vv.x * vscale;
                    vel[v++] = vv.y * vscale;
                    vel[v++] = vv.z * vscale;

                    vel[v++] = 0.0f;

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_EXPAND:
            {
                auto scale = clusterScale * nb_bodies / T{1024};

                if (scale < 1) {
                    scale = clusterScale;
                }

                const auto vscale = scale * velocityScale;

                auto p = std::size_t{0};
                auto v = std::size_t{0};

                for (auto i = 0; i < nb_bodies;) {
                    auto point = vec3<T>{};

                    point.x = rng_2<T>();
                    point.y = rng_2<T>();
                    point.z = rng_2<T>();

                    const auto r2 = dot<T>(point, point);

                    if (r2 > 1)
                        continue;

                    pos[p++] = point.x * scale;     // pos.x
                    pos[p++] = point.y * scale;     // pos.y
                    pos[p++] = point.z * scale;     // pos.z
                    pos[p++] = 1.0f;                // mass
                    vel[v++] = point.x * vscale;    // pos.x
                    vel[v++] = point.y * vscale;    // pos.x
                    vel[v++] = point.z * vscale;    // pos.x

                    vel[v++] = 0.0f;

                    ++i;
                }
            }
            break;
    }
}

template <std::floating_point T> auto randomise_bodies(NBodyConfig config, Coordinates<T>& pos, Coordinates<T>& vel, std::span<T> mass, float clusterScale, float velocityScale) noexcept -> void {
    using enum NBodyConfig;

    const auto nb_bodies = pos.x.size();
    assert(vel.x.size() == nb_bodies);
    assert(mass.size() == nb_bodies);

    switch (config) {
        default:
        case NBODY_CONFIG_RANDOM:
            {
                const auto scale  = clusterScale * std::max(T{1}, nb_bodies / T{1024});
                const auto vscale = velocityScale * scale;

                auto i = std::size_t{0};

                while (i < nb_bodies) {
                    auto point = vec3<T>{};
                    point.x    = rng_2<T>();
                    point.y    = rng_2<T>();
                    point.z    = rng_2<T>();
                    auto r2    = dot<T>(point, point);

                    if (r2 > 1)
                        continue;

                    auto velocity = vec3<T>{};
                    velocity.x    = rng_2<T>();
                    velocity.y    = rng_2<T>();
                    velocity.z    = rng_2<T>();
                    r2            = dot<T>(velocity, velocity);

                    if (r2 > 1)
                        continue;

                    pos.x[i] = point.x * scale;    // pos.x
                    pos.y[i] = point.y * scale;    // pos.y
                    pos.z[i] = point.z * scale;    // pos.z
                    mass[i]  = T{1};               // mass

                    vel.x[i] = velocity.x * vscale;    // pos.x
                    vel.y[i] = velocity.y * vscale;    // pos.x
                    vel.z[i] = velocity.z * vscale;    // pos.x

                    i++;
                }
            }
            break;

        case NBODY_CONFIG_SHELL:
            {
                const auto scale  = clusterScale;
                const auto vscale = scale * velocityScale;
                const auto inner  = T{2.5f} * scale;
                const auto outer  = T{4} * scale;

                auto i = std::size_t{0};

                while (i < nb_bodies) {
                    auto x = rng_2<T>();
                    auto y = rng_2<T>();
                    auto z = rng_2<T>();

                    auto       point = vec3<T>{x, y, z};
                    const auto len   = normalize<T>(point);

                    if (len > 1)
                        continue;

                    pos.x[i] = point.x * (inner + (outer - inner) * rng<T>());
                    pos.y[i] = point.y * (inner + (outer - inner) * rng<T>());
                    pos.z[i] = point.z * (inner + (outer - inner) * rng<T>());
                    mass[i]  = T{1};

                    auto axis = vec3<T>{0, 0, 1};

                    if (1 - point.z < 1e-6) {
                        axis.x = point.y;
                        axis.y = point.x;
                        normalize<T>(axis);
                    }

                    auto vv  = vec3<T>{pos.x[i], pos.y[i], pos.z[i]};
                    vv       = cross<T>(vv, axis);
                    vel.x[i] = vv.x * vscale;
                    vel.y[i] = vv.y * vscale;
                    vel.z[i] = vv.z * vscale;

                    ++i;
                }
            }
            break;

        case NBODY_CONFIG_EXPAND:
            {
                auto scale = clusterScale * nb_bodies / T{1024};

                if (scale < 1) {
                    scale = clusterScale;
                }

                const auto vscale = scale * velocityScale;

                for (auto i = std::size_t{0}; i < nb_bodies;) {
                    auto point = vec3<T>{};

                    point.x = rng_2<T>();
                    point.y = rng_2<T>();
                    point.z = rng_2<T>();

                    const auto r2 = dot<T>(point, point);

                    if (r2 > 1)
                        continue;

                    pos.x[i] = point.x * scale;
                    pos.y[i] = point.y * scale;
                    pos.z[i] = point.z * scale;
                    mass[i]  = T{1};
                    vel.x[i] = point.x * vscale;
                    vel.y[i] = point.y * vscale;
                    vel.z[i] = point.z * vscale;

                    ++i;
                }
            }
            break;
    }
}

template auto randomise_bodies<float>(NBodyConfig config, std::span<float> pos, std::span<float> vel, float clusterScale, float velocityScale) noexcept -> void;
template auto randomise_bodies<double>(NBodyConfig config, std::span<double> pos, std::span<double> vel, float clusterScale, float velocityScale) noexcept -> void;
template auto randomise_bodies<float>(NBodyConfig config, Coordinates<float>& pos, Coordinates<float>& vel, std::span<float> mass, float clusterScale, float velocityScale) noexcept -> void;
template auto randomise_bodies<double>(NBodyConfig config, Coordinates<double>& pos, Coordinates<double>& vel, std::span<double> mass, float clusterScale, float velocityScale) noexcept -> void;
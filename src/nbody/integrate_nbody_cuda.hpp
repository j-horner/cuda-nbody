#pragma once

#include <concepts>

template <std::floating_point T> void integrateNbodySystem(T* new_positions, const T* old_positions, T* velocities, T dt, T damping, unsigned int nb_bodies);

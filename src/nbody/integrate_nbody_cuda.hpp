#pragma once

#include <concepts>

template <std::floating_point T> void integrateNbodySystem(T* new_positions, const T* old_positions, T* velocities, unsigned int currentRead, T deltaTime, T damping, unsigned int numBodies, int blockSize);

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "coordinates.hpp"
#include "nbody_config.hpp"

#include <concepts>
#include <span>

// utility function
template <std::floating_point T> auto randomise_bodies(NBodyConfig config, std::span<T> pos, std::span<T> vel, float clusterScale, float velocityScale) noexcept -> void;
template <std::floating_point T> auto randomise_bodies(NBodyConfig config, Coordinates<T>& pos, Coordinates<T>& vel, std::span<T> mass, float clusterScale, float velocityScale) noexcept -> void;

extern template auto randomise_bodies<float>(NBodyConfig config, std::span<float> pos, std::span<float> vel, float clusterScale, float velocityScale) noexcept -> void;
extern template auto randomise_bodies<double>(NBodyConfig config, std::span<double> pos, std::span<double> vel, float clusterScale, float velocityScale) noexcept -> void;
extern template auto randomise_bodies<float>(NBodyConfig config, Coordinates<float>& pos, Coordinates<float>& vel, std::span<float> mass, float clusterScale, float velocityScale) noexcept -> void;
extern template auto randomise_bodies<double>(NBodyConfig config, Coordinates<double>& pos, Coordinates<double>& vel, std::span<double> mass, float clusterScale, float velocityScale) noexcept -> void;
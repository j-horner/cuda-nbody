#pragma once

#include <cuda/api/event.hpp>

#include <concepts>

template <std::floating_point T> struct DeviceData {
    T*           pos[2];    // mapped host pointers
    T*           vel;
    unsigned int offset;
    unsigned int nb_bodies;

    DeviceData(const cuda::device_t& device);

    auto record() const -> void;
    auto synchronise() const -> void;

 private:
    cuda::event_t event_;
};

extern template DeviceData<float>;
extern template DeviceData<double>;
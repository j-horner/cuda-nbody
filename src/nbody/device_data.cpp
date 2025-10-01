#include "device_data.hpp"

#include <cuda/api.hpp>

template <std::floating_point T> DeviceData<T>::DeviceData(const cuda::device_t& device) : event_(cuda::event::create(device)) {}

template <std::floating_point T> auto DeviceData<T>::record() const -> void {
    event_.record();
}
template <std::floating_point T> auto DeviceData<T>::synchronise() const -> void {
    event_.synchronize();
}

template DeviceData<float>;
template DeviceData<double>;
#pragma once

class ComputeConfig;
class Interface;
class Camera;
class Controls;

auto execute_graphics_loop(ComputeConfig& compute, Interface& interface, Camera& camera, Controls& controls) -> void;
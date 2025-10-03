#pragma once

class Compute;
class Interface;
class Camera;
class Controls;

auto execute_graphics_loop(Compute& compute, Interface& interface, Camera& camera, Controls& controls) -> void;
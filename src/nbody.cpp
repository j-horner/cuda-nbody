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

#include "git_commit_id.hpp"
#include "nbody/camera.hpp"
#include "nbody/compute.hpp"
#include "nbody/controls.hpp"
#include "nbody/gl_includes.hpp"
#include "nbody/graphics_loop.hpp"
#include "nbody/interface.hpp"
#include "nbody/render_particles.hpp"

#include <CLI/CLI.hpp>
#include <GL/freeglut.h>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define NOMINMAX
#include <GL/wglew.h>    // for wglSwapIntervalEXT()
#endif

#include <filesystem>
#include <limits>
#include <print>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <cstddef>

auto split_string(std::string_view to_split, char delimiter) -> std::vector<std::string_view> {
    if (to_split.empty()) {
        return {};
    }

    auto delim_pos = to_split.find_first_of(delimiter);

    auto the_split = std::vector<std::string_view>{};

    if (delim_pos != 0u) {
        the_split.push_back(to_split.substr(0, delim_pos));
    }

    while (delim_pos != std::string_view::npos) {
        const auto substr_start = delim_pos + 1u;

        if (substr_start >= to_split.size()) {
            break;
        }

        assert(substr_start != std::string_view::npos);

        delim_pos = to_split.find_first_of(delimiter, substr_start);

        if (delim_pos != substr_start) {
            the_split.push_back(to_split.substr(substr_start, delim_pos - substr_start));
        }
    }

    return the_split;
}

auto areGLExtensionsSupported(std::string_view extensions) -> bool {
    const auto all_extensions_str = std::string{reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS))};

    const auto all_extensions = split_string(all_extensions_str, ' ');

    const auto requested_extensions = split_string(extensions, ' ');

    return std::ranges::includes(all_extensions, requested_extensions);
}

auto isGLVersionSupported(unsigned reqMajor, unsigned reqMinor) -> bool {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    if (glewInit() != GLEW_OK) {
        std::println(stderr, "glewInit() failed!");
        return 0;
    }
#endif
    auto stream = std::stringstream(std::string{reinterpret_cast<const char*>(glGetString(GL_VERSION))});

    auto major = 0u;
    auto minor = 0u;
    auto dot   = '.';

    stream >> major >> dot >> minor;

    assert(dot == '.');
    return major > reqMajor || (major == reqMajor && minor >= reqMinor);
}

auto initGL(int* argc, char** argv, bool full_screen) -> void {
    // First initialize OpenGL context, so we can properly set the GL for CUDA.
    // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(1920, 1080);
    glutCreateWindow("CUDA n-body system");

    if (full_screen) {
        glutFullScreen();
    }

    if (!isGLVersionSupported(2, 0) || !areGLExtensionsSupported("GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
        throw std::runtime_error("Required OpenGL extensions missing.");
    } else {
#if defined(WIN32)
        wglSwapIntervalEXT(0);
#elif defined(LINUX)
        glxSwapIntervalSGI(0);
#endif
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0, 0.0, 0.0, 1.0);

    GLenum error;

    while ((error = glGetError()) != GL_NO_ERROR) {
        std::println(stderr, "initGL: error - {}", reinterpret_cast<const char*>(gluErrorString(error)));
    }
}

///
/// @brief  Describes the various outcomes after parsing the command-line arguments.
///
enum class Status {
    OK = 0,             // Proceed with the rest of the program
    CleanShutDown,      // Everything was fine but exit the program normally
    InvalidArguments    // Something went wrong parsing the command-line arguments!
};

struct Options {
    bool                  fullscreen = false;
    bool                  fp64       = false;
    bool                  hostmem    = false;
    bool                  benchmark  = false;
    std::size_t           numbodies  = 0;
    int                   device     = -1;
    int                   numdevices = 0;
    bool                  compare    = false;
    bool                  qatest     = false;
    bool                  cpu        = false;
    std::filesystem::path tipsy;
    std::size_t           iterations = 0;
    int                   block_size = 0;
};

auto parse_args(int argc, char** argv) -> std::pair<Status, Options> {
    auto options = Options{};

    auto app = CLI::App{"The CUDA NBody sample demo.", "cuda-nbody"};

    auto display_version = false;

    app.add_flag("--fullscreen", options.fullscreen, "Run n-body simulation in fullscreen mode");
    app.add_flag("--fp64", options.fp64, "Use double precision floating point values for simulation");
    app.add_flag("--hostmem", options.hostmem, "Stores simulation data in host memory");
    app.add_flag("--benchmark", options.benchmark, "Run benchmark to measure performance");
    app.add_option("--numbodies", options.numbodies, "Number of bodies (>= 1) to run in simulation")->check(CLI::Range(std::size_t{1u}, std::numeric_limits<std::size_t>::max()));
    const auto device_opt = app.add_option("--device", options.device, "The CUDA device to use")->check(CLI::Range(0, std::numeric_limits<int>::max()));
    app.add_option("--numdevices", options.numdevices, "Number of CUDA devices (> 0) to use for simulation")->check(CLI::Range(std::size_t{1u}, std::numeric_limits<std::size_t>::max()))->excludes(device_opt);
    app.add_flag("--compare", options.compare, "Compares simulation results running once on the default GPU and once on the CPU");
    app.add_flag("--qatest", options.qatest, "Runs a QA test");
    app.add_flag("--cpu", options.cpu, "Run n-body simulation on the CPU");
    app.add_option("--tipsy", options.tipsy, "Load a tipsy model file for simulation")->check(CLI::ExistingFile);
    app.add_option("-i,--iterations", options.iterations, "Number of iterations to run in the benchmark")->default_val(10);
    app.add_option("--blockSize", options.block_size, "The CUDA kernel block size")->default_val(256);

    // cppcheck-suppress unmatchedSuppression
    // cppcheck-suppress passedByValue
    auto error = [&](std::string_view message) {
        std::println(stderr,
                     "-------------------------------------------\n"
                     "CRITICAL ERROR:\n"
                     "{}\n"
                     "-------------------------------------------\n",
                     message);

        std::println(stderr, "{}", app.help());

        return std::pair(Status::InvalidArguments, std::move(options));
    };

    try {
        app.parse(argc, argv);

        if (display_version) {
            std::println("cuda-nbody: {}", git_commit_id);
            return std::pair(Status::CleanShutDown, std::move(options));
        }
    } catch (const CLI::CallForHelp&) {
        std::println("{}", app.help());

        return std::pair(Status::CleanShutDown, std::move(options));
    } catch (const CLI::ParseError& e) { return error(e.what()); }

    std::println(R"(Run " nbody - benchmark[-numbodies = <numBodies>] " to measure performance)");
    std::println("{}", app.help());

    return std::pair(Status::OK, std::move(options));
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
auto main(int argc, char** argv) -> int {
    try {
        // parse the command-line arguments
        const auto program_state = parse_args(argc, argv);

        const auto program_status = program_state.first;

        // check the arguments were valid and if we should continue
        if (Status::InvalidArguments == program_status) {
            // treat invalid arguments as an error and exit the program
            return 1;
        }
        if (Status::CleanShutDown == program_status) {
            // shut down the program normally if required (e.g. if --help was requested)
            return 0;
        }

        const auto cmd_options = program_state.second;

#if defined(__linux__)
        setenv("DISPLAY", ":0", 0);
#endif

        std::println("NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

        const auto full_screen = cmd_options.fullscreen;

        std::println("> {} mode", full_screen ? "Fullscreen" : "Windowed");

        auto show_sliders = !full_screen;

        auto tipsy_file = cmd_options.tipsy;
        auto cycle_demo = tipsy_file.empty();
        show_sliders    = tipsy_file.empty();

        // Initialize GL and GLUT if necessary
        // TODO: graphics stuf is currently setup inside Compute so this needs to be invoked first
        if ((!cmd_options.compare) && (!cmd_options.benchmark) && (!cmd_options.qatest)) {
            initGL(&argc, argv, full_screen);
        }

        const auto compare_to_cpu = (cmd_options.compare || cmd_options.qatest) && (!cmd_options.cpu);

        auto compute = Compute(
            cmd_options.fp64,
            cycle_demo,
            cmd_options.cpu,
            compare_to_cpu,
            cmd_options.benchmark,
            cmd_options.hostmem,
            cmd_options.device,
            cmd_options.numdevices,
            cmd_options.block_size,
            cmd_options.numbodies,
            tipsy_file);

        if (cmd_options.benchmark) {
            const auto nb_iterations = cmd_options.iterations == 0 ? 10 : static_cast<int>(cmd_options.iterations);
            compute.run_benchmark(nb_iterations);
            return 0;
        }

        if (compare_to_cpu) {
            const auto result = compute.compare_results();

            return static_cast<int>(!result);
        }

        auto sliders = ParamListGL{};

        compute.add_modifiable_parameters(sliders);

        auto interface = Interface{show_sliders, std::move(sliders), full_screen, ParticleRenderer(compute.nb_bodies())};

        auto camera = Camera{};

        auto controls = Controls{};

        execute_graphics_loop(compute, interface, camera, controls);

        std::println("Stopped graphics loop");

        return 0;
    } catch (const std::invalid_argument& e) {
        std::println(stderr, "ERROR: {}", e.what());
        return 1;
    } catch (const std::bad_alloc&) {
        std::println(stderr, "ERROR: Unable to allocate memory!");
        return 3;
    } catch (const std::exception& e) {
        std::println(stderr, "ERROR: ", e.what());
        return 2;
    } catch (...) {
        std::println("ERROR: An unknown error occurred! Please inform your local developer!");
        return 4;
    }
}

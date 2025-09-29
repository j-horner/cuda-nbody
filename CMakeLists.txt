# ===================================================================
#
#   Preamble
#
# ===================================================================
cmake_minimum_required(VERSION 3.31.6)

#
#   Don't allow in-source builds
#
include(cmake/prelude.cmake)

#
# download dependencies required for optional features
#
# BUILD_TESTING is a built-in option enabled by CTest by default.
# This disables testing by default but still allows it to be overridden.
#
option(BUILD_TESTING "Enable building of unit tests" OFF)
message("BUILD_TESTING: " ${BUILD_TESTING})
if(BUILD_TESTING)
    # this will trigger vcpkg to download any additional dependencies required for the "tests" feature, in this case it is just catch2
    list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

message("VCPKG_MANIFEST_FEATURES: ${VCPKG_MANIFEST_FEATURES}")

#
#   Explicitly keep VERSION as 0.1.0 as that will be a more sensible default for new projects.
#
project(cuda-nbody
        VERSION 0.1.0
        DESCRIPTION "A CUDA-NBody demo."
        HOMEPAGE_URL "https://github.com/j-horner-c4x/cuda-nbody"
        LANGUAGES CXX)

file(WRITE ./src/version_string.hpp "constexpr static auto version_string = \"${PROJECT_VERSION}\";\n")

# Optimisation for speeding up documentation generation in CI. See cmake/docs.cmake.
docs_early_return()


# ===================================================================
#
#   Project wide setup
#
# ===================================================================
#
#   If PROJECT_IS_TOP_LEVEL is false, the project is being built as a dependency (converse is not necessarily true).
#
message("PROJECT_IS_TOP_LEVEL: ${PROJECT_IS_TOP_LEVEL}")

# Sets developer mode and SYSTEM warning guard (disables warnings if being built as a dependency)
include(cmake/variables.cmake)
message("${PROJECT_NAME}_DEVELOPER_MODE: ${${PROJECT_NAME}_DEVELOPER_MODE}")
message("warning_guard: ${warning_guard}")

if(CMAKE_CXX_CLANG_TIDY)
    message("CMAKE_CXX_CLANG_TIDY:" ${CMAKE_CXX_CLANG_TIDY})
endif()
if(CMAKE_CXX_CPPCHECK)
    message("CMAKE_CXX_CPPCHECK:" ${CMAKE_CXX_CPPCHECK})
endif()

#
#   enable CUDA (if available)
#
include(cmake/enable-cuda-if-available.cmake)

#
#   generate `src/git_commit_id.hpp` so GIT_COMMIT_ID can be used in code
#
include(cmake/include-git-commit-id.cmake)

message("EXTRA_FLAGS: ${EXTRA_FLAGS}")
message("CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message("CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message("CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})

# enable any extra flags set in the presets
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${EXTRA_FLAGS}")

if(${PROJECT_NAME}_DEVELOPER_MODE)
    if(NOT PROJECT_IS_TOP_LEVEL)
        message(AUTHOR_WARNING "Developer mode is intended for developers of ${PROJECT_NAME}")
    endif()

    # linting, coverage, documentation generation
    include(cmake/dev-mode.cmake)
endif()

# enables unit testing if BUILD_TESTING is on
include(cmake/add-unit-test.cmake)

# rpath and symbol visibility must be set before targets are declared
include(cmake/set-library-symbol-visibility-and-rpath.cmake)

#
# Threads: Finds the relevant thread library for the system. Required for certain standard C++ features e.g. std::thread
#
find_package(Threads REQUIRED)

# ===================================================================
#
#   External Dependencies
#
# ===================================================================

#
# Add 3rd Party Libraries
#
# e.g. find_package(nlohmann_json CONFIG REQUIRED)

find_package(CLI11 CONFIG REQUIRED)

find_package(OpenGL REQUIRED)

find_package(GLUT REQUIRED)

find_package(GLEW REQUIRED)

find_package(cuda-api-wrappers CONFIG REQUIRED)

# ===================================================================
#
#   Main code to build
#
# ===================================================================

add_subdirectory(src)


# ===================================================================
#
#   Install rules
#
# ===================================================================

if(NOT CMAKE_SKIP_INSTALL_RULES)
    # enables find_package() for consumers of this project
    include(cmake/install-rules.cmake)
endif()

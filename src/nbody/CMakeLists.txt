#
#   Static library - main implementation.
#   Consider keeping all implementation libraries static for simplicity.
#
add_library(${PROJECT_NAME}_render_particles STATIC render_particles.cpp)
add_library(${PROJECT_NAME}_tipsy STATIC tipsy.cpp)
add_library(${PROJECT_NAME}_win_coords STATIC win_coords.cpp)
add_library(${PROJECT_NAME}_paramgl STATIC paramgl.cpp)
add_library(${PROJECT_NAME}_param STATIC param.cpp)
add_library(${PROJECT_NAME}_params STATIC params.cpp)

add_library(${PROJECT_NAME}_randomise_bodies STATIC randomise_bodies.cpp)
add_library(${PROJECT_NAME}_bodysystemcuda STATIC bodysystemcuda.cu bodysystemcuda.cpp bodysystemcuda_default.cu bodysystemcuda_host_memory.cpp unique_mapped_span.cpp bodysystemcuda_graphics.cpp)
add_library(${PROJECT_NAME}_bodysystemcpu STATIC bodysystemcpu.cpp)

add_library(${PROJECT_NAME}_compute STATIC compute.cpp)
add_library(${PROJECT_NAME}_compute_cpu STATIC compute_cpu.cpp)
add_library(${PROJECT_NAME}_compute_cuda STATIC compute_cuda.cpp)
add_library(${PROJECT_NAME}_camera STATIC camera.cpp)
add_library(${PROJECT_NAME}_interface STATIC interface.cpp)
add_library(${PROJECT_NAME}_controls STATIC controls.cpp)

add_library(${PROJECT_NAME}_graphics_loop STATIC graphics_loop.cpp)

target_link_libraries(${PROJECT_NAME}_camera PRIVATE GLEW::GLEW OpenGL::GL)
target_link_libraries(${PROJECT_NAME}_win_coords PRIVATE GLUT::GLUT)
target_link_libraries(${PROJECT_NAME}_render_particles PRIVATE OpenGL::GL GLEW::GLEW)
target_link_libraries(${PROJECT_NAME}_randomise_bodies PRIVATE CUDA::cudart_static)

target_link_libraries(${PROJECT_NAME}_paramgl
                        PUBLIC
                            ${PROJECT_NAME}_param
                        PRIVATE
                            GLUT::GLUT
                            OpenGL::GL GLEW::GLEW
                            ${PROJECT_NAME}_win_coords)

target_link_libraries(${PROJECT_NAME}_bodysystemcpu
                        PRIVATE
                            CUDA::cudart_static
                            ${PROJECT_NAME}_randomise_bodies)

target_link_libraries(${PROJECT_NAME}_bodysystemcuda
                        PUBLIC
                            CUDA::cudart_static
                        PRIVATE
                            ${PROJECT_NAME}_compute_cuda    
                            ${PROJECT_NAME}_params
                            ${PROJECT_NAME}_randomise_bodies
                            GLUT::GLUT)

target_link_libraries(${PROJECT_NAME}_compute
                        PUBLIC
                            ${PROJECT_NAME}_params
                        PRIVATE
                            ${PROJECT_NAME}_camera
                            ${PROJECT_NAME}_tipsy
                            ${PROJECT_NAME}_paramgl
                            ${PROJECT_NAME}_param
                            ${PROJECT_NAME}_compute_cpu
                            ${PROJECT_NAME}_compute_cuda)

target_link_libraries(${PROJECT_NAME}_compute_cuda
                        PUBLIC
                            cuda-api-wrappers::rtc cuda-api-wrappers::nvtx cuda-api-wrappers::runtime-and-driver cuda-api-wrappers::fatbin
                        PRIVATE
                            ${PROJECT_NAME}_interface
                            ${PROJECT_NAME}_bodysystemcuda
                            ${PROJECT_NAME}_bodysystemcpu)

target_link_libraries(${PROJECT_NAME}_compute_cpu
                        PRIVATE
                            ${PROJECT_NAME}_interface
                            ${PROJECT_NAME}_bodysystemcpu)

target_link_libraries(${PROJECT_NAME}_controls
                        PRIVATE
                            GLUT::GLUT
                            ${PROJECT_NAME}_camera
                            ${PROJECT_NAME}_compute
                            ${PROJECT_NAME}_interface)

target_link_libraries(${PROJECT_NAME}_interface 
                        PUBLIC
                            ${PROJECT_NAME}_paramgl
                            ${PROJECT_NAME}_render_particles
                        PRIVATE
                            cuda-api-wrappers::rtc cuda-api-wrappers::nvtx cuda-api-wrappers::runtime-and-driver cuda-api-wrappers::fatbin
                            GLEW::GLEW
                            OpenGL::GL
                            GLUT::GLUT
                            ${PROJECT_NAME}_compute
                            ${PROJECT_NAME}_camera
                            ${PROJECT_NAME}_param
                            ${PROJECT_NAME}_win_coords)

target_link_libraries(${PROJECT_NAME}_graphics_loop 
                        PRIVATE
                            GLEW::GLEW
                            OpenGL::GL
                            GLUT::GLUT
                            ${PROJECT_NAME}_controls
                            ${PROJECT_NAME}_interface)


#
# set_cuda_compile_options(<target>)
#
# Sets various compile and link options for the CUDA libraries
#
function(set_cuda_compile_options target)
    target_compile_features(${target} PUBLIC cxx_std_20)

    # p3.2xlarge
    # target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_70,code=sm_70 -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Werror cross-execution-space-call,deprecated-declarations,reorder -expt-extended-lambda -expt-relaxed-constexpr>)

    # We need to explicitly state that we need all CUDA files in the
    # gpu library to be built with -dc as the member functions
    # could be called by other libraries and executables
    set_target_properties(${target} PROPERTIES  CUDA_SEPARABLE_COMPILATION ON
                                                POSITION_INDEPENDENT_CODE ON
                                                CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                CUDA_STANDARD 20)

    # link to cublas
    # target_link_libraries(${target} PRIVATE CUDA::cublas CUDA::curand)
endfunction()

set_cuda_compile_options(${PROJECT_NAME}_bodysystemcuda)



#
#   Each unit test should correspond to one .cpp file and should be named with .test.
#   For more details see: https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.tests.merged
#   Unit test functions add ${PROJECT_NAME}_ prefix automatically.
#
# cuda_nbody_add_unit_test(library_impl_test library_impl.test.cpp)
# cuda_nbody_link_unit_test(library_impl_test PRIVATE ${PROJECT_NAME}_library_impl)

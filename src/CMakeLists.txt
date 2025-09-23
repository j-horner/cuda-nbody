#
#   the directory structure of include/ should match that of src/ all objects should have a matching namespace, starting with jh::
#   e.g. the library_class object defined in src/library/ has the full name jh::library::library_class
#   for more details see: https://api.csswg.org/bikeshed/?force=1&url=https://raw.githubusercontent.com/vector-of-bool/pitchfork/develop/data/spec.bs#src.header-placement.separate
#
add_subdirectory(nbody)

# ---- Declare executable ----

#
#   Target names are global so if this project is included as a subdirectory of another, name clashes can occur.
#   Therefore consider keeping target names project specific.
#
add_executable(${PROJECT_NAME}_exe nbody.cpp)

#
#   find_package() will import an appropriately namespaced target
#
add_executable(${PROJECT_NAME}::exe ALIAS ${PROJECT_NAME}_exe)
set_target_properties(${PROJECT_NAME}_exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME} EXPORT_NAME exe)

target_link_libraries(${PROJECT_NAME}_exe
                        PRIVATE
                            CLI11::CLI11
                            OpenGL::GL
                            GLUT::GLUT
                            GLEW::GLEW
                            ${PROJECT_NAME}_compute
                            ${PROJECT_NAME}_camera
                            ${PROJECT_NAME}_interface
                            ${PROJECT_NAME}_controls
                            ${PROJECT_NAME}_graphics_loop
                            ${PROJECT_NAME}_render_particles)


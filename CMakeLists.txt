cmake_minimum_required(VERSION 3.25)
project(GPURayTracer CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(IMGUI REQUIRED imgui)

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${IMGUI_INCLUDE_DIRS})

add_subdirectory(core)
add_subdirectory(utils)
add_subdirectory(common)
add_subdirectory(world)
add_subdirectory(exr)

add_executable(GPURayTracer main.cu)

target_link_libraries(GPURayTracer
        INTERFACE
            common
            utils
        PRIVATE
            core
            world
            ${OPENGL_LIBRARIES}
            glfw
            GLEW::GLEW
            hdr
            ${IMGUI_LIBRARIES}
)

target_include_directories(GPURayTracer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_compile_options(-G)

set_target_properties(GPURayTracer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
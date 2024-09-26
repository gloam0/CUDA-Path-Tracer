add_library(core
        camera.cu
        camera.cuh
        init.cu
        init.cuh
        input.cu
        input.hpp
        render.cu
        render.cuh
)

target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(core
        INTERFACE
            common
            utils
            material
            geometry
        PUBLIC
            world
            curand
)

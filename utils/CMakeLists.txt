cmake_minimum_required(VERSION 3.24)

add_library(utils INTERFACE)

target_include_directories(utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(JPEG REQUIRED)

target_include_directories(utils INTERFACE ${JPEG_INCLUDE_DIRS})

target_link_libraries(utils
        INTERFACE
            common
            ${JPEG_LIBRARIES})

add_library(geometry INTERFACE)

target_include_directories(geometry INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(geometry INTERFACE
        common
        utils)
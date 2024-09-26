add_library(material INTERFACE)

target_include_directories(material INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(material INTERFACE
        common
        utils)
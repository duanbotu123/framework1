# cpp/io/CMakeLists.txt



set(IO_HEADERS
    io_pc.h
    io_mesh.h
 )


add_library(io STATIC ${IO_SOURCES} ${IO_HEADERS})

target_include_directories(io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# io 可能依赖 core
target_link_libraries(io PUBLIC core)


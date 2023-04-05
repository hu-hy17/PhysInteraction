file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
add_executable(entry ${HDR} ${SRC})

# Link third libraries
target_link_libraries(entry ${LIBRARIES})

# Link project libraries
if(WITH_CUDA)
    target_link_libraries(entry cudax)
endif()

set(RECONSTRUCTION_LIB ${CMAKE_SOURCE_DIR}/reconstruction/lib/reconstruction.lib)

target_link_libraries(entry physhand spheremesh2mano hmodelsdk ${RECONSTRUCTION_LIB})

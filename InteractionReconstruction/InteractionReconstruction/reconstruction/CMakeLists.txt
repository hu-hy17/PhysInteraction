file(GLOB_RECURSE reconstruction_headers "${CMAKE_CURRENT_LIST_DIR}/include/*.h")
file(GLOB_RECURSE reconstruction_hpp "${CMAKE_CURRENT_LIST_DIR}/include/*.hpp")
add_custom_target(reconstruction SOURCES ${reconstruction_headers} ${reconstruction_hpp})

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
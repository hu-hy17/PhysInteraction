#--- STATICALLY LINKED!!!
find_package(GLEW REQUIRED)
if(GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_directories(${GLEW_LIBRARY_DIRS})
    add_definitions(-DGLEW_STATIC)
    add_definitions(-DWITH_GLEW)
    LIST(APPEND LIBRARIES ${GLEW_LIBRARIES})
    message(STATUS "GLEW_LIBRARIES: ${GLEW_LIBRARIES}")
else()
    message(ERROR "GLEW NOT FOUND!")
endif()
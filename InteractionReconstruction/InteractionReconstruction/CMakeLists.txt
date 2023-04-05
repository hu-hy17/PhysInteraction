cmake_minimum_required(VERSION 3.9)
project(PhysicalInteraction VERSION 1.0)

#--- Gets rid of annoying CMake 3 warnings
if(NOT (CMAKE_MAJOR_VERSION LESS 3))
    cmake_policy(SET CMP0043 OLD)
    cmake_policy(SET CMP0020 OLD)
    cmake_policy(SET CMP0005 OLD)
endif()

#--- Where to find the FindPackageName.cmake
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#--- Where exes/libs will be deployed
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR})

#--- Compiler setup
include(cmake/ConfigureCompiler.cmake)

#--- Configure the libraries
include(cmake/ConfigureBoost.cmake)
include(cmake/ConfigureCeres.cmake)
include(cmake/ConfigureCUDA.cmake)
include(cmake/ConfigureCurl.cmake)
include(cmake/ConfigureEigen.cmake)
include(cmake/ConfigureFlann.cmake)
include(cmake/ConfigureGLEW.cmake)
include(cmake/ConfigureJsoncpp.cmake)
include(cmake/ConfigureLibigl.cmake)
include(cmake/ConfigureOpenCV.cmake)
include(cmake/ConfigureOpenGL.cmake)
# include(cmake/ConfigureOpenNI.cmake)
include(cmake/ConfigureOpenGP.cmake)
include(cmake/ConfigurePCL.cmake)
include(cmake/ConfigureQt.cmake)
include(cmake/ConfigureRealSense.cmake)

#--- Add source directory as include directory
include_directories(${CMAKE_SOURCE_DIR})

#--- sub-projects
add_subdirectory(cudax)
add_subdirectory(spheremesh2mano)
add_subdirectory(reconstruction)
add_subdirectory(physhand)
add_subdirectory(tracker)
add_subdirectory(entry/main)
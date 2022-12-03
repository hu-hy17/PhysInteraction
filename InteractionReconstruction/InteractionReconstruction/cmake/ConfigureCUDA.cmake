set(WITH_CUDA TRUE)
add_definitions(-DWITH_CUDA)

#--- Import CUDA/CUBLAS
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
if(WIN32)
    list(APPEND LIBRARIES ${CUDA_SDK_ROOT_DIR}/lib/x64/cublas.lib
                        ${CUDA_SDK_ROOT_DIR}/lib/x64/cublas_device.lib
                        ${CUDA_SDK_ROOT_DIR}/lib/x64/cudart_static.lib
                        ${CUDA_SDK_ROOT_DIR}/lib/x64/cuda.lib
                        ${CUDA_SDK_ROOT_DIR}/lib/x64/cudart.lib
                        ${CUDA_SDK_ROOT_DIR}/lib/x64/cufft.lib)
endif()

message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")

#--- Find nvToolsExt
find_package(NvToolsExt REQUIRED)
if(NvToolsExt_FOUND)
    message(STATUS "NvToolsExt_INCLUDE_DIR: ${NvToolsExt_INCLUDE_DIR}")
    include_directories(${NvToolsExt_INCLUDE_DIR})
else()
    message(ERROR "NvToolsExt NOT FOUND!")
endif()

#--- For matrix operations within Kernels (Eigen not supported)
find_package(GLM REQUIRED)
if(GLM_FOUND)
    message(STATUS "GLM_INCLUDE_DIRS: ${GLM_INCLUDE_DIRS}")
    include_directories(${GLM_INCLUDE_DIRS})
    add_definitions(-DGLM_FORCE_CUDA)
else()
    message(ERROR "GLM NOT FOUND!")
endif()

#--- Card needs appropriate version
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61") # TiTan Xp

#--- Enable debugging flags
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g") # HOST debug mode
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G") # DEV debug mode
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DNDEBUG") #< disable asserts

if(WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS True)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math")
    endif()
else()
    #--- CUDA doesn't like "--compiler-options -std=c++11"
    set(CUDA_PROPAGATE_HOST_FLAGS False)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -use_fast_math")
endif()

message(STATUS "CUDA_PROPAGATE_HOST_FLAGS: ${CUDA_PROPAGATE_HOST_FLAGS}")
message(STATUS "CUDA_HOST_COMPILER: ${CUDA_HOST_COMPILER}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

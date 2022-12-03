find_package(RealSense REQUIRED)
if(RealSense_FOUND)
    message(STATUS "RealSense_ROOT_DIR: ${RealSense_ROOT_DIR}")
    # include_directories(${REALSENSE_INCLUDE_DIR} ${REALSENSE_UTILITY_DIR} ${REALSENSE2_INCLUDE_DIR})
    include_directories(${REALSENSE2_INCLUDE_DIR})

    # list(APPEND LIBRARIES ${REALSENSE_LIBRARY})
    # list(APPEND LIBRARIES ${REALSENSE_UTILITY_LIBRARY})
    list(APPEND LIBRARIES ${REALSENSE2_LIBRARY})

    add_definitions(-DHAS_REALSENSE)
else()
    message(ERROR "RealSense NOT FOUND!")
endif()

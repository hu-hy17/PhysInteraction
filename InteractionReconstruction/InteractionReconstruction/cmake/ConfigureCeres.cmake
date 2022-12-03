find_package(Ceres REQUIRED)

if(Ceres_FOUND)
    message(STATUS "Ceres_ROOT_DIR: ${Ceres_ROOT_DIR}")
    include_directories(${Ceres_INCLUDE_DIR})
    include_directories(${Glog_INCLUDE_DIR})
    LIST(APPEND LIBRARIES ${Ceres_LIBRARIES} ${Glog_LIBRARIES})

    add_definitions(-DGOOGLE_GLOG_DLL_DECL=)
    add_definitions(-DCERES_MSVC_USE_UNDERSCORE_PREFIXED_BESSEL_FUNCTIONS)
    add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)
else()
    message(ERROR "Ceres NOT FOUND!")
endif()
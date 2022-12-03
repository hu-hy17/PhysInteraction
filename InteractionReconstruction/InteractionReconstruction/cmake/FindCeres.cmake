FIND_PATH(Ceres_ROOT_DIR ceres-solver/include/ceres/ceres.h
	#--- FOR WINDOWS 
	C:/Developer
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/ceres)

if(WIN32 AND Ceres_ROOT_DIR)
    FIND_PATH(Ceres_INCLUDE_DIR ceres/ceres.h ${Ceres_ROOT_DIR}/ceres-solver/include)
    FIND_PATH(Glog_INCLUDE_DIR config.h ${Ceres_ROOT_DIR}/glog/src/windows)
        
    FIND_LIBRARY(Ceres_LIBRARIES NAMES ceres PATHS 
        ${Ceres_ROOT_DIR}/x64/Release)
    FIND_LIBRARY(Glog_LIBRARIES NAMES libglog_static PATHS 
        ${Ceres_ROOT_DIR}/x64/Release)
endif()

set(Ceres_FOUND "NO")
if(Ceres_INCLUDE_DIR AND Glog_INCLUDE_DIR AND Ceres_LIBRARIES AND Glog_LIBRARIES)
    set(Ceres_FOUND "YES")
endif()
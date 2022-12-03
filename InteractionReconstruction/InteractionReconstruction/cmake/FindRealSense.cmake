# FIND_PATH(RealSense_ROOT_DIR include/pxcversion.h
	#--- FOR WINDOWS 
	# C:/Developer/Intel/RSSDK
    #--- Deployed 
	# ${CMAKE_SOURCE_DIR}/external/Intel/RSSDK)

FIND_PATH(RealSense2_ROOT_DIR include/librealsense2/rs.h
	#--- FOR WINDOWS 
	C:/Developer/Intel RealSense SDK 2.0
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/Intel RealSense SDK 2.0)


# if(RealSense_ROOT_DIR)
    # set(REALSENSE_INCLUDE_DIR ${RealSense_ROOT_DIR}/include)
    # set(REALSENSE_UTILITY_DIR ${RealSense_ROOT_DIR}/sample/common/include)

    # FIND_LIBRARY(REALSENSE_LIBRARY NAMES libpxcmd PATHS ${RealSense_ROOT_DIR}/lib/x64)
    # FIND_LIBRARY(REALSENSE_UTILITY_LIBRARY NAMES libpxcutilsmd PATHS ${RealSense_ROOT_DIR}/sample/common/lib/x64/v120)
# endif()

if(RealSense2_ROOT_DIR)
    set(REALSENSE2_INCLUDE_DIR ${RealSense2_ROOT_DIR}/include)
    FIND_LIBRARY(REALSENSE2_LIBRARY NAMES realsense2 PATHS ${RealSense2_ROOT_DIR}/lib/x64)
endif()

set(RealSense_FOUND "NO")
if(RealSense2_ROOT_DIR)
    set(RealSense_FOUND "YES")
endif()
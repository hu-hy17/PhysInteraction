FIND_PATH(PCL_INCLUDE_DIR pcl/pcl_base.h
	#--- FOR WINDOWS 
	C:/Developer/pcl/include
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/pcl/include)

FIND_PATH(PCL_LIB_DIR pcl_common_release.lib
	#--- FOR WINDOWS 
	C:/Developer/pcl/lib
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/pcl/lib)

if(PCL_LIB_DIR)
	set(PCL_LIBRARIES "")
	FIND_LIBRARY(PCL_COMMON NAMES pcl_common_release PATHS ${PCL_LIB_DIR})
	FIND_LIBRARY(PCL_GPU_CONTAINERS NAMES pcl_gpu_containers_release PATHS ${PCL_LIB_DIR})
	FIND_LIBRARY(PCL_FEATURES NAMES pcl_features_release PATHS ${PCL_LIB_DIR})
	list(APPEND PCL_LIBRARIES ${PCL_COMMON} ${PCL_GPU_CONTAINERS} ${PCL_FEATURES})
endif()

set(PCL_FOUND "NO")
if(PCL_INCLUDE_DIR AND PCL_LIBRARIES)
    set(PCL_FOUND "YES")
endif()
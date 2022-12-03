FIND_PATH(Flann_INCLUDE_DIR flann/flann.h
	#--- FOR WINDOWS 
	C:/Developer/include
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/flann/include)
	
message(STATUS "Flann_INCLUDE_DIR: ${Flann_INCLUDE_DIR}")

set(Flann_FOUND "NO")
if(Flann_INCLUDE_DIR)
    set(Flann_FOUND "YES")
endif()
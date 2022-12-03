FIND_PATH(Libigl_INCLUDE_DIR igl/AABB.h
	#--- FOR WINDOWS 
	C:/Developer/include
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/libigl/include)

set(Libigl_FOUND "NO")
if(Libigl_INCLUDE_DIR)
    set(Libigl_FOUND "YES")
endif()
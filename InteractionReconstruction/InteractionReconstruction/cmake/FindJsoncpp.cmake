FIND_PATH(Jsoncpp_ROOT_DIR include/json/json.h
	#--- FOR WINDOWS 
	C:/Developer
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/jsoncpp)

if(Jsoncpp_ROOT_DIR)
    set(Jsoncpp_INCLUDE_DIR ${Jsoncpp_ROOT_DIR}/include)
        
    FIND_LIBRARY(Jsoncpp_LIBRARIES NAMES jsoncpp PATHS 
        ${Jsoncpp_ROOT_DIR}/lib)
endif()

set(Jsoncpp_FOUND "NO")
if(Jsoncpp_INCLUDE_DIR AND Jsoncpp_LIBRARIES)
    set(Jsoncpp_FOUND "YES")
endif()
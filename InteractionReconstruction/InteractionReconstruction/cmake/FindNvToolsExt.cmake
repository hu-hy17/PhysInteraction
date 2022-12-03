FIND_PATH(NvToolsExt_INCLUDE_DIR nvToolsExt.h
	#--- FOR WINDOWS 
	"C:/Program Files/NVIDIA Corporation/NvToolsExt/include"
    ${NVTOOLSEXT_PATH}/include
    #--- Deployed 
	${CMAKE_SOURCE_DIR}/external/NVIDIA Corporation/NvToolsExt/include)

set(NvToolsExt_FOUND "NO")
if(NvToolsExt_INCLUDE_DIR)
    set(NvToolsExt_FOUND "YES")
endif()
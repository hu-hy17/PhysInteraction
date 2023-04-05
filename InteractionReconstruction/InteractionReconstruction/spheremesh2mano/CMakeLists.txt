file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
add_library(spheremesh2mano STATIC ${HDR} ${SRC})

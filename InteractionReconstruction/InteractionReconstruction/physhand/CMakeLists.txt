INCLUDE_DIRECTORIES(../)
file(GLOB_RECURSE SRC "*.cpp")
file(GLOB_RECURSE HDR "*.h")
add_library(physhand STATIC ${HDR} ${SRC})

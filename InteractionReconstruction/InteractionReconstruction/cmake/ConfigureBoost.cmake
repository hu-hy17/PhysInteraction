find_package(Boost REQUIRED)
if(Boost_FOUND)
    message(STATUS Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS})
    include_directories(${Boost_INCLUDE_DIRS})
    list(APPEND LIBRARIES ${Boost_LIBRARIES})
else()
    message(ERROR "Boost NOT FOUND!")
endif()
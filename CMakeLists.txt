cmake_minimum_required (VERSION 2.6)
project (PLDA_PMF)

set (CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(mf src/mf.cpp src/mf.h src/mf_common.cpp)
add_executable(pmf src/pmf.cpp src/pmf.h src/mf.h src/mf_common.cpp)

target_link_libraries(pmf ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(hello PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(hello PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

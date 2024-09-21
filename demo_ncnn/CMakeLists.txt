cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)

project(nanodet_demo)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -w ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -w ")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -w ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -w ")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV REQUIRED)

# SET(ncnn_DIR /home/usr/ncnn/build/install/lib/cmake/ncnn)
SET(ncnn_DIR /mnt/d/libs/ncnn/build/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
if(NOT TARGET ncnn)
    message(WARNING "ncnn NOT FOUND!  Please set ncnn_DIR environment variable")
else()
    message("ncnn FOUND ")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
)


add_executable(nanodet_demo main.cpp nanodet.cpp)

target_link_libraries(
    nanodet_demo
    ncnn
    ${OpenCV_LIBS}
)

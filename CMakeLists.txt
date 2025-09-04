cmake_minimum_required(VERSION 3.16)
project(Lightweight3DRegLib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 第三方库路径
set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# 设置 CMake 查找模块的路径
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

#-----------------------------
# 查找 Eigen3
#-----------------------------
#find_package(Eigen3 REQUIRED)
#if(NOT EIGEN3_FOUND)
 #   message(FATAL_ERROR "Eigen3 not found! Please install Eigen or provide FindEigen3.cmake in cmake/")
#endif()
#message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIRS}")
#include_directories(${EIGEN3_INCLUDE_DIRS})

#-----------------------------
# 查找 NanoFlann
#-----------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

find_package(OpenMesh REQUIRED)
include_directories(${OpenMesh_INCLUDE_DIRS})
#-----------------------------
# 添加子模块
#-----------------------------
add_subdirectory(cpp)


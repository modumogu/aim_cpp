cmake_minimum_required(VERSION 3.20)
project(aim_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# 源码是 UTF-8，MSVC 默认按 GBK 读中文注释会直接报语法错
if(MSVC)
  add_compile_options(/utf-8)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "注意: 包含文件:  ")
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} @ ${OpenCV_DIR}")

add_executable(aim aim.cpp)
target_link_libraries(aim PRIVATE ${OpenCV_LIBS})

# 把 dll 拷到 exe 旁边，省得配 PATH
set(_ocv_bin "${OpenCV_DIR}/../bin")
if(EXISTS "${_ocv_bin}/opencv_world4140.dll")
  add_custom_command(TARGET aim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${_ocv_bin}/opencv_world4140.dll" $<TARGET_FILE_DIR:aim>)
endif()

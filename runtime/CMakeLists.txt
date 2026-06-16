# runtime/CMakeLists.txt
#
# Phase 1 离线骨架：构建 elastic_runtime 静态库（后续增量加入 metrics_logger
# 等模块）。不依赖 llama.cpp 顶层 CMake，独立 `cmake -S runtime -B runtime/build`
# 即可。Phase 2 接入 llama.cpp 主构建时再加入 examples/elastic-cli 与 OpenCL 钩子。

cmake_minimum_required(VERSION 3.14)
project(elastic_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(_ELASTIC_SOURCES
    budget_watcher.cpp
    elastic_profile_writer.cpp
    metrics_logger.cpp
    weight_buffer_manager.cpp
    weight_buffer_manager_cpu.cpp
    plan_ir.cpp
    plan_executor.cpp
    plan_provider.cpp
)

# OpenCL 包装层只在 find_package 找到 OpenCL 时进静态库。
# 桌面无 OpenCL 时 elastic_runtime 只含纯逻辑；Android 构建（build_android.sh
# 通过 OpenCL_INCLUDE_DIR / OpenCL_LIBRARY 显式喂路径）会编进 wbm_opencl。
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "elastic_runtime: OpenCL 找到，构建 weight_buffer_manager_opencl")
    list(APPEND _ELASTIC_SOURCES weight_buffer_manager_opencl.cpp)
else()
    message(STATUS "elastic_runtime: 未找到 OpenCL，跳过 weight_buffer_manager_opencl")
endif()

add_library(elastic_runtime STATIC ${_ELASTIC_SOURCES})
target_include_directories(elastic_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Plan IR / provider 用 vendor 的 header-only nlohmann json(repo 根 vendor/)。
target_include_directories(elastic_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../vendor)
target_link_libraries(elastic_runtime PUBLIC Threads::Threads)
target_compile_options(elastic_runtime PRIVATE -Wall -Wextra -Wpedantic)

if(OpenCL_FOUND)
    target_link_libraries(elastic_runtime PUBLIC ${OpenCL_LIBRARIES})
    target_include_directories(elastic_runtime PUBLIC ${OpenCL_INCLUDE_DIRS})
endif()

option(ELASTIC_BUILD_TESTS "构建 elastic runtime 单测" ON)
if(ELASTIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests/elastic
                     ${CMAKE_BINARY_DIR}/tests_elastic)
endif()

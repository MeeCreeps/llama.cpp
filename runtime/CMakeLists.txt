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

add_library(elastic_runtime STATIC
    budget_watcher.cpp
    metrics_logger.cpp
)
target_include_directories(elastic_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(elastic_runtime PUBLIC Threads::Threads)
target_compile_options(elastic_runtime PRIVATE -Wall -Wextra -Wpedantic)

option(ELASTIC_BUILD_TESTS "构建 elastic runtime 单测" ON)
if(ELASTIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests/elastic
                     ${CMAKE_BINARY_DIR}/tests_elastic)
endif()

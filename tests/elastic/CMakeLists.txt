# tests/elastic/CMakeLists.txt
#
# 单测目标。由 runtime/CMakeLists.txt 通过 ELASTIC_BUILD_TESTS=ON 拉进来。

add_executable(test_budget_watcher test_budget_watcher.cpp)
target_link_libraries(test_budget_watcher PRIVATE elastic_runtime)
target_compile_options(test_budget_watcher PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_budget_watcher COMMAND test_budget_watcher)

# OpenCL 探针：验证 clReleaseMemObject 是否真的还 DRAM。
# 没装 OpenCL 时静默跳过，不影响其他 target 与 ctest。
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL 找到 (${OpenCL_LIBRARIES})，构建 probe_cl_release")
    add_executable(probe_cl_release probe_cl_release.cpp)
    target_link_libraries(probe_cl_release PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(probe_cl_release PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_compile_options(probe_cl_release PRIVATE -Wall -Wextra)
else()
    message(STATUS "未找到 OpenCL，跳过 probe_cl_release（装 ocl-icd-opencl-dev + 任一 ICD 后重 configure 即可）")
endif()

# tests/elastic/CMakeLists.txt
#
# 单测目标。由 runtime/CMakeLists.txt 通过 ELASTIC_BUILD_TESTS=ON 拉进来。

add_executable(test_budget_watcher test_budget_watcher.cpp)
target_link_libraries(test_budget_watcher PRIVATE elastic_runtime)
target_compile_options(test_budget_watcher PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME test_budget_watcher COMMAND test_budget_watcher)

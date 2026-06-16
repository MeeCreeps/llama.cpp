# ggml-cpu-elastic：CPU backend 套 WBM 弹性内存管理
# 复用 runtime/ 的 elastic 算法（WBM/BudgetWatcher）

find_package(Threads REQUIRED)
set(_ELASTIC_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime")
set(_ELASTIC_RUNTIME_SOURCES
    ${_ELASTIC_RUNTIME_DIR}/budget_watcher.cpp
    ${_ELASTIC_RUNTIME_DIR}/elastic_profile_writer.cpp
    ${_ELASTIC_RUNTIME_DIR}/metrics_logger.cpp
    ${_ELASTIC_RUNTIME_DIR}/weight_buffer_manager.cpp
    ${_ELASTIC_RUNTIME_DIR}/weight_buffer_manager_cpu.cpp
)

ggml_add_backend_library(ggml-cpu-elastic
    ggml-cpu-elastic.cpp
    ${_ELASTIC_RUNTIME_SOURCES}
    ../../include/ggml-cpu-elastic.h
)

target_link_libraries(ggml-cpu-elastic PRIVATE Threads::Threads)
target_include_directories(ggml-cpu-elastic PRIVATE ${_ELASTIC_RUNTIME_DIR})

set(TARGET llama-opencl-transform-bench)

find_package(OpenCL REQUIRED)

add_executable(${TARGET} opencl-transform-bench.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE ${OpenCL_LIBRARIES})
target_compile_definitions(${TARGET} PRIVATE
    LLAMA_OPENCL_KERNEL_DIR="${PROJECT_SOURCE_DIR}/ggml/src/ggml-opencl/kernels")

install(TARGETS ${TARGET} RUNTIME)

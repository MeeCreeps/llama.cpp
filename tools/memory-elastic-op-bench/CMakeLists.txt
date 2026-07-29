set(TARGET llama-memory-elastic-op-bench)

add_executable(${TARGET} memory-elastic-op-bench.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE llama)

install(TARGETS ${TARGET} RUNTIME)

find_package(OpenCL REQUIRED)

set(LAYOUT_TARGET llama-memory-layout-compute-bench)
add_executable(${LAYOUT_TARGET} layout-compute-bench.cpp)
target_compile_features(${LAYOUT_TARGET} PRIVATE cxx_std_17)
target_include_directories(${LAYOUT_TARGET} PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${LAYOUT_TARGET} PRIVATE llama ${OpenCL_LIBRARIES})

install(TARGETS ${LAYOUT_TARGET} RUNTIME)

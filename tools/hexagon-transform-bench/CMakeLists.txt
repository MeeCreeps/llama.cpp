set(TARGET llama-hexagon-transform-bench)

add_executable(${TARGET} hexagon-transform-bench.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE ggml ggml-hexagon)

install(TARGETS ${TARGET} RUNTIME)

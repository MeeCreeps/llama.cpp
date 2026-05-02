set(TARGET multi-lora-bench)

add_executable(${TARGET}
    main.cpp
    adapter_pool.cpp
    metrics.cpp
)

target_link_libraries(${TARGET} PRIVATE
    llama
    llama-common
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

install(TARGETS ${TARGET} RUNTIME)

cmake_minimum_required(VERSION 3.22)
project(Hadamard_CUDA LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75 89)

find_package(CUDAToolkit REQUIRED)

set(HADAMARD_COMMON_SOURCES
    src/main.cpp
    src/hadamard_kernel.cu
    src/quantize_kernel.cu
)

add_executable(hadamard_test
    ${HADAMARD_COMMON_SOURCES}
    tests/test_correctness.cu
    tests/test_fused.cu
)
target_compile_definitions(hadamard_test PRIVATE
    HADAMARD_BUILD_TEST=1
    HADAMARD_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(hadamard_bench
    ${HADAMARD_COMMON_SOURCES}
    bench/benchmark.cu
)
target_compile_definitions(hadamard_bench PRIVATE
    HADAMARD_BUILD_BENCH=1
    HADAMARD_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

foreach(target hadamard_test hadamard_bench)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=/W4>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -Xcompiler=-Wall>
        )
    endif()

    target_link_libraries(${target} PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
endforeach()

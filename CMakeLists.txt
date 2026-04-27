cmake_minimum_required(VERSION 3.16)
project(moss_tts_nano VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# =============================================================================
# Dependencies
# =============================================================================

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ONNX Runtime
set(ONNXRUNTIME_VERSION "1.23.2")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(ONNXRUNTIME_ARCH "x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(ONNXRUNTIME_ARCH "aarch64")
    else()
        set(ONNXRUNTIME_ARCH "x64")
    endif()
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ONNXRUNTIME_ARCH}-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-${ONNXRUNTIME_VERSION}.zip")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

FetchContent_Declare(onnxruntime
    URL ${ONNXRUNTIME_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

set(ONNXRUNTIME_ROOT ${onnxruntime_SOURCE_DIR})
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
else()
    set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
endif()

# SentencePiece
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(SPM_ABSL_PROVIDER "internal" CACHE STRING "" FORCE)

FetchContent_Declare(sentencepiece
    URL "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.1.tar.gz"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(sentencepiece)

# dr_libs (single-header audio libraries)
FetchContent_Declare(dr_libs
    URL "https://github.com/mackron/dr_libs/archive/master.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(dr_libs)

set(DR_LIBS_INCLUDE_DIR ${dr_libs_SOURCE_DIR})

# =============================================================================
# Targets
# =============================================================================

add_executable(moss-tts-nano MOSS-TTS-Nano.cpp)

target_include_directories(moss-tts-nano PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${DR_LIBS_INCLUDE_DIR}
)

target_link_libraries(moss-tts-nano PRIVATE
    ${ONNXRUNTIME_LIB}
    sentencepiece
)

if(UNIX AND NOT APPLE)
    target_link_libraries(moss-tts-nano PRIVATE pthread)
endif()

if(MSVC)
    target_compile_options(moss-tts-nano PRIVATE /W4 /std:c++latest)
else()
    target_compile_options(moss-tts-nano PRIVATE 
        -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS moss-tts-nano DESTINATION bin)

# =============================================================================
# Summary
# =============================================================================

message(STATUS "MOSS-TTS-Nano Build Configuration")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime:    ${ONNXRUNTIME_VERSION}")
message(STATUS "  Target:          moss-tts-nano")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")

cmake_minimum_required(VERSION 3.14)

project(Codec63
  VERSION 3.0
  DESCRIPTION "CMake configuration file for Codec63 in C"
  LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

find_package(SISCI REQUIRED)

# Use the C standard from 2011
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
set(CMAKE_CXX_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
# Make sure Debug mode flags are similar to Release mode flags
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize")

# CUDA architectures settings
if(EXISTS "/etc/nv_tegra_release")
  # The CC list for Tegras and Jetson will require manual updates
  set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")

  # ARM architecture flags for CUDA and C++
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -march=armv8.2-a")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a")

  # Platform flag
  set(TEGRA TRUE)
else()
  # The CC list for discrete GPUs will require CMake updates
  set(CMAKE_CUDA_ARCHITECTURES "all-major"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")

  # Platform flag
  set(TEGRA FALSE)
endif()

# Debug compiles with -g or equivalent for debugging.
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

# Main CUDA library - same for both platforms
add_library(c63
  STATIC 
  tables.cu tables.h
  io.cu io.h
  common.cu common.h
  me.cu me.h)

# Add quantdct library only on Tegra
if(TEGRA)
  # C++ library for quantdct
  add_library(cpplibs STATIC
    quantdct.cpp quantdct.h)
endif()

# Executables
add_executable(c63enc
  c63enc.cu c63.h
  c63_write.cu c63_write.h)

add_executable(c63dec
  c63dec.cu c63.h)

add_executable(c63pred
  c63dec.cu c63.h)
target_compile_options(c63pred PUBLIC -DC63_PRED)

add_executable(c63server
  c63server.cu c63.h)
target_compile_options(c63server PUBLIC -D_REENTRANT)

add_executable(c63client
  c63client.cu c63.h
  c63_write.cu c63_write.h)
target_compile_options(c63client PUBLIC -D_REENTRANT)

# Link libraries differently based on platform
if(TEGRA)
  # On Tegra, link with both c63 and cpplibs
  target_link_libraries(c63enc c63 cpplibs m)
  target_link_libraries(c63dec c63 cpplibs m)
  target_link_libraries(c63pred c63 cpplibs m)
  target_link_libraries(c63server c63 cpplibs m SISCI)
  target_link_libraries(c63client c63 cpplibs m SISCI)
else()
  # On x86, link with only c63
  target_link_libraries(c63enc c63 m)
  target_link_libraries(c63dec c63 m)
  target_link_libraries(c63pred c63 m)
  target_link_libraries(c63server c63 m SISCI)
  target_link_libraries(c63client c63 m SISCI)
endif()
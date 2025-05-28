cmake_minimum_required(VERSION 3.14)

#
# This tells CMake that your project's name is homeexam-01, and that the programs
# here are written in C.
# 
project(Codec63
	VERSION 3.0
	DESCRIPTION "CMake configuration file for Codec63 in C"
        LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")

find_package(SISCI REQUIRED)

# Use the C standard from 2011. No effect on Windows (Visual Studio ignores
# the standards anyway).
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS ON) # definitely use GNU extensions (actually default)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
# Make sure Debug mode flags are similar to Release mode flags
set(CMAKE_C_FLAGS_DEBUG "-g -O1 -fno-tree-vectorize")

# Debug compiles with -g or equivalent for debugging.
# Change to Release for good speed.
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE Release)

#
# If this is a Tegra or something similar, set our processor subrevision.
# On Mac, CMake adds arch flags by itself.
#
if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  if(EXISTS "/etc/nv_tegra_release")
    add_compile_options(-march=armv8.2-a)
  endif()
else()
  message(FATAL "Not compiling with an gcc")
endif()

add_library(c63
            STATIC dsp.c dsp.h
	           tables.c tables.h
	           io.c io.h
	           common.c common.h
	           me.c me.h)

add_executable(c63enc
               c63enc.c c63.h
	       c63_write.c c63_write.h)
target_link_libraries(c63enc c63 m)

add_executable(c63dec
               c63dec.c c63.h)
target_link_libraries(c63dec c63 m)

add_executable(c63pred
               c63dec.c c63.h)
target_link_libraries(c63pred c63 m)
target_compile_options(c63pred PUBLIC -DC63_PRED)

add_executable(c63server
               c63server.c c63.h)
target_compile_options(c63server PUBLIC -D_REENTRANT)
target_link_libraries(c63server c63 m)
target_link_libraries(c63server SISCI)

add_executable(c63client
               c63client.c c63.h
               c63_write.c c63_write.h)
target_compile_options(c63client PUBLIC -D_REENTRANT)
target_link_libraries(c63client c63 m)
target_link_libraries(c63client SISCI)


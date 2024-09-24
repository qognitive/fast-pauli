#
# Boilerplate CMakeLists.txt for C++ projects
#
cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

include(cmake/CPM.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Need to enforce -fPIC across whole project to build shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# TODO different compilation flags for Debug/Release modes; also pull in
# target_compile_options for fast_pauli target defined below
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# TODO adding a package lock to help with SBOM
# cpmusepackagelock(package-lock.cmake)

#
# Project specific configuration
#

# Dependencies
cpmaddpackage("gh:doctest/doctest@2.4.11")
cpmaddpackage("gh:pybind/pybind11@2.11.1")
cpmaddpackage("gh:fmtlib/fmt#10.2.1")
cpmaddpackage("gh:kokkos/mdspan#b885a2c60ad42f9e1aaa0d317a38105b950cbed0")
cpmaddpackage("gh:wjakob/nanobind#v2.0.0")

#
# User Options
#

# TODO NOT WORKING YET

# option(ENABLE_COVERAGE "Enable coverage reporting" OFF) if(ENABLE_COVERAGE)
# message(STATUS "[FAST_PAULI] Enabling coverage reporting") message(STATUS
# "[FAST_PAULI]") set(FAST_PAULI_EXTRA_CXX_COMPILE_FLAGS "-coverage")
# set(FAST_PAULI_EXTRA_CXX_LD_FLAGS "-lgcov;--coverage") endif()

#
# Fast Pauli
#

project(fast_pauli LANGUAGES CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Set up OpenMP
find_package(OpenMP REQUIRED)

# Setup Python
find_package(Python 3.10 # This is a minimum version
             COMPONENTS Interpreter Development REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address
# -fno-omit-frame-pointer")

# Our primary target
add_library(fast_pauli INTERFACE)
target_include_directories(
  fast_pauli INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/fast_pauli/cpp/include/)
target_link_libraries(fast_pauli INTERFACE fmt::fmt mdspan OpenMP::OpenMP_CXX)
target_compile_options(fast_pauli INTERFACE -g3 -Wall -Wextra -Werror)

# Testing
include(CTest)
enable_testing()
add_subdirectory(fast_pauli/cpp/tests)

# Examples
add_subdirectory(fast_pauli/cpp/examples)

# Python bindings
add_subdirectory(fast_pauli/cpp/src)

option(ENABLE_DOCS "Enable documentation generation" OFF)
if(ENABLE_DOCS)
  add_subdirectory(docs)
endif()

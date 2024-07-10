#
# Boilerplate CMakeLists.txt for C++ projects
#
cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

include(cmake/CPM.cmake)
# TODO adding a package lock to help with SBOM
# cpmusepackagelock(package-lock.cmake)

#
# Project specific configuration
#

# Dependencies
cpmaddpackage("gh:doctest/doctest@2.4.11")
# cpmaddpackage("gh:p-ranav/argparse@3.0")
cpmaddpackage("gh:pybind/pybind11@2.11.1")
cpmaddpackage("gh:fmtlib/fmt#10.2.1")
cpmaddpackage("gh:kokkos/mdspan#b885a2c60ad42f9e1aaa0d317a38105b950cbed0")

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

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set up OpenMP
find_package(OpenMP REQUIRED)

# Our primary target
add_library(fast_pauli INTERFACE)
target_include_directories(fast_pauli
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
target_link_libraries(fast_pauli INTERFACE fmt::fmt mdspan OpenMP::OpenMP_CXX)
target_compile_options(
  fast_pauli
  INTERFACE -g3
            -O3
            -march=native
            -Wall
            -Wextra
            -Werror
            # -stdlib=libc++
            ${FAST_PAULI_EXTRA_CXX_COMPILE_FLAGS})
# target_link_options(fast_pauli INTERFACE ${FAST_PAULI_EXTRA_CXX_LD_FLAGS}
# -fuse-ld=mold) target_compile_definitions(fast_pauli INTERFACE)

# Testing
include(CTest)
enable_testing()
add_subdirectory(tests)

# Examples
add_subdirectory(examples)

cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

include(cmake/CPM.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Need to enforce -fPIC across whole project to build shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(
  Python 3.12
  COMPONENTS Interpreter Development
  REQUIRED)

# TODO different compilation flags for Debug/Release modes; also pull in
# target_compile_options for fast_pauli target defined below
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Dependencies
cpmaddpackage("gh:wjakob/nanobind#v2.0.0")
cpmaddpackage("gh:fmtlib/fmt#10.2.1")
cpmaddpackage("gh:kokkos/mdspan#b885a2c60ad42f9e1aaa0d317a38105b950cbed0")

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${nanobind_BUILD_DIR})
nanobind_add_module(mdspan_wrapper mdspan_wrapper.cpp)
target_link_libraries(mdspan_wrapper PRIVATE mdspan fmt::fmt)

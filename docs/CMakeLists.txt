# See
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/

find_package(Doxygen REQUIRED)

# Find all the public headers
get_target_property(fast_pauli_PUBLIC_HEADER_DIR fast_pauli
                    INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE fast_pauli_PUBLIC_HEADERS
     ${fast_pauli_PUBLIC_HEADER_DIR}/*.hpp)

message(STATUS "fast_pauli_PUBLIC_HEADERS: ${fast_pauli_PUBLIC_HEADERS}")

# This will be the main output of our command
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)

add_custom_command(
  OUTPUT ${DOXYGEN_INDEX_FILE}
  DEPENDS ${fast_pauli_PUBLIC_HEADERS}
  COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  MAIN_DEPENDENCY Doxyfile
  COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

# ##############################################################################
# This code is part of Fast Pauli.
#
# (C) Copyright Qognitive Inc 2024.
#
# This code is licensed under the BSD 2-Clause License. You may obtain a copy of
# this license in the LICENSE.txt file in the root directory of this source
# tree.
#
# Any modifications or derivative works of this code must retain this copyright
# notice, and modified files need to carry a notice indicating that they have
# been altered from the originals.

# ##############################################################################

# Grab all *.cpp files in the directory
file(GLOB TEST_SRC CONFIGURE_DEPENDS "*.cpp")

foreach(TEST_SRC_FILE ${TEST_SRC})
  # Strip file extension
  get_filename_component(TEST_NAME ${TEST_SRC_FILE} NAME_WLE)
  # Add test executable
  add_executable(${TEST_NAME} ${TEST_SRC_FILE})
  target_link_libraries(${TEST_NAME} PUBLIC ${PROJECT_NAME} doctest::doctest)
  add_test(${TEST_NAME} ${TEST_NAME})
endforeach()

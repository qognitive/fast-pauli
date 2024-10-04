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
file(GLOB EXAMPLE CONFIGURE_DEPENDS "*.cpp")

foreach(EXAMPLE_FILE ${EXAMPLE})
  # Strip file extension
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WLE)
  # Add example executable
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${EXAMPLE_NAME} PUBLIC ${PROJECT_NAME})
endforeach()

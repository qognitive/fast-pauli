# Grab all *.cpp files in the directory
file(GLOB FAST_PAULI_BINDINGS CONFIGURE_DEPENDS "*.cpp")

find_package(pybind11 REQUIRED)

# # TODO use different set of parameters depending on CMAKE_BUILD_TYPE
# pybind11_add_module(_fast_pauli SHARED ${FAST_PAULI_BINDINGS})
# target_link_libraries(_fast_pauli PUBLIC fast_pauli) # TODO recheck from where
# it is pulling in numpy headers

# install(TARGETS _fast_pauli LIBRARY DESTINATION
# "${PROJECT_SOURCE_DIR}/fast_pauli")

# See
# https://nanobind.readthedocs.io/en/latest/api_cmake.html#command:nanobind_add_module
# for more information on the nanobind_add_module command

# Since the code for our bindings contain several helper functions we allow the
# bindings to respect the optimization flags used by the project as a whole

# TODO RENAME to fppy
nanobind_add_module(_fast_pauli NOMINSIZE NOSTRIP PROTECT_STACK fast_pauli.cpp)
target_link_libraries(_fast_pauli PRIVATE fast_pauli)

install(TARGETS _fast_pauli LIBRARY DESTINATION "${PROJECT_BINARY_DIR}/fppy")
install(TARGETS _fast_pauli
        LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/fast_pauli")

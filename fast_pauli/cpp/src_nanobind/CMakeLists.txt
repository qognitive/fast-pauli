# Since the code for our bindings contain several helper functions we allow the
# bindings to respect the optimization flags used by the project as a whole
nanobind_add_module(fppy NOMINSIZE fast_pauli.cpp)
target_link_libraries(fppy PRIVATE fast_pauli)

install(TARGETS fppy LIBRARY DESTINATION "${PROJECT_BINARY_DIR}/fppy")
install(TARGETS fppy LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/fast_pauli")

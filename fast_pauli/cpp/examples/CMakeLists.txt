# Grab all *.cpp files in the directory
file(GLOB EXAMPLE CONFIGURE_DEPENDS "*.cpp")

foreach(EXAMPLE_FILE ${EXAMPLE})
  # Strip file extension
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WLE)
  # Add example executable
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
  target_link_libraries(${EXAMPLE_NAME} PUBLIC ${PROJECT_NAME})
endforeach()

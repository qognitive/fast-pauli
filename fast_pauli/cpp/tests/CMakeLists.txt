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

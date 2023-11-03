cmake_minimum_required(VERSION 3.22.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

project(predictor_node)

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

ament_auto_find_build_dependencies()

generate_parameter_library(
  predictor_node_parameters # cmake target name for the parameter library
  src/predictor_node_parameters.yaml # path to input yaml file
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC
  include/${PROJECT_NAME}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  predictor_node_parameters
  Eigen3::Eigen
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN helios_cv::PredictorNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING) 
  find_package(ament_lint_auto REQUIRED)
  list(
    APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
  )
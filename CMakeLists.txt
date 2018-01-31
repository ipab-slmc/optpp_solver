cmake_minimum_required(VERSION 2.8.12)
project(optpp_solver)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  exotica
  optpp_catkin
)

AddInitializer(
  OptppMotionSolver
  OptppIKLBFGS
  OptppIKCG
  OptppIKQNewton
  OptppIKFDNewton
  OptppIKGSS
  OptppTrajLBFGS
  OptppTrajCG
  OptppTrajQNewton
  OptppTrajFDNewton
  OptppTrajGSS
)
GenInitializers()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES optpp_core
  CATKIN_DEPENDS exotica optpp_catkin
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(optpp_core src/optpp_core.cpp)
target_link_libraries(optpp_core ${catkin_LIBRARIES})

add_library(optpp_ik src/optpp_ik.cpp src/optpp_traj.cpp)
target_link_libraries(optpp_ik optpp_core ${catkin_LIBRARIES})
add_dependencies(optpp_ik optpp_core ${PROJECT_NAME}_initializers)

install(TARGETS optpp_core optpp_ik
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(FILES exotica_plugins.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY resources/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/resources)
FILE(GLOB python_examples "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*")
catkin_install_python(PROGRAMS ${python_examples} DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

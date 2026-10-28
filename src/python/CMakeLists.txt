#=========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

# Core Parthenon Python bindings and the pure-Python parthenon package. Only used with
# PARTHENON_ENABLE_PYTHON=ON.
#
# Nothing is built here: parthenon_add_python_module compiles the bindings into each
# application module and copies the parthenon package into each application package.

set(_binding_sources
  bindings/bind_all.cpp
  bindings/driver.cpp
  bindings/manager.cpp
  bindings/mesh.cpp
  bindings/parameter_input.cpp
  bindings/state.cpp
)
list(TRANSFORM _binding_sources PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
# Used by parthenon_add_python_module
set_property(GLOBAL PROPERTY PARTHENON_PYTHON_BINDING_SOURCES ${_binding_sources})

set(_package_files
  parthenon/__init__.py
  parthenon/cli.py
  parthenon/py.typed
)
list(TRANSFORM _package_files PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set_property(GLOBAL PROPERTY PARTHENON_PYTHON_PACKAGE_FILES ${_package_files})

# Lets application modules include parthenon_nanobind.hpp
add_library(parthenon_python_headers INTERFACE)
target_include_directories(parthenon_python_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings)

install(FILES bindings/parthenon_nanobind.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/python)
install(FILES ${PROJECT_SOURCE_DIR}/cmake/ParthenonPython.cmake
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake")
# Downstream applications compile the bindings into their own module
install(FILES ${_binding_sources}
  DESTINATION ${CMAKE_INSTALL_DATADIR}/parthenon/python/bindings)
install(FILES ${_package_files}
  DESTINATION ${CMAKE_INSTALL_DATADIR}/parthenon/python/parthenon)

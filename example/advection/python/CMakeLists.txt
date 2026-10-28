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

# Python version of the advection example. The C++ advection package and driver are
# compiled into the _advection module of the importable package advection, a library that
# also contains Parthenon's Python package. The application, run_advection.py, imports it
# to set up and run the problem, like a downstream code would.

parthenon_add_python_module(_advection
  PACKAGE advection
  SOURCES advection_bindings.cpp
  PYTHON_SOURCES advection/__init__.py
  LINK_LIBRARIES advection-lib
  OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  STUBS
)
lint_target(_advection)

# The application, next to the package so that it can import it
configure_file(run_advection.py ${CMAKE_CURRENT_BINARY_DIR}/run_advection.py COPYONLY)

# Launcher that can be used like the advection-example executable
set(ADVECTION_PY_PYTHON ${Python_EXECUTABLE})
set(ADVECTION_PY_SCRIPT ${CMAKE_CURRENT_BINARY_DIR}/run_advection.py)
configure_file(advection-example-py.in
  ${PROJECT_BINARY_DIR}/example/advection/advection-example-py
  @ONLY
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
                   WORLD_READ WORLD_EXECUTE)
add_custom_target(advection-example-py ALL DEPENDS _advection)

# Tests of the parthenon Python package (tst/python), which use this package
if (PARTHENON_ENABLE_UNIT_TESTS)
  add_test(NAME python:cli
    COMMAND ${Python_EXECUTABLE} -m unittest -v test_cli
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tst/python)
  set_tests_properties(python:cli PROPERTIES
    LABELS "python"
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}")
endif()

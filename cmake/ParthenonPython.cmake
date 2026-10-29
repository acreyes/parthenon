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

# Helpers for building nanobind Python modules that use Parthenon.
#
# Requires that Python (Interpreter and Development.Module components) and nanobind
# have been found, e.g.,
#   find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
#   find_package(nanobind CONFIG REQUIRED)
#
# parthenon_add_python_module(<name>
#   PACKAGE <package>
#   SOURCES <sources>...
#   [PYTHON_SOURCES <files>...]
#   [LINK_LIBRARIES <libraries>...]
#   [OUTPUT_DIRECTORY <dir>]
#   [STUBS]
#   [INSTALL_DESTINATION <dir>])
#
# Builds the importable Python package <package> in <dir> (OUTPUT_DIRECTORY, default
# CMAKE_CURRENT_BINARY_DIR):
#
#   <dir>/<package>/
#     <PYTHON_SOURCES>         the application's Python files (e.g., __init__.py)
#     <name>.<ext>             the extension module, built from SOURCES and the Parthenon
#                              bindings, and linked against Parthenon and LINK_LIBRARIES
#     parthenon/               Parthenon's Python package: the Parthenon bindings and
#                              pure-Python helpers (e.g., parthenon.cli)
#
# so that, with <dir> on the Python path, the application's code can use
# "from <package> import <name>, parthenon". The module's NB_MODULE must call
# parthenon::python::ImportParthenon(m) (from parthenon_nanobind.hpp) before using
# Parthenon types; it registers the Parthenon bindings as the submodule <name>.parthenon,
# which <package>.parthenon re-exports. PYTHON_SOURCES are copied into the top level of
# the package.
#
# With STUBS, type stubs for the extension module (<name>/__init__.pyi and
# <name>/parthenon.pyi) and a py.typed marker are generated in the package with
# nanobind's stubgen after the module is built. With INSTALL_DESTINATION, the package is
# installed to that directory.
#
# Parthenon and Kokkos are linked into the module, so they must be compiled as position
# independent code. Only one module built this way should be loaded in a Python
# process, since each one contains its own copy of Parthenon and Kokkos.

function(parthenon_add_python_module name)
  cmake_parse_arguments(ARG "STUBS" "PACKAGE;OUTPUT_DIRECTORY;INSTALL_DESTINATION"
                        "SOURCES;PYTHON_SOURCES;LINK_LIBRARIES" ${ARGN})
  if (NOT ARG_PACKAGE)
    message(FATAL_ERROR "parthenon_add_python_module(${name}): PACKAGE is required")
  endif()
  if (NOT ARG_OUTPUT_DIRECTORY)
    set(ARG_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  set(_package_dir ${ARG_OUTPUT_DIRECTORY}/${ARG_PACKAGE})

  get_property(_binding_sources GLOBAL PROPERTY PARTHENON_PYTHON_BINDING_SOURCES)
  get_property(_parthenon_files GLOBAL PROPERTY PARTHENON_PYTHON_PACKAGE_FILES)
  if (NOT _binding_sources)
    # Installed Parthenon
    file(GLOB _binding_sources ${PARTHENON_PYTHON_BINDINGS_DIR}/*.cpp)
    file(GLOB _parthenon_files ${PARTHENON_PYTHON_PACKAGE_DIR}/*)
  endif()

  # A separate nanobind domain per module keeps the Parthenon types of different
  # modules from clashing
  nanobind_add_module(${name} NB_DOMAIN ${name} ${ARG_SOURCES} ${_binding_sources})
  target_link_libraries(${name} PRIVATE Parthenon::parthenon ${ARG_LINK_LIBRARIES})
  if (TARGET parthenon_python_headers)
    target_link_libraries(${name} PRIVATE parthenon_python_headers)
  else()
    target_include_directories(${name} PRIVATE ${PARTHENON_PYTHON_INCLUDE_DIR})
  endif()
  set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${_package_dir})

  # Copy the Python files into the package
  set(_copies)
  foreach (_src IN LISTS ARG_PYTHON_SOURCES _parthenon_files)
    get_filename_component(_src ${_src} ABSOLUTE)
    get_filename_component(_file ${_src} NAME)
    if (_src IN_LIST _parthenon_files)
      set(_dst ${_package_dir}/parthenon/${_file})
    else()
      set(_dst ${_package_dir}/${_file})
    endif()
    add_custom_command(OUTPUT ${_dst}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_src} ${_dst}
      DEPENDS ${_src})
    list(APPEND _copies ${_dst})
  endforeach()
  # The Parthenon bindings of this module, re-exported by the parthenon package
  file(GENERATE OUTPUT ${_package_dir}/parthenon/_core.py CONTENT
    "# Generated by parthenon_add_python_module: the Parthenon bindings of ${name}\n\
from ..${name}.parthenon import *  # noqa: F401,F403\n")
  add_custom_target(${name}_python ALL DEPENDS ${_copies})
  add_dependencies(${name} ${name}_python)

  if (ARG_STUBS)
    nanobind_add_stub(${name}_stubs
      MODULE ${ARG_PACKAGE}.${name}
      RECURSIVE
      PYTHON_PATH ${ARG_OUTPUT_DIRECTORY}
      OUTPUT_PATH ${_package_dir}
      OUTPUT ${_package_dir}/${name}/__init__.pyi ${_package_dir}/${name}/parthenon.pyi
      MARKER_FILE ${_package_dir}/py.typed
      DEPENDS ${name} ${name}_python)
  endif()

  if (ARG_INSTALL_DESTINATION)
    install(DIRECTORY ${_package_dir} DESTINATION ${ARG_INSTALL_DESTINATION}
      USE_SOURCE_PERMISSIONS)
  endif()
endfunction()

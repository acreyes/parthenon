//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI.

#ifndef PYTHON_BINDINGS_PARTHENON_NANOBIND_HPP_
#define PYTHON_BINDINGS_PARTHENON_NANOBIND_HPP_

// Common include for Parthenon nanobind bindings. The Parthenon bindings are compiled
// into each application module (built with parthenon_add_python_module, see
// cmake/ParthenonPython.cmake), which should include this header and call
// parthenon::python::ImportParthenon(m) at the top of its NB_MODULE so that the
// Parthenon types are registered before they are used.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace parthenon {
namespace python {

namespace nb = nanobind;

// Name of the Python package that holds the core Parthenon bindings
constexpr char kParthenonModule[] = "parthenon";

// Bind all of the core Parthenon types and functions into m
void BindAll(nb::module_ &m);

// Register the Parthenon bindings as the submodule m.parthenon and return it. The
// submodule is also added to sys.modules so that it can be imported by its qualified
// name (e.g., "mypackage._myapp.parthenon"), which is how the parthenon Python package
// that parthenon_add_python_module puts next to the module re-exports it.
inline nb::module_ ImportParthenon(nb::module_ &m) {
  nb::module_ parthenon = m.def_submodule(kParthenonModule);
  BindAll(parthenon);
  nb::module_::import_("sys").attr("modules")[parthenon.attr("__name__")] = parthenon;
  return parthenon;
}

void BindParameterInput(nb::module_ &m);
void BindManager(nb::module_ &m);
void BindState(nb::module_ &m);
void BindMesh(nb::module_ &m);
void BindDriver(nb::module_ &m);

} // namespace python
} // namespace parthenon

#endif // PYTHON_BINDINGS_PARTHENON_NANOBIND_HPP_

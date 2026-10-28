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

// Bindings for the parts of the advection example that stay in C++: the advection
// package and the driver that evolves it. The problem generator and main program are
// written in Python (run_advection.py).

#include <parthenon_nanobind.hpp>

#include "../advection_driver.hpp"
#include "../advection_package.hpp"

namespace nb = nanobind;

NB_MODULE(_advection, m) {
  m.doc() = "C++ components of the Parthenon advection example";
  // Registers the Parthenon types used below
  parthenon::python::ImportParthenon(m);

  m.def("Initialize", &advection_package::Initialize, nb::arg("pin"),
        "Create the advection package.");
  m.def("PreFill", &advection_package::PreFill, nb::arg("rc"));
  m.def("PostFill", &advection_package::PostFill, nb::arg("rc"));
  m.def("UserWorkAfterLoop", &advection_example::UserWorkAfterLoop, nb::arg("mesh"),
        nb::arg("pin"), nb::arg("tm"), "Compute and report errors of the solution.");
  m.def("UserMeshWorkBeforeOutput", &advection_example::UserMeshWorkBeforeOutput,
        nb::arg("mesh"), nb::arg("pin"), nb::arg("tm"));

  nb::class_<advection_example::AdvectionDriver, parthenon::MultiStageDriver>(
      m, "AdvectionDriver")
      .def(nb::init<parthenon::ParameterInput *, parthenon::ApplicationInput *,
                    parthenon::Mesh *>(),
           nb::arg("pin"), nb::arg("app_input"), nb::arg("mesh"),
           // The driver refers to these, so they must outlive it
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(), nb::keep_alive<1, 4>());
}

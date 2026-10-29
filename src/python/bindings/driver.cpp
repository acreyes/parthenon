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

#include "driver/driver.hpp"
#include "driver/multistage.hpp"
#include "parthenon_nanobind.hpp"
#include "utils/utils.hpp"

namespace parthenon {
namespace python {

void BindDriver(nb::module_ &m) {
  nb::enum_<DriverStatus>(m, "DriverStatus")
      .value("complete", DriverStatus::complete)
      .value("timeout", DriverStatus::timeout)
      .value("failed", DriverStatus::failed);

  // Drivers are defined in C++ by applications, which bind their concrete driver
  // as a subclass of one of these (see example/advection/python).
  nb::class_<Driver>(m, "Driver")
      .def(
          "Execute",
          [](Driver &self) {
            // Python installs its own signal handlers, so restore Parthenon's
            SignalHandler::SignalHandlerInit();
            // Python callbacks reacquire the GIL when they are called
            nb::gil_scoped_release release;
            return self.Execute();
          },
          "Run the driver to completion.");
  nb::class_<EvolutionDriver, Driver>(m, "EvolutionDriver");
  nb::class_<MultiStageDriver, EvolutionDriver>(m, "MultiStageDriver");
}

} // namespace python
} // namespace parthenon

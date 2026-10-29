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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "application_input.hpp"
#include "parthenon_manager.hpp"
#include "parthenon_nanobind.hpp"
#include "runtime_options.hpp"
#include "utils/utils.hpp"

namespace parthenon {
namespace python {

namespace {
// ApplicationInput::ProcessPackages takes a std::unique_ptr<ParameterInput> &, which
// cannot be passed to Python. This adapter lets Python provide a function that takes a
// ParameterInput instead, and lets the property getter recover that function.
struct ProcessPackagesAdapter {
  std::function<Packages_t(ParameterInput *)> func;
  Packages_t operator()(std::unique_ptr<ParameterInput> &pin) const {
    return func(pin.get());
  }
};

// Drop every callback (and therefore every Python callable) held by the manager and
// every Python object held in package params, so that nothing that references Python
// or Kokkos outlives finalization.
void ReleaseCallbacks(ParthenonManager &pman) {
  if (pman.pmesh) {
    for (auto &[name, pkg] : pman.pmesh->packages.AllPackages()) {
      pkg->AllParams().reset();
    }
  }
  pman.ProcessPackages = ParthenonManager::ProcessPackagesDefault;
  *pman.app_input = ApplicationInput();
}
} // namespace

void BindManager(nb::module_ &m) {
  m.def("ShowConfig", &ShowConfig, "Print the Parthenon build configuration.");

  nb::enum_<ParthenonStatus>(m, "ParthenonStatus")
      .value("ok", ParthenonStatus::ok)
      .value("complete", ParthenonStatus::complete)
      .value("error", ParthenonStatus::error);

  nb::class_<RuntimeOptions>(m, "RuntimeOptions",
                             "Runtime options corresponding to the command line flags "
                             "(other than the input file).")
      .def(
          "__init__",
          [](RuntimeOptions *self, std::optional<std::string> restart_filename,
             bool analysis, std::optional<std::string> run_dir,
             std::optional<std::string> params_regex, int mesh_flag, int wall_time_limit,
             std::optional<int> watchdog_timeout, std::vector<std::string> kokkos_args) {
            new (self) RuntimeOptions{std::move(restart_filename),
                                      analysis,
                                      std::move(run_dir),
                                      std::move(params_regex),
                                      mesh_flag,
                                      wall_time_limit,
                                      watchdog_timeout,
                                      std::move(kokkos_args)};
          },
          nb::kw_only(), nb::arg("restart_filename") = nb::none(),
          nb::arg("analysis") = false, nb::arg("run_dir") = nb::none(),
          nb::arg("params_regex") = nb::none(), nb::arg("mesh_flag") = 0,
          nb::arg("wall_time_limit") = 0, nb::arg("watchdog_timeout") = nb::none(),
          nb::arg("kokkos_args") = std::vector<std::string>{})
      .def_rw("restart_filename", &RuntimeOptions::restart_filename)
      .def_rw("analysis", &RuntimeOptions::analysis)
      .def_rw("run_dir", &RuntimeOptions::run_dir)
      .def_rw("params_regex", &RuntimeOptions::params_regex)
      .def_rw("mesh_flag", &RuntimeOptions::mesh_flag)
      .def_rw("wall_time_limit", &RuntimeOptions::wall_time_limit)
      .def_rw("watchdog_timeout", &RuntimeOptions::watchdog_timeout)
      .def_rw("kokkos_args", &RuntimeOptions::kokkos_args)
      .def_prop_ro("is_restart", &RuntimeOptions::IsRestart);

  using AI = ApplicationInput;
  nb::class_<AI>(m, "ApplicationInput",
                 "Application callbacks. Assign Python callables to the hooks.")
      .def_prop_rw(
          "ProcessPackages",
          [](AI &self) -> std::optional<std::function<Packages_t(ParameterInput *)>> {
            if (auto *adapter = self.ProcessPackages.target<ProcessPackagesAdapter>()) {
              return adapter->func;
            }
            return std::nullopt;
          },
          [](AI &self, std::optional<std::function<Packages_t(ParameterInput *)>> f) {
            if (f) {
              self.ProcessPackages = ProcessPackagesAdapter{std::move(*f)};
            } else {
              self.ProcessPackages = nullptr;
            }
          },
          nb::arg("func").none(), "Callable (pin: ParameterInput) -> Packages")
      .def_rw("InitUserMeshData", &AI::InitUserMeshData)
      .def_rw("MeshProblemGenerator", &AI::MeshProblemGenerator)
      .def_rw("PreStepMeshUserWorkInLoop", &AI::PreStepMeshUserWorkInLoop)
      .def_rw("PostStepMeshUserWorkInLoop", &AI::PostStepMeshUserWorkInLoop)
      .def_rw("UserMeshWorkBeforeOutput", &AI::UserMeshWorkBeforeOutput)
      .def_rw("UserWorkAfterLoop", &AI::UserWorkAfterLoop)
      .def_rw("UserWorkBeforeLoop", &AI::UserWorkBeforeLoop)
      .def_rw("InitMeshBlockUserData", &AI::InitMeshBlockUserData)
      .def_rw("ProblemGenerator", &AI::ProblemGenerator)
      .def_rw("PostInitialization", &AI::PostInitialization)
      .def("RegisterDefaultReflectingBoundaryConditions",
           &AI::RegisterDefaultReflectingBoundaryConditions);

  using PM = ParthenonManager;
  nb::class_<PM>(m, "ParthenonManager")
      .def(nb::init<>())
      .def(
          "ParthenonInitEnv",
          [](PM &self, std::vector<std::string> args) {
            // argv must stay valid for the duration of the call
            std::vector<char *> argv;
            for (auto &arg : args) {
              argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            return self.ParthenonInitEnv(static_cast<int>(args.size()), argv.data());
          },
          nb::arg("argv"),
          "Initialize from command line arguments. argv[0] is the program name.")
      .def(
          "ParthenonInitEnv",
          [](PM &self, const ParameterInput &pin, RuntimeOptions opts) {
            // Objects created in Python cannot transfer ownership to a std::unique_ptr,
            // so the manager gets its own copy of the parameters.
            auto pin_copy = std::make_unique<ParameterInput>();
            pin_copy->Overlay(pin);
            return self.ParthenonInitEnv(std::move(pin_copy), std::move(opts));
          },
          nb::arg("pin"), nb::arg("opts") = RuntimeOptions{},
          "Initialize from a populated ParameterInput. The manager keeps a copy of the "
          "parameters (use ParthenonManager.pinput to access it afterwards).")
      .def("ParthenonInitPackagesAndMesh",
           [](PM &self) { self.ParthenonInitPackagesAndMesh(); })
      .def(
          "ParthenonFinalize",
          [](PM &self) {
            ReleaseCallbacks(self);
            return self.ParthenonFinalize();
          },
          "Finalize Parthenon. Drivers and other objects holding simulation data must "
          "be deleted before calling this.")
      .def_prop_ro(
          "pinput", [](PM &self) { return self.pinput.get(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "pmesh", [](PM &self) { return self.pmesh.get(); },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "app_input", [](PM &self) { return self.app_input.get(); },
          nb::rv_policy::reference_internal);
}

} // namespace python
} // namespace parthenon

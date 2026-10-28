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

#ifndef RUNTIME_OPTIONS_HPP_
#define RUNTIME_OPTIONS_HPP_

#include <optional>
#include <string>
#include <vector>

namespace parthenon {

//! Runtime options that control a Parthenon run, independent of the input parameters.
//! Each member corresponds to a command line flag (see ArgParse), except for the input
//! file (-i) which is replaced by passing a ParameterInput directly to
//! ParthenonManager::ParthenonInitEnv.
struct RuntimeOptions {
  // -r <file> / -a <file>: restart (or analyze) from this file
  std::optional<std::string> restart_filename;
  // -a <file>: run analysis only on the restart file
  bool analysis = false;
  // -d <directory>: run directory
  std::optional<std::string> run_dir;
  // -p [regex]: report parameters for blocks matching regex and quit
  std::optional<std::string> params_regex;
  // -m <nproc>: output mesh structure and quit
  int mesh_flag = 0;
  // -t hh:mm:ss: wall time limit for final output, in seconds
  int wall_time_limit = 0;
  // -w <seconds>: watchdog timeout
  std::optional<int> watchdog_timeout;
  // Arguments forwarded to Kokkos::initialize when Parthenon initializes Kokkos
  std::vector<std::string> kokkos_args;

  bool IsRestart() const { return restart_filename.has_value(); }
};

} // namespace parthenon

#endif // RUNTIME_OPTIONS_HPP_

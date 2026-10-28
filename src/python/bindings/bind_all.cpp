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

#include "parthenon_nanobind.hpp"

namespace parthenon {
namespace python {

void BindAll(nb::module_ &m) {
  m.doc() = "Python bindings for the Parthenon performance portable AMR framework";

  // Order matters: later bindings refer to types registered by earlier ones in
  // their signatures.
  BindParameterInput(m);
  BindState(m);
  BindMesh(m);
  BindManager(m);
  BindDriver(m);
}

} // namespace python
} // namespace parthenon

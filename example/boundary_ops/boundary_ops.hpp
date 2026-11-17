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
#ifndef EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_
#define EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_

#include <cstddef>
#include <memory>

#include <bvals/comms/bnd_ops.hpp>
#include <parthenon/package.hpp>

namespace boundary_ops {
using namespace parthenon::package::prelude;

// Every block fills its interior with this value of its gid. The values are not exactly
// representable, so a sum over several neighbors depends on the summation order.
KOKKOS_INLINE_FUNCTION Real BlockValue(int gid) { return 1.0 / (gid + 3.0); }

// Registered unpack op: each ghost region gets the mean of the buffers of all the
// neighbors that map onto it, summed in ascending sender gid. On a regular mesh every
// region has one sender and this is the default copy. At a node of a 2D forest with
// valence 5 or more, several node neighbors share one corner ghost region; the default
// copy would keep whichever buffer is unpacked last.
struct AverageOverlappingGhosts {
  static void Unpack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                     const parthenon::BndInfoArrHost_t &info_h,
                     const parthenon::BndOpIdx_t &idx,
                     const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus SetBlockValues(MeshData<Real> *md);
// Checks every ghost region of every local block against the mean of its neighbors'
// values. Returns the global number of wrong cells; reports the global number of
// regions that averaged two or more neighbors in naveraged.
int CheckGhosts(Mesh *pmesh, int *naveraged);

} // namespace boundary_ops

#endif // EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_

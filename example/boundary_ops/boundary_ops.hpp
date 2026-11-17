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
#include <cstdint>
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

// Every block sets every face (and edge) flux component c to this value of its gid.
KOKKOS_INLINE_FUNCTION Real FluxValue(int gid, int c) { return (c + 1.0) / (gid + 3.0); }

// Registered flux-correction combine op: every element shared by blocks on the same
// level (a face of the face flux, a face or node of the 2D edge flux) gets the mean of
// the values of all blocks sharing it, summed in ascending gid, so every block holds
// the same bits (up to the sign of the frame). Shared faces have two blocks, nodes of
// the edge flux two to six.
struct AverageSharedFluxes {
  static void Combine(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                      const parthenon::BndInfoArrHost_t &info_h,
                      const parthenon::BndOpIdx_t &idx,
                      const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus SetBlockValues(MeshData<Real> *md);
TaskStatus SetFluxes(MeshData<Real> *md);
// Checks every ghost region of every local block against the mean of its neighbors'
// values. Returns the global number of wrong cells; reports the global number of
// regions that averaged two or more neighbors in naveraged.
int CheckGhosts(Mesh *pmesh, int *naveraged);
// Checks every same-level shared flux element of every local block against the mean
// of the values of the blocks sharing it, in ascending gid. Returns the global number of
// wrong elements; counts the shared elements by number of blocks sharing them in
// nshared (index 2..kMaxShared-1) and returns a hash of all fluxes, in gid order, in
// hash.
constexpr int kMaxShared = 16;
int CheckFluxes(Mesh *pmesh, int nshared[kMaxShared], std::uint64_t *hash);

} // namespace boundary_ops

#endif // EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_

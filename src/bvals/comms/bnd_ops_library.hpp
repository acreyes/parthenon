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

#ifndef BVALS_COMMS_BND_OPS_LIBRARY_HPP_
#define BVALS_COMMS_BND_OPS_LIBRARY_HPP_

#include <cstddef>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bnd_ops.hpp"
#include "kokkos_abstraction.hpp"

// Ready-made registered ops (see Metadata::RegisterBoundaryOps). They run on host and
// launch their kernels in DevExecSpace().

namespace parthenon {
namespace boundary_ops {

namespace impl {
// Splits a sorted bucket into groups of consecutive entries with the same local block,
// variable and neighbor offset, i.e. the buffers that land on one ghost region (or one
// shared flux-correction region). Returns the group starts, followed by nbuffers.
inline ParArray1D<int> GroupByRegion(const BndInfoArrHost_t &info_h,
                                     const BndOpIdxHost_t &idx_h, std::size_t nbuffers,
                                     int *ngroups) {
  std::vector<int> starts{0};
  for (std::size_t n = 1; n < nbuffers; ++n) {
    const BndInfo &prev = info_h(idx_h(n - 1));
    const BndInfo &cur = info_h(idx_h(n));
    if (cur.block != prev.block || cur.var.data() != prev.var.data() ||
        cur.offsets.GetIdx() != prev.offsets.GetIdx())
      starts.push_back(static_cast<int>(n));
  }
  starts.push_back(static_cast<int>(nbuffers));
  *ngroups = static_cast<int>(starts.size()) - 1;
  ParArray1D<int> group_start("bnd_ops_group_start", starts.size());
  auto group_start_h = Kokkos::create_mirror_view(group_start);
  for (std::size_t g = 0; g < starts.size(); ++g)
    group_start_h(g) = starts[g];
  Kokkos::deep_copy(group_start, group_start_h);
  return group_start;
}

// Calls f(var_element, buffer_value) for every active element that buffer bi maps onto,
// over the threads of team, as the default unpack does: the buffer value carries the
// coordinate-transformation sign, and an unallocated (sparse) buffer gives the variable's
// sparse default value.
template <class F>
KOKKOS_INLINE_FUNCTION void ForEachBufferElement(team_mbr_t team, const BndInfo &bi,
                                                 const F &f) {
  int idx_offset = 0;
  for (int it = 0; it < bi.ntopological_elements; ++it) {
    const auto &idxer = bi.idxer[it];
    const auto [tel, ftemp] = bi.lcoord_trans.InverseTransform(bi.topo_idx[it]);
    const int iel = static_cast<int>(tel) % 3;
    const Real fac = ftemp;
    const int offset = idx_offset;
    Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team, static_cast<int>(idxer.size())),
                         [&](const int m) {
                           const auto [t, u, v, k, j, i] = idxer(m);
                           const auto [il, jl, kl] =
                               bi.lcoord_trans.InverseTransform({i, j, k});
                           if (!idxer.IsActive(kl, jl, il)) return;
                           const Real val = bi.buf_allocated ? fac * bi.buf(m + offset)
                                                             : bi.var.sparse_default_val;
                           f(bi.var(iel, t, u, v, kl, jl, il), val);
                         });
    idx_offset += idxer.size();
  }
}
} // namespace impl

// Unpack op: every ghost element gets the mean of all same-level buffers that land on
// it, summed in ascending sender gid, so the result is bitwise independent of the
// number of ranks and of the order in which buffers arrive. Where one neighbor fills a
// ghost region (any regular mesh) this is the default copy. Where several neighbors map
// onto one region, e.g. the node neighbors of a node of valence other than four in a 2D
// forest, the default copy would keep whichever buffer is unpacked last.
// Register with m.RegisterBoundaryOps<boundary_ops::AverageOverlapping>().
struct AverageOverlapping {
  static void Unpack(MeshData<Real> *md, const BndInfoArr_t &info,
                     const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                     const BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
    if (nbuffers == 0) return;
    int nregions = 0;
    auto region_start = impl::GroupByRegion(info_h, idx_h, nbuffers, &nregions);
    // One team per ghost region: zero it, add the buffers one at a time in bucket
    // order, then divide by the number of buffers.
    Kokkos::parallel_for(
        "boundary_ops::AverageOverlapping",
        Kokkos::TeamPolicy<>(DevExecSpace(), nregions, Kokkos::AUTO),
        KOKKOS_LAMBDA(team_mbr_t team) {
          const int first = region_start(team.league_rank());
          const int last = region_start(team.league_rank() + 1);
          // All buffers of a region share the receiving variable
          if (!info(idx(first)).allocated) return;
          const Real count = last - first;
          for (int n = first; n < last; ++n) {
            impl::ForEachBufferElement(team, info(idx(n)),
                                       [](Real &ghost, const Real) { ghost = 0.0; });
          }
          team.team_barrier();
          for (int n = first; n < last; ++n) {
            impl::ForEachBufferElement(team, info(idx(n)),
                                       [](Real &ghost, const Real val) { ghost += val; });
            team.team_barrier();
          }
          impl::ForEachBufferElement(team, info(idx(first)),
                                     [=](Real &ghost, const Real) { ghost /= count; });
        });
  }
};

} // namespace boundary_ops
} // namespace parthenon

#endif // BVALS_COMMS_BND_OPS_LIBRARY_HPP_

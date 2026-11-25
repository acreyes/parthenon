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

#include <array>
#include <cstddef>
#include <tuple>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bnd_ops.hpp"
#include "kokkos_abstraction.hpp"

// Ready-made registered ops (see Metadata::RegisterBoundaryOps and
// Metadata::RegisterFluxCorrectionOps). They run on host and launch their kernels in
// DevExecSpace().

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

namespace flux_correction_ops {

namespace impl {
// Position in the buffer of bi of the receiving block's element (iel, t, u, v, k, j, i)
// and the sign between the two frames, or -1 if the buffer does not hold it.
KOKKOS_INLINE_FUNCTION int FindInBuffer(const BndInfo &bi, int iel, int t, int u, int v,
                                        int k, int j, int i, Real *fac) {
  const auto ijk = bi.lcoord_trans.Transform(std::array<int, 3>{i, j, k});
  int offset = 0;
  for (int it = 0; it < bi.ntopological_elements; ++it) {
    const auto &idxer = bi.idxer[it];
    const auto [tel, f] = bi.lcoord_trans.InverseTransform(bi.topo_idx[it]);
    const int ind[6] = {t, u, v, ijk[2], ijk[1], ijk[0]};
    const int s[6] = {idxer.StartIdx<0>(), idxer.StartIdx<1>(), idxer.StartIdx<2>(),
                      idxer.StartIdx<3>(), idxer.StartIdx<4>(), idxer.StartIdx<5>()};
    const int e[6] = {idxer.EndIdx<0>(), idxer.EndIdx<1>(), idxer.EndIdx<2>(),
                      idxer.EndIdx<3>(), idxer.EndIdx<4>(), idxer.EndIdx<5>()};
    bool inside = static_cast<int>(tel) % 3 == iel && idxer.size() > 0;
    int flat = 0;
    for (int d = 0; d < 6; ++d) {
      inside = inside && ind[d] >= s[d] && ind[d] <= e[d];
      flat = flat * (e[d] - s[d] + 1) + (ind[d] - s[d]);
    }
    if (inside) {
      *fac = f;
      return offset + flat;
    }
    offset += idxer.size();
  }
  return -1;
}
} // namespace impl

// Combine op: every flux element shared with same-level neighbors (the shared face of a
// face flux; shared faces and edges of an edge flux; shared faces, edges and nodes of a
// node flux) gets the mean of the local value and the values of all neighbors sharing
// it, summed in ascending gid. Every block then holds the same bits (up to the sign of
// its frame), independent of the number of ranks. Up to kMaxShared blocks may share an
// element (the valence of a node of a 2D forest, for example). Unallocated (sparse)
// neighbor buffers are skipped. Register with
// m.RegisterFluxCorrectionOps<flux_correction_ops::AverageShared>().
struct AverageShared {
  static constexpr int kMaxShared = 16;
  static void Combine(MeshData<Real> *md, const BndInfoArr_t &info,
                      const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                      const BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
    if (nbuffers == 0) return;
    // The bucket is sorted by (block, variable, ...), so the buffers sent to one
    // variable of one block are consecutive: [first(n), last(n)) for entry n.
    ParArray1D<int> first("AverageShared_first", nbuffers);
    ParArray1D<int> last("AverageShared_last", nbuffers);
    auto first_h = Kokkos::create_mirror_view(first);
    auto last_h = Kokkos::create_mirror_view(last);
    for (std::size_t n = 0; n < nbuffers; ++n) {
      const BndInfo &cur = info_h(idx_h(n));
      const bool same = n > 0 && cur.block == info_h(idx_h(n - 1)).block &&
                        cur.var.data() == info_h(idx_h(n - 1)).var.data();
      first_h(n) = same ? first_h(n - 1) : static_cast<int>(n);
    }
    for (int n = static_cast<int>(nbuffers) - 1; n >= 0; --n)
      last_h(n) = (n + 1 < static_cast<int>(nbuffers) && first_h(n + 1) == first_h(n))
                      ? last_h(n + 1)
                      : n + 1;
    Kokkos::deep_copy(first, first_h);
    Kokkos::deep_copy(last, last_h);

    // One team per buffer. Each element is combined by the first buffer of its group
    // that holds it: it collects the local value and the value of every buffer holding
    // it, in ascending gid, and writes their mean.
    Kokkos::parallel_for(
        "flux_correction_ops::AverageShared",
        Kokkos::TeamPolicy<>(DevExecSpace(), nbuffers, Kokkos::AUTO),
        KOKKOS_LAMBDA(team_mbr_t team) {
          const int n = team.league_rank();
          const BndInfo &bi = info(idx(n));
          if (!bi.allocated) return;
          for (int it = 0; it < bi.ntopological_elements; ++it) {
            const auto &idxer = bi.idxer[it];
            const int iel = static_cast<int>(std::get<0>(
                                bi.lcoord_trans.InverseTransform(bi.topo_idx[it]))) %
                            3;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange<>(team, static_cast<int>(idxer.size())),
                [&](const int m) {
                  const auto [t, u, v, k, j, i] = idxer(m);
                  const auto [il, jl, kl] = bi.lcoord_trans.InverseTransform({i, j, k});
                  Real fac;
                  for (int n2 = first(n); n2 < n; ++n2)
                    if (impl::FindInBuffer(info(idx(n2)), iel, t, u, v, kl, jl, il,
                                           &fac) >= 0)
                      return;
                  int gid[kMaxShared];
                  Real val[kMaxShared];
                  int count = 0;
                  auto insert = [&](int g, Real x) {
                    if (count == kMaxShared)
                      Kokkos::abort("AverageShared: too many blocks share an element");
                    int c = count++;
                    for (;
                         c > 0 && (gid[c - 1] > g || (gid[c - 1] == g && val[c - 1] > x));
                         --c) {
                      gid[c] = gid[c - 1];
                      val[c] = val[c - 1];
                    }
                    gid[c] = g;
                    val[c] = x;
                  };
                  Real &own = bi.var(iel, t, u, v, kl, jl, il);
                  insert(bi.recv_gid, own);
                  for (int n2 = first(n); n2 < last(n); ++n2) {
                    const BndInfo &b2 = info(idx(n2));
                    if (!b2.buf_allocated) continue;
                    const int p = impl::FindInBuffer(b2, iel, t, u, v, kl, jl, il, &fac);
                    if (p >= 0) insert(b2.send_gid, fac * b2.buf(p));
                  }
                  Real sum = 0.0;
                  for (int c = 0; c < count; ++c)
                    sum += val[c];
                  own = sum / count;
                });
          }
        });
  }
};

} // namespace flux_correction_ops
} // namespace parthenon

#endif // BVALS_COMMS_BND_OPS_LIBRARY_HPP_

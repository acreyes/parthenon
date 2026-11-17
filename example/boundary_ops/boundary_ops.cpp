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

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "boundary_ops.hpp"

namespace boundary_ops {
using parthenon::BndInfo;

void AverageOverlappingGhosts::Unpack(MeshData<Real> *md,
                                      const parthenon::BndInfoArr_t &info,
                                      const parthenon::BndInfoArrHost_t &info_h,
                                      const parthenon::BndOpIdx_t &idx,
                                      const parthenon::BndOpIdxHost_t &idx_h,
                                      std::size_t nbuffers) {
  // The bucket is sorted by (block, variable, neighbor offset, neighbor gid), so the
  // buffers that fill one ghost region are consecutive and in ascending sender gid. This
  // example registers the op for one variable only, so (block, offset) is the region.
  std::vector<int> starts{0};
  for (std::size_t n = 1; n < nbuffers; ++n) {
    const BndInfo &prev = info_h(idx_h(n - 1));
    const BndInfo &cur = info_h(idx_h(n));
    if (cur.block != prev.block || cur.offsets.GetIdx() != prev.offsets.GetIdx())
      starts.push_back(n);
  }
  starts.push_back(nbuffers);
  const int nregions = starts.size() - 1;
  parthenon::ParArray1D<int> region_start("region_start", starts.size());
  auto region_start_h = Kokkos::create_mirror_view(region_start);
  for (std::size_t r = 0; r < starts.size(); ++r)
    region_start_h(r) = starts[r];
  Kokkos::deep_copy(region_start, region_start_h);

  // One team per ghost region: the first buffer overwrites the region, the next ones add
  // to it in order, the last one divides by the count.
  Kokkos::parallel_for(
      "AverageOverlappingGhosts",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nregions, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team) {
        const int first = region_start(team.league_rank());
        const int last = region_start(team.league_rank() + 1);
        for (int n = first; n < last; ++n) {
          auto &bi = info(idx(n));
          if (!bi.allocated || !bi.buf_allocated) continue; // not sparse in this example
          const auto &idxer = bi.idxer[0];
          const auto [tel, fac] = bi.lcoord_trans.InverseTransform(bi.topo_idx[0]);
          const int iel = static_cast<int>(tel) % 3;
          const Real count = last - first;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team, idxer.size()), [&](const int m) {
                const auto [t, u, v, k, j, i] = idxer(m);
                const auto [il, jl, kl] = bi.lcoord_trans.InverseTransform({i, j, k});
                if (!idxer.IsActive(kl, jl, il)) return;
                Real &ghost = bi.var(iel, t, u, v, kl, jl, il);
                ghost = (n == first) ? bi.buf(m) : ghost + bi.buf(m);
                if (n == last - 1) ghost /= count;
              });
          team.team_barrier();
        }
      });
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("boundary_ops");
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  m.RegisterBoundaryOps<AverageOverlappingGhosts>();
  pkg->AddField("ghost_avg", m);
  return pkg;
}

TaskStatus SetBlockValues(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    auto q = rc->Get("ghost_avg").data;
    const Real val = BlockValue(rc->GetBlockPointer()->gid);
    IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
    IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
    // Ghost zones that no neighbor fills keep -1
    Kokkos::deep_copy(q, -1.0);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) { q(k, j, i) = val; });
  }
  return TaskStatus::complete;
}

int CheckGhosts(Mesh *pmesh, int *naveraged) {
  int counts[2] = {0, 0}; // wrong cells, averaged regions
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    auto q = rc->Get("ghost_avg").data.GetHostMirrorAndCopy();
    // Neighbors filling each ghost region, keyed by offset index, in ascending gid
    std::map<int, std::vector<int>> senders;
    for (const auto &nb : pmb->GetNeighbors())
      senders[nb.offsets.GetIdx()].push_back(nb.gid);
    const IndexRange bnds[3] = {rc->GetBoundsI(IndexDomain::interior),
                                rc->GetBoundsJ(IndexDomain::interior),
                                rc->GetBoundsK(IndexDomain::interior)};
    const int ng = parthenon::Globals::nghost;
    for (auto &[loc, gids] : senders) {
      std::sort(gids.begin(), gids.end());
      Real expected = 0.0;
      for (const int gid : gids)
        expected += BlockValue(gid);
      expected /= gids.size();
      if (gids.size() > 1) counts[1]++;
      // Ghost cell range of this region in each direction
      int s[3], e[3];
      for (int d = 0; d < 3; ++d) {
        const int o = (d == 0) ? loc % 3 - 1 : (d == 1) ? (loc / 3) % 3 - 1 : loc / 9 - 1;
        s[d] = (o < 0) ? bnds[d].s - ng : (o > 0) ? bnds[d].e + 1 : bnds[d].s;
        e[d] = (o < 0) ? bnds[d].s - 1 : (o > 0) ? bnds[d].e + ng : bnds[d].e;
      }
      for (int k = s[2]; k <= e[2]; ++k)
        for (int j = s[1]; j <= e[1]; ++j)
          for (int i = s[0]; i <= e[0]; ++i)
            if (q(k, j, i) != expected) counts[0]++;
    }
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
#endif
  *naveraged = counts[1];
  return counts[0];
}

} // namespace boundary_ops

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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <bvals/comms/bnd_ops_library.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "boundary_ops.hpp"

namespace boundary_ops {
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("boundary_ops");
  // Each ghost region gets the mean of the buffers of all the neighbors that map onto
  // it, summed in ascending sender gid. At a node of a 2D forest with valence 5 or more,
  // several node neighbors share one corner ghost region; the default copy would keep
  // whichever buffer is unpacked last.
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  m.RegisterBoundaryOps<parthenon::boundary_ops::AverageOverlapping>();
  pkg->AddField("ghost_avg", m);
  // Members of a sparse pool inherit the registered ops of the pool's metadata
  Metadata ms(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::Sparse});
  ms.RegisterBoundaryOps<parthenon::boundary_ops::AverageOverlapping>();
  pkg->AddSparsePool("ghost_avg_sparse", ms, std::vector<int>{0});
  return pkg;
}

const std::vector<std::string> kVariables{"ghost_avg", "ghost_avg_sparse_0"};

void AllocateSparse(Mesh *pmesh) {
  for (auto &pmb : pmesh->block_list)
    pmb->AllocateSparse("ghost_avg_sparse_0");
}

TaskStatus SetBlockValues(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    const Real val = BlockValue(rc->GetBlockPointer()->gid);
    IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
    IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
    for (const auto &name : kVariables) {
      auto q = rc->Get(name).data;
      // Ghost zones that no neighbor fills keep -1
      Kokkos::deep_copy(q, -1.0);
      parthenon::par_for(
          PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) { q(k, j, i) = val; });
    }
  }
  return TaskStatus::complete;
}

int CheckGhosts(Mesh *pmesh, int *naveraged) {
  int counts[2] = {0, 0}; // wrong cells, averaged regions
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
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
      for (const auto &name : kVariables) {
        auto q = rc->Get(name).data.GetHostMirrorAndCopy();
        for (int k = s[2]; k <= e[2]; ++k)
          for (int j = s[1]; j <= e[1]; ++j)
            for (int i = s[0]; i <= e[0]; ++i)
              if (q(k, j, i) != expected) counts[0]++;
      }
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

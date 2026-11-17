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
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <bvals/comms/bnd_ops_library.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "boundary_ops.hpp"

namespace boundary_ops {
using parthenon::BndInfo;

namespace {
constexpr const char *flux_vars[3] = {"flux_avg", "emf_avg", "flux_avg_sparse_0"};
std::string FluxLabel(const std::string &base) {
  return parthenon::internal_fluxname + parthenon::internal_varname_seperator + base;
}
} // namespace

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

  // A face flux (of a cell variable) and an edge flux (of a face variable) whose shared
  // elements are averaged over the blocks sharing them
  Metadata mc({Metadata::Cell, Metadata::WithFluxes});
  mc.RegisterFluxCorrectionOps<parthenon::flux_correction_ops::AverageShared>();
  pkg->AddField(flux_vars[0], mc);
  Metadata mf({Metadata::Face, Metadata::WithFluxes});
  mf.RegisterFluxCorrectionOps<parthenon::flux_correction_ops::AverageShared>();
  pkg->AddField(flux_vars[1], mf);
  Metadata mcs({Metadata::Cell, Metadata::WithFluxes, Metadata::Sparse});
  mcs.RegisterFluxCorrectionOps<parthenon::flux_correction_ops::AverageShared>();
  pkg->AddSparsePool("flux_avg_sparse", mcs, std::vector<int>{0});
  return pkg;
}

const std::vector<std::string> kVariables{"ghost_avg", "ghost_avg_sparse_0"};

void AllocateSparse(Mesh *pmesh) {
  for (auto &pmb : pmesh->block_list) {
    pmb->AllocateSparse("ghost_avg_sparse_0");
    pmb->AllocateSparse("flux_avg_sparse_0");
  }
}

TaskStatus SetFluxes(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    const int gid = rc->GetBlockPointer()->gid;
    for (const auto &base : flux_vars) {
      auto f = rc->Get(FluxLabel(base)).data.Get();
      for (int c = 0; c < f.extent_int(0); ++c) {
        auto fc = Kokkos::subview(f, c, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(),
                                  Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
        Kokkos::deep_copy(fc, FluxValue(gid, c));
      }
    }
  }
  return TaskStatus::complete;
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

int CheckFluxes(Mesh *pmesh, int nshared[kMaxShared], std::uint64_t *hash) {
  constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
  constexpr std::uint64_t fnv_prime = 1099511628211ULL;
  auto fnv = [&](std::uint64_t h, std::uint64_t x) { return (h ^ x) * fnv_prime; };

  int nwrong = 0;
  for (int c = 0; c < kMaxShared; ++c)
    nshared[c] = 0;
  std::vector<std::uint64_t> block_hash(pmesh->nbtotal, 0);
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    std::uint64_t h = fnv_offset;
    for (const auto &base : flux_vars) {
      auto v = rc->GetVarPtr(FluxLabel(base));
      auto q = v->data.GetHostMirrorAndCopy();
      // Values of all the blocks sharing each element, keyed by (component, k, j, i)
      std::map<std::array<int, 4>, std::vector<std::pair<int, Real>>> shared;
      for (const auto &nb : pmb->GetNeighbors()) {
        if (nb.loc.level() != pmb->loc.level()) continue;
        // Face fluxes are shared with face neighbors, edge fluxes also with edge ones
        if (v->IsSet(Metadata::Face) ? !nb.offsets.IsFace() : nb.offsets.IsNode())
          continue;
        auto bi = BndInfo::GetSetBndInfo(pmb.get(), nb, v, nullptr);
        for (int it = 0; it < bi.ntopological_elements; ++it) {
          const auto &idxer = bi.idxer[it];
          const auto [tel, fac] = bi.lcoord_trans.InverseTransform(bi.topo_idx[it]);
          const Real x = fac * FluxValue(nb.gid, static_cast<int>(bi.topo_idx[it]) % 3);
          for (int m = 0; m < idxer.size(); ++m) {
            const auto [t, u, w, k, j, i] = idxer(m);
            const auto [il, jl, kl] = bi.lcoord_trans.InverseTransform({i, j, k});
            shared[{static_cast<int>(tel) % 3, kl, jl, il}].emplace_back(nb.gid, x);
          }
        }
      }
      for (auto &[e, vals] : shared) {
        vals.emplace_back(pmb->gid, FluxValue(pmb->gid, e[0]));
        std::sort(vals.begin(), vals.end());
        Real sum = 0.0;
        for (const auto &gx : vals)
          sum += gx.second;
        nshared[std::min<int>(vals.size(), kMaxShared - 1)]++;
        if (q(e[0], 0, 0, 0, e[1], e[2], e[3]) != sum / vals.size()) nwrong++;
      }
      auto qv = q.Get();
      for (std::size_t n = 0; n < qv.size(); ++n) {
        std::uint64_t bits;
        std::memcpy(&bits, qv.data() + n, sizeof(bits));
        h = fnv(h, bits);
      }
    }
    block_hash[pmb->gid] = h;
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &nwrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, nshared, kMaxShared, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, block_hash.data(), block_hash.size(),
                                    MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD));
#endif
  *hash = fnv_offset;
  for (const auto bh : block_hash)
    *hash = fnv(*hash, bh);
  return nwrong;
}

} // namespace boundary_ops

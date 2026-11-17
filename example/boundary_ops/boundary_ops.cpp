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

namespace {
constexpr const char *flux_vars[2] = {"flux_avg", "emf_avg"};
std::string FluxLabel(const std::string &base) {
  return parthenon::internal_fluxname + parthenon::internal_varname_seperator + base;
}

// Position in the buffer of bi of the receiving block's element (iel, t, u, v, k, j, i)
// and the sign between the two frames, or -1 if the buffer does not hold it.
KOKKOS_INLINE_FUNCTION int FindInBuffer(const BndInfo &bi, int iel, int t, int u, int v,
                                        int k, int j, int i, Real *fac) {
  const auto ijk =
      bi.lcoord_trans.Transform(std::array<int, 3>{i, j, k}); // sender's frame
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
} // namespace

void AverageSharedFluxes::Combine(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                                  const parthenon::BndInfoArrHost_t &info_h,
                                  const parthenon::BndOpIdx_t &idx,
                                  const parthenon::BndOpIdxHost_t &idx_h,
                                  std::size_t nbuffers) {
  // The bucket is sorted by (block, variable, ...), so the buffers sent to one variable
  // of one block are consecutive: [first(n), last(n)) for entry n.
  parthenon::ParArray1D<int> first("first", nbuffers), last("last", nbuffers);
  auto first_h = Kokkos::create_mirror_view(first);
  auto last_h = Kokkos::create_mirror_view(last);
  for (std::size_t n = 0; n < nbuffers; ++n) {
    const BndInfo &cur = info_h(idx_h(n));
    const bool same = n > 0 && cur.block == info_h(idx_h(n - 1)).block &&
                      cur.var.data() == info_h(idx_h(n - 1)).var.data();
    first_h(n) = same ? first_h(n - 1) : n;
  }
  for (int n = static_cast<int>(nbuffers) - 1; n >= 0; --n)
    last_h(n) = (n + 1 < static_cast<int>(nbuffers) && first_h(n + 1) == first_h(n))
                    ? last_h(n + 1)
                    : n + 1;
  Kokkos::deep_copy(first, first_h);
  Kokkos::deep_copy(last, last_h);

  // One team per buffer. Each element is combined by the first buffer of its group that
  // holds it: it collects its own value and the value of every buffer holding it, in
  // ascending gid, and writes their mean.
  Kokkos::parallel_for(
      "AverageSharedFluxes",
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbuffers, Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team) {
        const int n = team.league_rank();
        const BndInfo &bi = info(idx(n));
        for (int it = 0; it < bi.ntopological_elements; ++it) {
          const auto &idxer = bi.idxer[it];
          const int iel = static_cast<int>(std::get<0>(
                              bi.lcoord_trans.InverseTransform(bi.topo_idx[it]))) %
                          3;
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange<>(team, idxer.size()), [&](const int m) {
                const auto [t, u, v, k, j, i] = idxer(m);
                const auto [il, jl, kl] = bi.lcoord_trans.InverseTransform({i, j, k});
                Real fac;
                for (int n2 = first(n); n2 < n; ++n2)
                  if (FindInBuffer(info(idx(n2)), iel, t, u, v, kl, jl, il, &fac) >= 0)
                    return;
                int gid[kMaxShared];
                Real val[kMaxShared];
                int count = 0;
                auto insert = [&](int g, Real x) {
                  if (count == kMaxShared)
                    Kokkos::abort("Too many blocks share an element");
                  int c = count++;
                  for (; c > 0 && (gid[c - 1] > g || (gid[c - 1] == g && val[c - 1] > x));
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
                  const int p = FindInBuffer(b2, iel, t, u, v, kl, jl, il, &fac);
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

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("boundary_ops");
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost});
  m.RegisterBoundaryOps<AverageOverlappingGhosts>();
  pkg->AddField("ghost_avg", m);

  // A face flux (of a cell variable) and an edge flux (of a face variable) whose shared
  // elements are averaged over the blocks sharing them
  Metadata mc({Metadata::Cell, Metadata::WithFluxes});
  mc.RegisterFluxCorrectionOps<AverageSharedFluxes>();
  pkg->AddField(flux_vars[0], mc);
  Metadata mf({Metadata::Face, Metadata::WithFluxes});
  mf.RegisterFluxCorrectionOps<AverageSharedFluxes>();
  pkg->AddField(flux_vars[1], mf);
  return pkg;
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

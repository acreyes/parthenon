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
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

#include "provider_lanes.hpp"

namespace provider_lanes {
using parthenon::BndInfo;

namespace {
std::uint64_t SplitMix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::vector<int> Shuffle(int n, std::uint64_t seed) {
  std::vector<int> p(n);
  std::iota(p.begin(), p.end(), 0);
  for (int i = n - 1; i > 0; --i) {
    seed = SplitMix64(seed);
    std::swap(p[i], p[seed % (i + 1)]);
  }
  return p;
}

std::string FluxLabel(const std::string &base) {
  return parthenon::internal_fluxname + parthenon::internal_varname_seperator + base;
}

// Starts of the consecutive runs of a bucket with the same (block, variable, channel),
// plus the end. The bucket is sorted by (block, variable, channel, neighbor gid), so
// each run holds the buffers of one channel in ascending sender gid.
parthenon::ParArray1D<int> ChannelRuns(const parthenon::BndInfoArrHost_t &info_h,
                                       const parthenon::BndOpIdxHost_t &idx_h,
                                       std::size_t nbuffers, int *nruns) {
  std::vector<int> starts{0};
  for (std::size_t n = 1; n < nbuffers; ++n) {
    const BndInfo &prev = info_h(idx_h(n - 1));
    const BndInfo &cur = info_h(idx_h(n));
    if (cur.block != prev.block || cur.var.data() != prev.var.data() ||
        cur.channel != prev.channel)
      starts.push_back(n);
  }
  starts.push_back(nbuffers);
  *nruns = starts.size() - 1;
  parthenon::ParArray1D<int> runs("channel_runs", starts.size());
  auto runs_h = Kokkos::create_mirror_view(runs);
  for (std::size_t r = 0; r < starts.size(); ++r)
    runs_h(r) = starts[r];
  Kokkos::deep_copy(runs, runs_h);
  return runs;
}

struct FluxIndex {
  int k, j, i;
};
FluxIndex CandidateIndex(MeshData<Real> *md) {
  auto &rc = md->GetBlockData(0);
  return {rc->GetBoundsK(IndexDomain::interior).s,
          rc->GetBoundsJ(IndexDomain::interior).s,
          rc->GetBoundsI(IndexDomain::interior).s};
}
} // namespace

LaneGraph::LaneGraph(int nblocks, int version) : nblocks(nblocks), version(version) {
  group.assign(kLayers, std::vector<int>(nblocks));
  members.resize(kLayers);
  for (int layer = 0; layer < kLayers; ++layer) {
    const auto perm = Shuffle(nblocks, 1000 * version + layer + 1);
    int start = 0, k = (layer + version) % (kMaxGroup - 1);
    while (start < nblocks) {
      const int left = nblocks - start;
      int size = std::min(left, 2 + (k++) % (kMaxGroup - 1));
      if (left - size == 1) size += size < kMaxGroup ? 1 : -1;
      std::vector<int> g(perm.begin() + start, perm.begin() + start + size);
      std::sort(g.begin(), g.end());
      for (int gid : g)
        group[layer][gid] = members[layer].size();
      members[layer].push_back(g);
      start += size;
    }
  }
  ring = Shuffle(nblocks, 4242);
  pos.resize(nblocks);
  for (int p = 0; p < nblocks; ++p)
    pos[ring[p]] = p;
}

void LaneGraph::GroupNeighbors(int gid, std::vector<parthenon::LaneNeighbor> &out) const {
  for (int layer = 0; layer < kLayers; ++layer)
    for (int m : members[layer][group[layer][gid]])
      if (m != gid) out.push_back({m, Channel(layer, gid), Channel(layer, m)});
}

void LaneGraph::RingNeighbors(int gid, std::vector<parthenon::LaneNeighbor> &out) const {
  const int p = pos[gid];
  out.push_back({ring[(p + nblocks - 1) % nblocks], 0, 1});
  out.push_back({ring[(p + 1) % nblocks], 1, 0});
}

void RecordPack::Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                      const parthenon::BndInfoArrHost_t &info_h,
                      const parthenon::BndOpIdx_t &idx,
                      const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
  Kokkos::parallel_for(
      "RecordPack", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nbuffers),
      KOKKOS_LAMBDA(const int n) {
        const BndInfo &bi = info(idx(n));
        bi.buf(0) = bi.var(0, 0, 0, 0, 0, 0, 0);
      });
}

void SumByChannel::Unpack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                          const parthenon::BndInfoArrHost_t &info_h,
                          const parthenon::BndOpIdx_t &idx,
                          const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
  int nruns;
  auto runs = ChannelRuns(info_h, idx_h, nbuffers, &nruns);
  Kokkos::parallel_for(
      "SumByChannel", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nruns),
      KOKKOS_LAMBDA(const int r) {
        Real sum = 0.0;
        for (int n = runs(r); n < runs(r + 1); ++n)
          sum += info(idx(n)).buf(0);
        const BndInfo &bi = info(idx(runs(r)));
        bi.var(0, 0, 0, 0, 0, 0, 1 + bi.channel) = sum;
      });
}

void FluxCandidatePack::Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                             const parthenon::BndInfoArrHost_t &info_h,
                             const parthenon::BndOpIdx_t &idx,
                             const parthenon::BndOpIdxHost_t &idx_h,
                             std::size_t nbuffers) {
  const auto c = CandidateIndex(md);
  Kokkos::parallel_for(
      "FluxCandidatePack", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nbuffers),
      KOKKOS_LAMBDA(const int n) {
        const BndInfo &bi = info(idx(n));
        bi.buf(0) = bi.var(0, 0, 0, 0, c.k, c.j + bi.channel, c.i);
      });
}

void GroupMean::Combine(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                        const parthenon::BndInfoArrHost_t &info_h,
                        const parthenon::BndOpIdx_t &idx,
                        const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
  int nruns;
  auto runs = ChannelRuns(info_h, idx_h, nbuffers, &nruns);
  const auto c = CandidateIndex(md);
  Kokkos::parallel_for(
      "GroupMean", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nruns),
      KOKKOS_LAMBDA(const int r) {
        const BndInfo &bi = info(idx(runs(r)));
        Real &own = bi.var(0, 0, 0, 0, c.k, c.j + bi.channel, c.i);
        // The local value joins the senders' in ascending gid
        Real sum = 0.0;
        bool own_added = false;
        for (int n = runs(r); n < runs(r + 1); ++n) {
          const BndInfo &b = info(idx(n));
          if (!own_added && bi.recv_gid < b.send_gid) {
            sum += own;
            own_added = true;
          }
          sum += b.buf(0);
        }
        if (!own_added) sum += own;
        own = sum / (runs(r + 1) - runs(r) + 1);
      });
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("provider_lanes");
  // Per-block records: slot 0 is the block's own value, slot 1 + c what arrived on
  // channel c
  Metadata mg({Metadata::None, Metadata::Derived, Metadata::OneCopy},
              std::vector<int>{1 + kLayers});
  mg.SetLane(kGroups);
  mg.RegisterBoundaryOps<SumByChannel, RecordPack>();
  pkg->AddField("group_rec", mg);

  Metadata mr({Metadata::None, Metadata::Derived, Metadata::OneCopy},
              std::vector<int>{3});
  mr.SetLane(kRing);
  mr.RegisterBoundaryOps<SumByChannel, RecordPack>();
  pkg->AddField("ring_rec", mr);

  // A cell variable whose face flux is combined over the groups
  Metadata mq({Metadata::Cell, Metadata::WithFluxes});
  mq.SetLane(kGroups);
  mq.RegisterFluxCorrectionOps<GroupMean, FluxCandidatePack>();
  pkg->AddField("q", mq);

  // An ordinary ghost-exchanged variable, in the same MeshData as the lane variables
  Metadata mu(
      {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
  pkg->AddField("u", mu);
  return pkg;
}

TaskStatus SetValues(MeshData<Real> *md, int cycle) {
  const auto c = CandidateIndex(md);
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    const int gid = rc->GetBlockPointer()->gid;
    auto grec = rc->Get("group_rec").data.Get();
    auto rrec = rc->Get("ring_rec").data.Get();
    auto flux = rc->Get(FluxLabel("q")).data.Get();
    auto u = rc->Get("u").data.Get();
    Kokkos::deep_copy(grec, -1.0);
    Kokkos::deep_copy(rrec, -1.0);
    Kokkos::deep_copy(flux, 0.0);
    Kokkos::deep_copy(u, gid + 0.25 * cycle);
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, 0, KOKKOS_LAMBDA(const int) {
          grec(0, 0, 0, 0, 0, 0, 0) = RecordValue(gid, cycle);
          rrec(0, 0, 0, 0, 0, 0, 0) = RingValue(gid, cycle);
          for (int ch = 0; ch < kLayers; ++ch)
            flux(0, 0, 0, 0, c.k, c.j + ch, c.i) = FluxCandidate(gid, ch, cycle);
        });
  }
  return TaskStatus::complete;
}

CheckResult Check(Mesh *pmesh, const LaneGraph &graph, int cycle) {
  CheckResult res;
  auto check = [&res](Real got, Real expected, std::uint64_t tag) {
    std::uint64_t bits;
    std::memcpy(&bits, &got, sizeof(bits));
    res.hash += SplitMix64(bits ^ SplitMix64(tag));
    res.nchecked++;
    if (std::memcmp(&got, &expected, sizeof(Real)) != 0) res.nwrong++;
  };
  for (auto &pmb : pmesh->block_list) {
    auto &rc = pmb->meshblock_data.Get();
    const int gid = pmb->gid;
    auto grec = rc->Get("group_rec").data.GetHostMirrorAndCopy();
    auto rrec = rc->Get("ring_rec").data.GetHostMirrorAndCopy();
    auto flux = rc->Get(FluxLabel("q")).data.GetHostMirrorAndCopy();
    const int ks = rc->GetBoundsK(IndexDomain::interior).s;
    const int js = rc->GetBoundsJ(IndexDomain::interior).s;
    const int is = rc->GetBoundsI(IndexDomain::interior).s;
    const int ie = rc->GetBoundsI(IndexDomain::interior).e;
    const std::uint64_t tag = 16 * static_cast<std::uint64_t>(gid);

    // u: the x1 face ghosts hold the geometric neighbor's value
    auto u = rc->Get("u").data.GetHostMirrorAndCopy();
    for (const auto &nb : pmb->GetNeighbors()) {
      if (!nb.offsets.IsFace() || nb.offsets(parthenon::X1DIR) == 0) continue;
      const int i = nb.offsets(parthenon::X1DIR) < 0 ? is - 1 : ie + 1;
      check(u(0, 0, 0, 0, ks, js, i), nb.gid + 0.25 * cycle, tag + 10 + (i > is));
    }

    check(grec(0, 0, 0, 0, 0, 0, 0), RecordValue(gid, cycle), tag);
    for (int layer = 0; layer < kLayers; ++layer) {
      const auto &g = graph.members[layer][graph.group[layer][gid]];
      if (g.front() == gid) res.group_sizes[g.size()]++;
      const int ch = LaneGraph::Channel(layer, gid);
      // Sum of the other members' records and mean of all members' candidates, both
      // in ascending gid
      Real sum = 0.0, mean = 0.0;
      for (int m : g) {
        if (m != gid) sum += RecordValue(m, cycle);
        mean += FluxCandidate(m, LaneGraph::Channel(layer, m), cycle);
      }
      mean /= g.size();
      check(grec(0, 0, 0, 0, 0, 0, 1 + ch), sum, tag + 1 + ch);
      check(flux(0, 0, 0, 0, ks, js + ch, is), mean, tag + 4 + ch);
    }

    const int p = graph.pos[gid];
    const int n = graph.nblocks;
    check(rrec(0, 0, 0, 0, 0, 0, 0), RingValue(gid, cycle), tag + 7);
    check(rrec(0, 0, 0, 0, 0, 0, 1), RingValue(graph.ring[(p + n - 1) % n], cycle),
          tag + 8);
    check(rrec(0, 0, 0, 0, 0, 0, 2), RingValue(graph.ring[(p + 1) % n], cycle), tag + 9);
  }
#ifdef MPI_PARALLEL
  int counts[2 + kMaxGroup + 1] = {res.nwrong, res.nchecked};
  for (int s = 0; s <= kMaxGroup; ++s)
    counts[2 + s] = res.group_sizes[s];
  MPI_Allreduce(MPI_IN_PLACE, counts, 2 + kMaxGroup + 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  res.nwrong = counts[0];
  res.nchecked = counts[1];
  for (int s = 0; s <= kMaxGroup; ++s)
    res.group_sizes[s] = counts[2 + s];
  MPI_Allreduce(MPI_IN_PLACE, &res.hash, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  return res;
}

} // namespace provider_lanes

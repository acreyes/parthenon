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

//! \file mesh-lanes.cpp
//  \brief Provider lanes: neighbor lists, tag maps and buffers of user-defined exchanges

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/neighbor_block.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

void Mesh::RegisterNeighborProvider(LaneId lane, NeighborProvider_t provider) {
  PARTHENON_REQUIRE_THROWS(static_cast<int>(lane) >= 0, "Lane ids are non-negative.");
  PARTHENON_REQUIRE_THROWS(!do_coalesced_comms,
                           "Provider lanes do not support coalesced comms.");
  PARTHENON_REQUIRE_THROWS(static_cast<bool>(provider), "Empty neighbor provider.");
  lane_providers_[static_cast<int>(lane)] = std::move(provider);
  // Before the first BuildAndRegisterCommBuffers_ the epoch is 1; the lane is then
  // built with the other buffers
  if (boundary_comm_map.GetCurrentEpoch() > 1) RebuildLane(lane);
}

void Mesh::ListLaneNeighbors_(const MeshBlock &pmb, int lane,
                              std::vector<LaneNeighbor> &lnbs) {
  const int nbtot = static_cast<int>(ranklist.size());
  lnbs.clear();
  lane_providers_.at(lane)(pmb, lnbs);
  std::sort(lnbs.begin(), lnbs.end(), [](const auto &a, const auto &b) {
    return std::make_tuple(a.channel_me, a.gid, a.channel_nb) <
           std::make_tuple(b.channel_me, b.gid, b.channel_nb);
  });
  for (std::size_t n = 0; n < lnbs.size(); ++n) {
    const auto &l = lnbs[n];
    PARTHENON_REQUIRE(l.gid >= 0 && l.gid < nbtot,
                      "Lane neighbor gid " + std::to_string(l.gid) + " of block " +
                          std::to_string(pmb.gid) + " out of range.");
    PARTHENON_REQUIRE(l.channel_me >= 0 && l.channel_me < MaxLaneChannels &&
                          l.channel_nb >= 0 && l.channel_nb < MaxLaneChannels,
                      "Lane channel ids must be in [0, MaxLaneChannels).");
    PARTHENON_REQUIRE(n == 0 || l.gid != lnbs[n - 1].gid ||
                          l.channel_me != lnbs[n - 1].channel_me ||
                          l.channel_nb != lnbs[n - 1].channel_nb,
                      "Duplicate lane neighbor of block " + std::to_string(pmb.gid));
    // Lane ops run on same-level exchanges only, like registered boundary ops
    PARTHENON_REQUIRE(loclist[l.gid].level() == pmb.loc.level(),
                      "Lane neighbors must be on the same refinement level.");
  }
}

NeighborBlock Mesh::MakeLaneNeighbor_(const MeshBlock &pmb, const LaneNeighbor &l) {
  NeighborBlock nb(this, loclist[l.gid], loclist[l.gid], ranklist[l.gid], l.gid,
                   Kokkos::Array<int, 3>{0, 0, 0}, -1, -1, 0, 0, pmb.block_coarsenings);
  nb.lane_channel = l.channel_me;
  nb.lane_nb_channel = l.channel_nb;
  return nb;
}

void Mesh::SetLaneNeighbors_(int lane) {
  std::vector<LaneNeighbor> lnbs;
  for (auto &pmb : block_list) {
    ListLaneNeighbors_(*pmb, lane, lnbs);
    if (static_cast<int>(pmb->lane_neighbors.size()) <= lane)
      pmb->lane_neighbors.resize(lane + 1);
    auto &nbs = pmb->lane_neighbors[lane];
    nbs.clear();
    nbs.reserve(lnbs.size());
    for (const auto &l : lnbs)
      nbs.push_back(MakeLaneNeighbor_(*pmb, l));
  }
}

int Mesh::LaneCommLanes_() {
  for (auto btype : {BoundaryType::lane, BoundaryType::lane_flxcor_send})
    LockCommLaneNumbers(btype);
  return std::max(GetNumberOfCommLanes(BoundaryType::lane),
                  GetNumberOfCommLanes(BoundaryType::lane_flxcor_send));
}

void Mesh::BuildLaneTagMap_(int lane) {
  const int comm_lanes = LaneCommLanes_();
  auto &tags = lane_tag_maps_[lane];
  tags.clear();
  for (auto &pmb : block_list)
    tags.AddLaneNeighbors(pmb.get(), pmb->GetLaneNeighbors(lane), comm_lanes);
  tags.ResolveMap();
}

namespace {
// The variables of a block with buffers on lane: those with registered boundary ops and
// the fluxes with a registered combine op, in the order ForEachBoundary visits them
template <class F>
void ForEachLaneVariable(MeshBlock *pmb, int lane, F func) {
  for (auto &v : pmb->meshblock_data.Get()->GetVariableVector()) {
    if (!v->IsSet(Metadata::Lane) || v->GetLane() != lane) continue;
    if (v->IsSet(Metadata::Flux) ? v->HasFluxCorrectionOps() : v->HasBoundaryOps())
      func(v);
  }
}
} // namespace

void Mesh::LaneBufferSizes_(MeshBlock *pmb, int lane, std::vector<int> &sizes) {
  sizes.clear();
  const auto &nbs = pmb->GetLaneNeighbors(lane);
  ForEachLaneVariable(pmb, lane, [&](const auto &v) {
    for (const auto &nb : nbs)
      sizes.push_back(GetLaneBufferSize(pmb, nb, v, v->IsSet(Metadata::Flux)));
  });
}

void Mesh::RecordLaneBufferSizes_(int lane) {
  for (auto &pmb : block_list) {
    if (static_cast<int>(pmb->lane_buffer_sizes.size()) <= lane)
      pmb->lane_buffer_sizes.resize(lane + 1);
    LaneBufferSizes_(pmb.get(), lane, pmb->lane_buffer_sizes[lane]);
  }
}

bool Mesh::RebuildLane(LaneId lane, bool full) {
  return RebuildLane_(static_cast<int>(lane), full, nullptr);
}

bool Mesh::RebuildLane(LaneId lane, const std::vector<int> &gids) {
  return RebuildLane_(static_cast<int>(lane), false, &gids);
}

bool Mesh::RebuildLane_(const int lane, const bool full, const std::vector<int> *gids) {
  PARTHENON_INSTRUMENT
  PARTHENON_REQUIRE_THROWS(lane_providers_.count(lane) > 0,
                           "RebuildLane on lane " + std::to_string(lane) +
                               " without a neighbor provider.");
  constexpr std::int64_t max_it = 1e10;

  if (full) {
    // Variables (and fluxes) on this lane
    std::unordered_set<std::string> labels;
    for (const auto &[vid, m] : resolved_packages->AllFields())
      if (m.IsSet(Metadata::Lane) && m.GetLane() == lane) labels.insert(vid.label());
    auto on_lane = [&labels](const channel_key_t &key) {
      return labels.count(std::get<2>(key)) > 0;
    };
    // The lane is quiescent by contract; wait for its last sends to complete
    bool can_delete;
    std::int64_t test_iters = 0;
    do {
      can_delete = true;
      for (auto &[key, buf] : boundary_comm_map)
        if (on_lane(key)) can_delete = buf.IsSafeToDelete() && can_delete;
      test_iters++;
    } while (!can_delete && test_iters < max_it);
    PARTHENON_REQUIRE(test_iters < max_it,
                      "Too many iterations waiting to delete lane buffers.");

    SetLaneNeighbors_(lane);
    BuildLaneTagMap_(lane);
    // Bumps the lane's epoch: only this lane's boundary caches re-initialize
    boundary_comm_map.EraseLane(lane, on_lane);
    BuildLaneBoundaryBuffers(mesh_data.Add("base", GetBasePartition()), lane);
    RecordLaneBufferSizes_(lane);
    return true;
  }

  // Incremental: diff each block's new neighbor list and buffer sizes against the ones
  // the lane was built with
  const int comm_lanes = LaneCommLanes_();
  const int ncl_lane = GetNumberOfCommLanes(BoundaryType::lane);
  const int ncl_fsend = GetNumberOfCommLanes(BoundaryType::lane_flxcor_send);
  const int ncl_frecv = GetNumberOfCommLanes(BoundaryType::lane_flxcor_recv);
  const int me = Globals::my_rank;
  auto &tags = lane_tag_maps_[lane];

  std::vector<channel_key_t> stale; // buffers to drop (and rebuild if still listed)
  std::vector<bool> dirty(block_list.size(), false); // blocks with buffers to build
  std::set<int> touched;                              // ranks whose lane tags change
  // The send buffer and, from another rank, the receive buffer of one channel
  auto drop = [&](MeshBlock *pmb, const NeighborBlock &nb, const auto &v) {
    const bool flux = v->IsSet(Metadata::Flux);
    const auto send = flux ? BoundaryType::lane_flxcor_send : BoundaryType::lane;
    const auto recv = flux ? BoundaryType::lane_flxcor_recv : BoundaryType::lane;
    for (int cl = 0; cl < (flux ? ncl_fsend : ncl_lane); ++cl)
      stale.push_back(SendKey(pmb, nb, v, send, cl));
    if (nb.rank != me)
      for (int cl = 0; cl < (flux ? ncl_frecv : ncl_lane); ++cl)
        stale.push_back(ReceiveKey(pmb, nb, v, recv, cl));
  };
  auto drop_all = [&](MeshBlock *pmb, const NeighborBlock &nb) {
    ForEachLaneVariable(pmb, lane, [&](const auto &v) { drop(pmb, nb, v); });
  };

  for (auto &spmb : block_list) {
    if (static_cast<int>(spmb->lane_neighbors.size()) <= lane)
      spmb->lane_neighbors.resize(lane + 1);
    if (static_cast<int>(spmb->lane_buffer_sizes.size()) <= lane)
      spmb->lane_buffer_sizes.resize(lane + 1);
  }
  // The blocks to diff: all local blocks, or the listed ones (local gids are contiguous)
  std::vector<MeshBlock *> diffed;
  if (gids == nullptr) {
    for (auto &spmb : block_list)
      diffed.push_back(spmb.get());
  } else if (!block_list.empty()) {
    const int gid0 = block_list[0]->gid;
    std::vector<bool> listed(block_list.size(), false);
    for (const int gid : *gids) {
      const int lid = gid - gid0;
      PARTHENON_REQUIRE(lid >= 0 && lid < static_cast<int>(block_list.size()) &&
                            block_list[lid]->gid == gid,
                        "RebuildLane: listed gid " + std::to_string(gid) +
                            " is not a local block.");
      listed[lid] = true;
    }
    for (std::size_t lid = 0; lid < block_list.size(); ++lid)
      if (listed[lid]) diffed.push_back(block_list[lid].get());
  }

  std::vector<LaneNeighbor> lnbs;
  std::vector<NeighborBlock> row;
  std::vector<int> old_of_new, sizes;
  for (MeshBlock *pmb : diffed) {
    auto &old = pmb->lane_neighbors[lane];
    ListLaneNeighbors_(*pmb, lane, lnbs);

    // Merge the sorted old and new lists
    const auto key_new = [&](const LaneNeighbor &l) {
      return std::make_tuple(l.channel_me, l.gid, l.channel_nb);
    };
    const auto key_old = [](const NeighborBlock &nb) {
      return std::make_tuple(nb.lane_channel, nb.gid, nb.lane_nb_channel);
    };
    old_of_new.assign(lnbs.size(), -1);
    bool same = lnbs.size() == old.size();
    std::size_t j = 0;
    for (std::size_t n = 0; n < lnbs.size(); ++n) {
      while (j < old.size() && key_old(old[j]) < key_new(lnbs[n])) {
        same = false;
        ++j;
      }
      if (j < old.size() && key_old(old[j]) == key_new(lnbs[n]) &&
          old[j].rank == ranklist[lnbs[n].gid]) {
        old_of_new[n] = static_cast<int>(j++);
      } else {
        same = false;
      }
    }
    same = same && j == old.size();

    if (!same) {
      std::vector<bool> kept(old.size(), false);
      for (const int o : old_of_new)
        if (o >= 0) kept[o] = true;
      for (std::size_t o = 0; o < old.size(); ++o) {
        if (kept[o]) continue;
        drop_all(pmb, old[o]);
        tags.RemoveLaneNeighbor(pmb, old[o], comm_lanes);
        touched.insert(old[o].rank);
      }
      row.clear();
      row.reserve(lnbs.size());
      for (std::size_t n = 0; n < lnbs.size(); ++n) {
        if (old_of_new[n] >= 0) {
          row.push_back(old[old_of_new[n]]);
        } else {
          row.push_back(MakeLaneNeighbor_(*pmb, lnbs[n]));
          tags.AddLaneNeighbor(pmb, row.back(), comm_lanes);
          touched.insert(row.back().rank);
        }
      }
      std::swap(old, row);
      dirty[pmb->lid] = true;
    }

    // Buffer sizes, laid out (lane variable, neighbor)
    LaneBufferSizes_(pmb, lane, sizes);
    auto &old_sizes = pmb->lane_buffer_sizes[lane];
    // (after the swap above, row holds the old list)
    const std::size_t nn = lnbs.size(), no = same ? nn : row.size();
    const std::size_t nv = nn > 0 ? sizes.size() / nn : 0;
    const bool comparable = old_sizes.size() == nv * no;
    if (!same || sizes != old_sizes) {
      int iv = 0;
      ForEachLaneVariable(pmb, lane, [&](const auto &v) {
        for (std::size_t n = 0; n < nn; ++n) {
          const int o = old_of_new[n];
          if (o < 0) continue; // new channel, no buffer yet
          if (!comparable || sizes[iv * nn + n] != old_sizes[iv * no + o]) {
            drop(pmb, old[n], v);
            dirty[pmb->lid] = true;
          }
        }
        ++iv;
      });
      std::swap(old_sizes, sizes);
    }
  }

  // Every channel to a rank whose tags change gets new buffers (tags of channels on this
  // rank are not used, so those buffers stay)
  const bool local_touched = touched.erase(me) > 0;
  if (!touched.empty()) {
    for (auto &spmb : block_list) {
      MeshBlock *pmb = spmb.get();
      for (const auto &nb : pmb->lane_neighbors[lane]) {
        if (touched.count(nb.rank) == 0) continue;
        drop_all(pmb, nb);
        dirty[pmb->lid] = true;
      }
    }
  }
  tags.ResolveMap(touched);
  if (local_touched) tags.ResolveMap(std::set<int>{me});

  if (stale.empty() && std::none_of(dirty.begin(), dirty.end(), [](bool d) { return d; }))
    return false;

  // The lane is quiescent by contract; wait for the last sends of dropped buffers
  bool can_delete;
  std::int64_t test_iters = 0;
  do {
    can_delete = true;
    for (const auto &key : stale) {
      auto it = boundary_comm_map.count(key) ? &boundary_comm_map.at(key) : nullptr;
      if (it != nullptr) can_delete = it->IsSafeToDelete() && can_delete;
    }
    test_iters++;
  } while (!can_delete && test_iters < max_it);
  PARTHENON_REQUIRE(test_iters < max_it,
                    "Too many iterations waiting to delete lane buffers.");
  for (const auto &key : stale)
    boundary_comm_map.erase(key);
  // Only this lane's boundary caches re-initialize
  boundary_comm_map.BumpLaneEpoch(lane);
  BuildLaneBoundaryBuffers(mesh_data.Add("base", GetBasePartition()), lane, &dirty);
  return true;
}

} // namespace parthenon

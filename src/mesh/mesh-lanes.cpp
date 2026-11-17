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
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bvals/comms/bvals_in_one.hpp"
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

void Mesh::SetLaneNeighbors_(int lane) {
  const auto &provider = lane_providers_.at(lane);
  const int nbtot = static_cast<int>(ranklist.size());
  std::vector<LaneNeighbor> lnbs;
  for (auto &pmb : block_list) {
    lnbs.clear();
    provider(*pmb, lnbs);
    std::sort(lnbs.begin(), lnbs.end(), [](const auto &a, const auto &b) {
      return std::make_tuple(a.channel_me, a.gid, a.channel_nb) <
             std::make_tuple(b.channel_me, b.gid, b.channel_nb);
    });
    if (static_cast<int>(pmb->lane_neighbors.size()) <= lane)
      pmb->lane_neighbors.resize(lane + 1);
    auto &nbs = pmb->lane_neighbors[lane];
    nbs.clear();
    nbs.reserve(lnbs.size());
    for (std::size_t n = 0; n < lnbs.size(); ++n) {
      const auto &l = lnbs[n];
      PARTHENON_REQUIRE(l.gid >= 0 && l.gid < nbtot,
                        "Lane neighbor gid " + std::to_string(l.gid) + " of block " +
                            std::to_string(pmb->gid) + " out of range.");
      PARTHENON_REQUIRE(l.channel_me >= 0 && l.channel_me < MaxLaneChannels &&
                            l.channel_nb >= 0 && l.channel_nb < MaxLaneChannels,
                        "Lane channel ids must be in [0, MaxLaneChannels).");
      PARTHENON_REQUIRE(n == 0 || l.gid != lnbs[n - 1].gid ||
                            l.channel_me != lnbs[n - 1].channel_me ||
                            l.channel_nb != lnbs[n - 1].channel_nb,
                        "Duplicate lane neighbor of block " + std::to_string(pmb->gid));
      // Lane ops run on same-level exchanges only, like registered boundary ops
      PARTHENON_REQUIRE(loclist[l.gid].level() == pmb->loc.level(),
                        "Lane neighbors must be on the same refinement level.");
      nbs.emplace_back(this, loclist[l.gid], loclist[l.gid], ranklist[l.gid], l.gid,
                       Kokkos::Array<int, 3>{0, 0, 0}, -1, -1, 0, 0,
                       pmb->block_coarsenings);
      nbs.back().lane_channel = l.channel_me;
      nbs.back().lane_nb_channel = l.channel_nb;
    }
  }
}

void Mesh::BuildLaneTagMap_(int lane) {
  for (auto btype : {BoundaryType::lane, BoundaryType::lane_flxcor_send})
    LockCommChannelNumbers(btype);
  const int channels = std::max(GetNumberOfCommChannels(BoundaryType::lane),
                                GetNumberOfCommChannels(BoundaryType::lane_flxcor_send));
  auto &tags = lane_tag_maps_[lane];
  tags.clear();
  for (auto &pmb : block_list)
    tags.AddLaneNeighbors(pmb.get(), pmb->GetLaneNeighbors(lane), channels);
  tags.ResolveMap();
}

void Mesh::RebuildLane(LaneId lane_id) {
  PARTHENON_INSTRUMENT
  const int lane = static_cast<int>(lane_id);
  PARTHENON_REQUIRE_THROWS(lane_providers_.count(lane) > 0,
                           "RebuildLane on lane " + std::to_string(lane) +
                               " without a neighbor provider.");

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
  constexpr std::int64_t max_it = 1e10;
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
  boundary_comm_map.EraseIf(on_lane);
  BuildLaneBoundaryBuffers(mesh_data.Add("base", GetBasePartition()), lane);
}

} // namespace parthenon

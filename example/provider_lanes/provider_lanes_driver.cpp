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

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <bvals/comms/bnd_info.hpp>
#include <bvals/comms/coalesced_buffers.hpp>
#include <parthenon/driver.hpp>

#include "provider_lanes.hpp"

using namespace parthenon::driver::prelude;
using parthenon::BoundaryType;
using provider_lanes::kGroups;
using provider_lanes::kRing;
using provider_lanes::LaneGraph;

namespace {
// How often the boundary caches of the exchanges were initialized, summed over the
// MeshData partitions and ranks
struct CacheInits {
  std::int64_t groups = 0, ring = 0, ghosts = 0, partitions = 0;
};

// Runs ncycles lane exchanges, checking all of them, and reshuffles the groups lane
// before cycle rebuild_cycle.
class ProviderLanesDriver : public Driver {
 public:
  ProviderLanesDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm), graph_(std::make_shared<LaneGraph>(pm->nbtotal, 0)) {
    // The providers read the current graph; registering them after the mesh is built
    // builds the lanes right away
    auto graph = graph_;
    pmesh->RegisterNeighborProvider(kGroups,
                                    [graph](const parthenon::MeshBlock &pmb,
                                            std::vector<parthenon::LaneNeighbor> &nbs) {
                                      graph->GroupNeighbors(pmb.gid, nbs);
                                    });
    pmesh->RegisterNeighborProvider(kRing,
                                    [graph](const parthenon::MeshBlock &pmb,
                                            std::vector<parthenon::LaneNeighbor> &nbs) {
                                      graph->RingNeighbors(pmb.gid, nbs);
                                    });
  }

  DriverStatus Execute() override {
    const int ncycles = pinput->GetOrAddInteger("provider_lanes", "ncycles", 4);
    const int rebuild_cycle =
        pinput->GetOrAddInteger("provider_lanes", "rebuild_cycle", 2);
    // rebuild = false changes the graph without rebuilding the lane: the check fails
    const bool rebuild = pinput->GetOrAddBoolean("provider_lanes", "rebuild", true);
    // full_rebuild = true rebuilds the whole groups lane instead of what changed
    const bool full = pinput->GetOrAddBoolean("provider_lanes", "full_rebuild", false);
    // Number of graph changes, each with its RebuildLane, before the next exchange
    const int nrebuilds = pinput->GetOrAddInteger("provider_lanes", "rebuilds", 1);
    const bool coalesced = pmesh->do_coalesced_comms;
    // pair_change = true: on every other cycle after the first, the rows capacity of one
    // pair of group neighbors on two ranks grows, and only that pair's lane info is sent
    const bool pair_change =
        pinput->GetOrAddBoolean("provider_lanes", "pair_change", false);
    // idle_rank = r: in cycle 1 the other ranks take rank r's blocks out of the groups
    // and rebuild the lane, while rank r keeps its graph, neither rebuilds nor exchanges
    // the groups lane; in cycle 2 the other ranks go back to the graph and rebuild, and
    // rank r, whose lane did not change, again does not rebuild. Ignored below r + 1 ranks.
    int idle_rank = pinput->GetOrAddInteger("provider_lanes", "idle_rank", -1);
    if (idle_rank >= parthenon::Globals::nranks) idle_rank = -1;
    const auto rank_of = RankOfGid();
    const auto pair = OffRankPair(rank_of);
    int capacity = provider_lanes::GetCapacity(pmesh);
    int bump = 0;
    for (int cycle = 0; cycle < ncycles; ++cycle) {
      const CacheInits before = CountCacheInits();
      const auto infos_before = CountLaneInfos();
      // The pair's buffers change only while the two blocks are group neighbors
      const bool bumped = pair_change && pair[0] >= 0 && cycle % 2 == 1 &&
                          cycle != rebuild_cycle && GroupNeighbors(pair[0], pair[1]);
      const bool idling = idle_rank >= 0 && (cycle == 1 || cycle == 2);
      const bool idle = idling && cycle == 1 && parthenon::Globals::my_rank == idle_rank;
      const bool rebuilt = (cycle == rebuild_cycle && rebuild) || bumped;
      if (idling) {
        auto graph = LaneGraph(pmesh->nbtotal, graph_->version);
        std::vector<bool> out(pmesh->nbtotal, false);
        for (int gid = 0; gid < pmesh->nbtotal; ++gid)
          out[gid] = cycle == 1 && rank_of[gid] == idle_rank;
        graph.Isolate(out);
        if (parthenon::Globals::my_rank != idle_rank) {
          *graph_ = graph;
          pmesh->RebuildLane(kGroups, full);
        }
      }
      if (bumped) {
        provider_lanes::SetBump(pmesh, pair[0], pair[1], ++bump);
        pmesh->RebuildLane(kGroups, full);
      }
      if (cycle == rebuild_cycle) {
        for (int r = 0; r < nrebuilds; ++r) {
          // Every rank changes the graph the same way and rebuilds the lane between
          // exchanges (no lane message in flight)
          *graph_ = LaneGraph(pmesh->nbtotal, graph_->version + 1);
          // The rows payload may now need more room: raise the capacity (BufferSize)
          // and reallocate the lane's buffers with RebuildLane
          if (rebuild) {
            capacity += 3;
            provider_lanes::SetCycleAndCapacity(pmesh, cycle, capacity);
            pmesh->RebuildLane(kGroups, full);
            // The ring lane did not change: its rebuild keeps its buffers and caches
            pmesh->RebuildLane(kRing);
          }
        }
      }
      provider_lanes::SetCycleAndCapacity(pmesh, cycle, capacity);
      RunCycle(cycle, !idle);
      const auto res = provider_lanes::Check(pmesh, *graph_, cycle, !idle);
      const CacheInits after = CountCacheInits();
      // After cycle 0 only RebuildLane(kGroups) re-initializes caches, and only the
      // groups lane's (send and receive of the record and flux exchanges)
      // (a pair change re-initializes only the caches on the pair's two ranks)
      const std::int64_t expected_groups =
          cycle == rebuild_cycle && rebuild ? 4 * after.partitions : 0;
      const std::int64_t dgroups = after.groups - before.groups;
      const bool groups_ok = bumped || idling
                                 ? dgroups > 0 && dgroups <= 4 * after.partitions
                                 : dgroups == expected_groups;
      const int cache_errors = cycle > 0 && (!groups_ok || after.ring != before.ring ||
                                             after.ghosts != before.ghosts);
      // With coalesced comms only a rebuild that changed the groups lane sends lane
      // infos, and only the groups lane's
      const auto infos_after = CountLaneInfos();
      const std::int64_t infos_groups = infos_after[0] - infos_before[0];
      const std::int64_t infos_ring = infos_after[1] - infos_before[1];
      const bool multirank = parthenon::Globals::nranks > 1;
      const int info_errors =
          coalesced &&
          (infos_ring != 0 || (!idling && (rebuilt && multirank) != (infos_groups > 0)) ||
           (bumped && infos_groups != 2));
      if (parthenon::Globals::my_rank == 0) {
        std::cout << "provider_lanes: cycle " << cycle << ", graph version "
                  << graph_->version << ", groups of size";
        for (int s = 2; s <= provider_lanes::kMaxGroup; ++s)
          std::cout << " " << s << ":" << res.group_sizes[s];
        std::cout << ", " << res.nchecked << " values checked, " << res.nwrong
                  << " wrong, hash " << std::hex << res.hash << std::dec << ", rows "
                  << res.rows_sent << " of " << res.rows_capacity << " sent"
                  << "; cache inits groups +" << after.groups - before.groups << " ring +"
                  << after.ring - before.ring << " ghosts +"
                  << after.ghosts - before.ghosts << (cache_errors ? " WRONG" : "");
        if (coalesced)
          std::cout << "; lane infos groups +" << infos_groups << " ring +" << infos_ring
                    << (info_errors ? " WRONG" : "");
        std::cout << std::endl;
      }
      nwrong += res.nwrong + cache_errors + info_errors;
    }
    return nwrong == 0 ? DriverStatus::complete : DriverStatus::failed;
  }

  int nwrong = 0;

 private:
  // groups = false leaves out the groups lane's exchanges
  void RunCycle(int cycle, bool groups) {
    TaskCollection tc;
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      // One MeshData holds both lanes' variables and u; each lane exchange touches only
      // its lane's variables, the ghost exchange only u
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      auto set = tl.AddTask(TaskID(0), provider_lanes::SetValues, md.get(), cycle);
      if (groups) parthenon::AddLaneExchangeTasks(set, tl, md, kGroups);
      parthenon::AddLaneExchangeTasks(set, tl, md, kRing);
      if (groups) parthenon::AddLaneFluxCorrectionTasks(set, tl, md, kGroups);
      parthenon::AddBoundaryExchangeTasks(set, tl, md, pmesh->multilevel);
    }
    tc.Execute();
  }

  // Rank of every gid
  std::vector<int> RankOfGid() const {
    std::vector<int> rank(pmesh->nbtotal, parthenon::Globals::my_rank);
#ifdef MPI_PARALLEL
    std::vector<int> mine(pmesh->nbtotal, -1);
    for (const auto &pmb : pmesh->block_list)
      mine[pmb->gid] = parthenon::Globals::my_rank;
    MPI_Allreduce(mine.data(), rank.data(), pmesh->nbtotal, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    return rank;
  }

  // A pair of blocks on two ranks that are group neighbors in layer 0 ({-1, -1} if none)
  std::array<int, 2> OffRankPair(const std::vector<int> &rank) const {
    for (const auto &g : graph_->members[0])
      for (int a : g)
        for (int b : g)
          if (rank[a] != rank[b]) return {a, b};
    return {-1, -1};
  }
  bool GroupNeighbors(int a, int b) const {
    for (int layer = 0; layer < provider_lanes::kLayers; ++layer)
      if (graph_->group[layer][a] == graph_->group[layer][b]) return true;
    return false;
  }

  // Lane info messages sent by all ranks, groups and ring lanes
  std::array<std::int64_t, 2> CountLaneInfos() {
    auto &sent = pmesh->pcoalesced_comms->lane_infos_sent;
    std::array<std::int64_t, 2> n{0, 0};
    for (int l = 0; l < 2; ++l) {
      auto it = sent.find(l == 0 ? static_cast<int>(kGroups) : static_cast<int>(kRing));
      if (it != sent.end()) n[l] = it->second;
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, n.data(), 2, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    return n;
  }

  CacheInits CountCacheInits() {
    constexpr int groups = static_cast<int>(kGroups), ring = static_cast<int>(kRing);
    CacheInits n;
    n.partitions = pmesh->DefaultNumPartitions();
    for (int i = 0; i < n.partitions; ++i) {
      auto &cache = pmesh->mesh_data.GetOrAdd("base", i)->GetBvarsCache();
      for (bool send : {true, false}) {
        n.groups +=
            cache.GetLaneSubCache(BoundaryType::lane, send, groups).initializations;
        n.groups += cache
                        .GetLaneSubCache(send ? BoundaryType::lane_flxcor_send
                                              : BoundaryType::lane_flxcor_recv,
                                         send, groups)
                        .initializations;
        n.ring += cache.GetLaneSubCache(BoundaryType::lane, send, ring).initializations;
        n.ghosts += cache.GetSubCache(BoundaryType::any, send).initializations;
      }
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, &n, 4, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    return n;
  }

  std::shared_ptr<LaneGraph> graph_;
};
} // namespace

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = [](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    packages.Add(provider_lanes::Initialize(pin.get()));
    return packages;
  };
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  pman.ParthenonInitPackagesAndMesh();

  int nwrong = 0;
  {
    ProviderLanesDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver.Execute();
    nwrong = driver.nwrong;
  }
  pman.ParthenonFinalize();
  return nwrong == 0 ? 0 : 1;
}

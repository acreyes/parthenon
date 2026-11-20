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

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <bvals/comms/bnd_info.hpp>
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
    int capacity = provider_lanes::GetCapacity(pmesh);
    for (int cycle = 0; cycle < ncycles; ++cycle) {
      const CacheInits before = CountCacheInits();
      if (cycle == rebuild_cycle) {
        // Every rank changes the graph the same way and rebuilds the lane between
        // exchanges (no lane message in flight)
        *graph_ = LaneGraph(pmesh->nbtotal, graph_->version + 1);
        // The rows payload may now need more room: raise the capacity (BufferSize) and
        // reallocate the lane's buffers with RebuildLane
        if (rebuild) {
          capacity += 3;
          provider_lanes::SetCycleAndCapacity(pmesh, cycle, capacity);
          pmesh->RebuildLane(kGroups);
        }
      }
      provider_lanes::SetCycleAndCapacity(pmesh, cycle, capacity);
      RunCycle(cycle);
      const auto res = provider_lanes::Check(pmesh, *graph_, cycle);
      const CacheInits after = CountCacheInits();
      // After cycle 0 only RebuildLane(kGroups) re-initializes caches, and only the
      // groups lane's (send and receive of the record and flux exchanges)
      const std::int64_t expected_groups =
          cycle == rebuild_cycle && rebuild ? 4 * after.partitions : 0;
      const int cache_errors =
          cycle > 0 && (after.groups - before.groups != expected_groups ||
                        after.ring != before.ring || after.ghosts != before.ghosts);
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
                  << after.ghosts - before.ghosts << (cache_errors ? " WRONG" : "")
                  << std::endl;
      }
      nwrong += res.nwrong + cache_errors;
    }
    return nwrong == 0 ? DriverStatus::complete : DriverStatus::failed;
  }

  int nwrong = 0;

 private:
  void RunCycle(int cycle) {
    TaskCollection tc;
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      // One MeshData holds both lanes' variables and u; each lane exchange touches only
      // its lane's variables, the ghost exchange only u
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      auto set = tl.AddTask(TaskID(0), provider_lanes::SetValues, md.get(), cycle);
      parthenon::AddLaneExchangeTasks(set, tl, md, kGroups);
      parthenon::AddLaneExchangeTasks(set, tl, md, kRing);
      parthenon::AddLaneFluxCorrectionTasks(set, tl, md, kGroups);
      parthenon::AddBoundaryExchangeTasks(set, tl, md, pmesh->multilevel);
    }
    tc.Execute();
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

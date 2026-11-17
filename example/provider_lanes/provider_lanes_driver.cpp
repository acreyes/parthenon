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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <parthenon/driver.hpp>

#include "provider_lanes.hpp"

using namespace parthenon::driver::prelude;
using provider_lanes::kGroups;
using provider_lanes::kRing;
using provider_lanes::LaneGraph;

namespace {
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
    for (int cycle = 0; cycle < ncycles; ++cycle) {
      if (cycle == rebuild_cycle) {
        // Every rank changes the graph the same way and rebuilds the lane between
        // exchanges (no lane message in flight)
        *graph_ = LaneGraph(pmesh->nbtotal, graph_->version + 1);
        if (rebuild) pmesh->RebuildLane(kGroups);
      }
      RunCycle(cycle);
      const auto res = provider_lanes::Check(pmesh, *graph_, cycle);
      if (parthenon::Globals::my_rank == 0) {
        std::cout << "provider_lanes: cycle " << cycle << ", graph version "
                  << graph_->version << ", groups of size";
        for (int s = 2; s <= provider_lanes::kMaxGroup; ++s)
          std::cout << " " << s << ":" << res.group_sizes[s];
        std::cout << ", " << res.nchecked << " values checked, " << res.nwrong
                  << " wrong, hash " << std::hex << res.hash << std::dec << std::endl;
      }
      nwrong += res.nwrong;
    }
    return nwrong == 0 ? DriverStatus::complete : DriverStatus::failed;
  }

  int nwrong = 0;

 private:
  void RunCycle(int cycle) {
    TaskCollection tc;
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &region = tc.AddRegion(num_partitions);
    const std::vector<std::string> groups_vars{"group_rec"};
    const std::vector<std::string> ring_vars{"ring_rec"};
    const std::vector<std::string> flux_vars{
        "q", parthenon::internal_fluxname + parthenon::internal_varname_seperator + "q"};
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      // One MeshData per lane exchange, holding only that lane's variables
      auto &md_groups = pmesh->mesh_data.Add("lane_groups", md, groups_vars);
      auto &md_ring = pmesh->mesh_data.Add("lane_ring", md, ring_vars);
      auto &md_flux = pmesh->mesh_data.Add("lane_flux", md, flux_vars);
      auto set = tl.AddTask(TaskID(0), provider_lanes::SetValues, md.get(), cycle);
      parthenon::AddLaneExchangeTasks(set, tl, md_groups, kGroups);
      parthenon::AddLaneExchangeTasks(set, tl, md_ring, kRing);
      parthenon::AddLaneFluxCorrectionTasks(set, tl, md_flux, kGroups);
    }
    tc.Execute();
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

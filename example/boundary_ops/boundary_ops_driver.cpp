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

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <unordered_map>

#include <parthenon/driver.hpp>

#include "boundary_ops.hpp"
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"

using namespace parthenon::driver::prelude;

namespace {
// Exchanges ghost zones once and checks them.
class BoundaryOpsDriver : public Driver {
 public:
  BoundaryOpsDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : Driver(pin, app_in, pm) {
    InitializeOutputs();
  }

  DriverStatus Execute() override {
    TaskCollection tc;
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      auto set = tl.AddTask(TaskID(0), boundary_ops::SetBlockValues, md.get());
      parthenon::AddBoundaryExchangeTasks(set, tl, md, true);
    }
    tc.Execute();

    int naveraged = 0;
    nwrong = boundary_ops::CheckGhosts(pmesh, &naveraged);
    if (parthenon::Globals::my_rank == 0)
      std::cout << "boundary_ops: " << pmesh->nbtotal << " blocks, " << naveraged
                << " ghost regions averaged over several neighbors, " << nwrong
                << " wrong ghost cells" << std::endl;
    pouts->MakeOutputs(pmesh, pinput);
    return nwrong == 0 ? DriverStatus::complete : DriverStatus::failed;
  }

  int nwrong = 0;
};

// One fan of `valence` trees around a central node, the forest of
// tst/unit/test_forest.cpp. Trees meet at the central node and pairwise along its
// spokes; the outer edges are domain boundaries.
void AddFan(parthenon::forest::ForestDefinition *forest_def, int valence, int *node_count,
            int *tree_count, Real x0) {
  using parthenon::forest::Node;
  const int nc = *node_count;
  std::unordered_map<std::uint64_t, std::shared_ptr<Node>> n;
  for (int p = 0; p < 2 * valence; ++p)
    n[nc + p] = Node::create(
        nc + p, {std::sin(p * M_PI / valence) + x0, std::cos(p * M_PI / valence)});
  n[nc + 2 * valence] = Node::create(nc + 2 * valence, {x0, 0.0});
  for (int t = 0; t < valence; ++t) {
    const auto &a = n[nc + 2 * t + 1];
    const auto &b = n[nc + 2 * t];
    const auto &c = n[nc + (2 * t + 2) % (2 * valence)];
    forest_def->AddFace(*tree_count + t, {a, b, c, n[nc + 2 * valence]});
    forest_def->AddBC(parthenon::forest::Edge({a, b}), parthenon::BoundaryFlag::user);
    forest_def->AddBC(parthenon::forest::Edge({a, c}), parthenon::BoundaryFlag::user);
    // Refine each tree once, so the fan also has nodes and edges inside trees
    forest_def->AddInitialRefinement(
        parthenon::LogicalLocation(*tree_count + t, 1, 0, 0, 0));
  }
  *node_count += 2 * valence + 1;
  *tree_count += valence;
}
} // namespace

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = [](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    packages.Add(boundary_ops::Initialize(pin.get()));
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

  // Disconnected fans whose central nodes have valence min_valence..max_valence
  auto *pin = pman.pinput.get();
  const int vmin = pin->GetOrAddInteger("boundary_ops", "min_valence", 3);
  const int vmax = pin->GetOrAddInteger("boundary_ops", "max_valence", 6);
  parthenon::forest::ForestDefinition forest_def;
  int node_count = 0, tree_count = 0;
  for (int v = vmin; v <= vmax; ++v)
    AddFan(&forest_def, v, &node_count, &tree_count, 2.2 * (v - vmin));
  pman.ParthenonInitPackagesAndMesh(forest_def);

  int nwrong = 0;
  {
    BoundaryOpsDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver.Execute();
    nwrong = driver.nwrong;
  }
  pman.ParthenonFinalize();
  return nwrong == 0 ? 0 : 1;
}

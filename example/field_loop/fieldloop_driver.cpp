//========================================================================================
// (C) (or copyright) 2020-2024. Triad National Security, LLC. All rights reserved.
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

#include <memory>
#include <string>
#include <vector>

// Local Includes
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "fieldloop_driver.hpp"
#include "fieldloop_package.hpp"
#include "interface/metadata.hpp"
#include "interface/update.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parthenon/driver.hpp"
#include "prolong_restrict/prolong_restrict.hpp"

using namespace parthenon::driver::prelude;

namespace fieldloop_example {

// *************************************************//
// define the application driver. in this case,    *//
// that mostly means defining the MakeTaskList     *//
// function.                                       *//
// *************************************************//
FieldLoopDriver::FieldLoopDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("parthenon/mesh", "ix1_bc");
  pin->CheckRequired("parthenon/mesh", "ox1_bc");
  pin->CheckRequired("parthenon/mesh", "ix2_bc");
  pin->CheckRequired("parthenon/mesh", "ox2_bc");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/mesh", "refinement");
  pin->CheckDesired("parthenon/mesh", "numlevel");
  pin->CheckDesired("fieldloop", "cfl");
  pin->CheckDesired("fieldloop", "vx");
}

// See the fieldloop.hpp declaration for a description of how this function gets called.
TaskCollection FieldLoopDriver::MakeTaskCollection(BlockList_t &blocks, const int stage) {
  using namespace parthenon::Update;
  TaskCollection tc;
  TaskID none(0);

  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  const auto &stage_name = integrator->stage_name;

  // first make other useful containers
  if (stage == 1) {
    for (int i = 0; i < blocks.size(); i++) {
      auto &pmb = blocks[i];
      // first make other useful containers
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      for (int s = 1; s < integrator->nstages; s++)
        pmb->meshblock_data.Add(stage_name[s], base);
    }
  }

  const int num_partitions = pmesh->DefaultNumPartitions();

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region2 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region2[i];
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);

    const auto any = parthenon::BoundaryType::any;

    tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mc1);
    tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mc0);
  }

  // Number of task lists that can be executed independently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  assert(blocks.size() == async_region1.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region1[i];

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto advect_flux = tl.AddTask(none, fieldloop_package::CalculateFluxes, sc0);
  }

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

    auto set_flx = parthenon::AddFluxCorrectionTasks(none, tl, mc0, pmesh->multilevel);

    // compute the divergence of fluxes of conserved variables
    using parthenon::Metadata;
    using parthenon::MetadataFlag;

    auto staggered_update =
        tl.AddTask(set_flx, fieldloop_package::StaggeredUpdate<MeshData<Real>>,
                   mbase.get(), mc0.get(), beta, dt, mc1.get());

    // do boundary exchange
    parthenon::AddBoundaryExchangeTasks(staggered_update, tl, mc1, pmesh->multilevel);
  }

  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);
  assert(blocks.size() == async_region2.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    // set physical boundaries
    auto set_bc = tl.AddTask(none, parthenon::ApplyBoundaryConditions, sc1);

    // fill in derived fields
    auto fill_derived = tl.AddTask(
        set_bc, parthenon::Update::FillDerived<MeshBlockData<Real>>, sc1.get());

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt =
          tl.AddTask(fill_derived, EstimateTimestep<MeshBlockData<Real>>, sc1.get());

      // Update refinement
      if (pmesh->adaptive) {
        auto tag_refine = tl.AddTask(
            fill_derived, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
      }
    }
  }
  return tc;
}

} // namespace fieldloop_example

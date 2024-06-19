//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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

#include "amr_criteria/refinement_package.hpp"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>

#include "Kokkos_Array.hpp"
#include "Kokkos_Macros.hpp"
#include "amr_criteria/amr_criteria.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/meshblock.hpp"
#include "parameter_input.hpp"
#include "utils/instrument.hpp"

namespace parthenon {
namespace Refinement {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto ref = std::make_shared<StateDescriptor>("Refinement");

  int numcrit = 0;
  while (true) {
    std::string block_name = "parthenon/refinement" + std::to_string(numcrit);
    if (!pin->DoesBlockExist(block_name)) {
      break;
    }
    std::string method =
        pin->GetOrAddString(block_name, "method", "PLEASE SPECIFY method");
    ref->amr_criteria.push_back(AMRCriteria::MakeAMRCriteria(method, pin, block_name));
    numcrit++;
  }
  return ref;
}

AmrTag CheckAllRefinement(MeshBlockData<Real> *rc) {
  // Check all refinement criteria and return the maximum recommended change in
  // refinement level:
  //   delta_level = -1 => recommend derefinement
  //   delta_level = 0  => leave me alone
  //   delta_level = 1  => recommend refinement
  // NOTE: recommendations from this routine are NOT always followed because
  //    1) the code will not refine more than the global maximum level defined in
  //       <parthenon/mesh>/numlevel in the input
  //    2) the code must maintain proper nesting, which sometimes means a block that is
  //       tagged as "derefine" must be left alone (or possibly refined?) because of
  //       neighboring blocks.  Similarly for "do nothing"
  PARTHENON_INSTRUMENT
  MeshBlock *pmb = rc->GetBlockPointer();
  // delta_level holds the max over all criteria.  default to derefining.
  AmrTag delta_level = AmrTag::derefine;
  for (auto &pkg : pmb->packages.AllPackages()) {
    auto &desc = pkg.second;
    delta_level = std::max(delta_level, desc->CheckRefinement(rc));
    if (delta_level == AmrTag::refine) {
      // since 1 is the max, we can return without having to look at anything else
      return AmrTag::refine;
    }
    // call parthenon criteria that were registered
    for (auto &amr : desc->amr_criteria) {
      // get the recommended change in refinement level from this criteria
      AmrTag temp_delta = (*amr)(rc);
      if ((temp_delta == AmrTag::refine) && pmb->loc.level() >= amr->max_level) {
        // don't refine if we're at the max level
        temp_delta = AmrTag::same;
      }
      // maintain the max across all criteria
      delta_level = std::max(delta_level, temp_delta);
      if (delta_level == AmrTag::refine) {
        // 1 is the max, so just return
        return AmrTag::refine;
      }
    }
  }
  return delta_level;
}

AmrTag FirstDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                       const Real refine_criteria, const Real derefine_criteria) {
  PARTHENON_INSTRUMENT
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);
  Real maxd = 0.0;
  par_reduce(
      loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), bnds.ks, bnds.ke,
      bnds.js, bnds.je, bnds.is, bnds.ie,
      KOKKOS_LAMBDA(int k, int j, int i, Real &maxd) {
        Real scale = std::abs(q(k, j, i));
        Real d =
            0.5 * std::abs((q(k, j, i + 1) - q(k, j, i - 1))) / (scale + TINY_NUMBER);
        maxd = (d > maxd ? d : maxd);
        if (ndim > 1) {
          d = 0.5 * std::abs((q(k, j + 1, i) - q(k, j - 1, i))) / (scale + TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
        if (ndim > 2) {
          d = 0.5 * std::abs((q(k + 1, j, i) - q(k - 1, j, i))) / (scale + TINY_NUMBER);
          maxd = (d > maxd ? d : maxd);
        }
      },
      Kokkos::Max<Real>(maxd));

  if (maxd > refine_criteria) return AmrTag::refine;
  if (maxd < derefine_criteria) return AmrTag::derefine;
  return AmrTag::same;
}

AmrTag SecondDerivative(const AMRBounds &bnds, const ParArray3D<Real> &q,
                        const Real refine_criteria, const Real derefine_criteria) {
  PARTHENON_INSTRUMENT
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);
  Real maxd = 0.0;
  par_reduce(
      loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), bnds.ks, bnds.ke,
      bnds.js, bnds.je, bnds.is, bnds.ie,
      KOKKOS_LAMBDA(int k, int j, int i, Real &maxd) {
        Real aqt = std::abs(q(k, j, i)) + TINY_NUMBER;
        Real qavg = 0.5 * (q(k, j, i + 1) + q(k, j, i - 1));
        Real d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
        maxd = (d > maxd ? d : maxd);
        if (ndim > 1) {
          qavg = 0.5 * (q(k, j + 1, i) + q(k, j - 1, i));
          d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
          maxd = (d > maxd ? d : maxd);
        }
        if (ndim > 2) {
          qavg = 0.5 * (q(k + 1, j, i) + q(k - 1, j, i));
          d = std::abs(qavg - q(k, j, i)) / (std::abs(qavg) + aqt);
          maxd = (d > maxd ? d : maxd);
        }
      },
      Kokkos::Max<Real>(maxd));

  if (maxd > refine_criteria) return AmrTag::refine;
  if (maxd < derefine_criteria) return AmrTag::derefine;
  return AmrTag::same;
}

AmrTag LoehnerEstimator(const AMRBounds &bnds, const Coordinates_t &coords, const ParArray3D<Real> &q,
                        const ParArray4D<Real> &d1Scratch, const ParArray5D<Real> &d2Scratch,
                        const Real refine_criteria, const Real derefine_criteria, const Real refine_filter) {
  PARTHENON_INSTRUMENT
  const int ndim = 1 + (bnds.je > bnds.js) + (bnds.ke > bnds.ks);
  const int k3d = ndim > 2 ? 1 : 0;
  const int k2d = ndim > 1 ? 1 : 0;
  Kokkos::Array<Real,3> idx = {0.5/coords.Dxc<1>(), 0.5/coords.Dxc<2>(), 0.5/coords.Dxc<3>()};

  // first derivatives
  par_for(
     loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
     0, ndim-1, bnds.ks, bnds.ke, bnds.js, bnds.je, bnds.is, bnds.ie,
     KOKKOS_LAMBDA(const int dir, const int k, const int j, const int i)
     {
      const int ip = dir == 0 ? 1 : 0;
      const int jp = dir == 1 ? 1 : 0;
      const int kp = dir == 2 ? 1 : 0;
      d1Scratch(dir,k,j,i) = (q(k+kp,j+jp,i+ip) - q(k-kp,j-jp,i-ip))*idx[dir];
     });

  Kokkos::fence();
  const int is = bnds.is+1  ; const int ie = bnds.ie-1;
  const int js = bnds.js+k2d; const int je = bnds.je-k2d;
  const int ks = bnds.ks+k3d; const int ke = bnds.ke-k3d;
  par_for(
     loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
     0, ndim-1, 0, ndim-1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(const int dir2, const int dir1, const int k, const int j, const int i)
     {
      const int ip1 = dir1 == 0 ? 1 : 0;
      const int jp1 = dir1 == 1 ? 1 : 0;
      const int kp1 = dir1 == 2 ? 1 : 0;
      const int ip2 = dir2 == 0 ? 1 : 0;
      const int jp2 = dir2 == 1 ? 1 : 0;
      const int kp2 = dir2 == 2 ? 1 : 0;

      const int dir = dir1 + (ndim)*dir2;

      d2Scratch(0,dir,k,j,i) = ( d1Scratch(dir1,k+kp2,j+jp2,i+ip2) 
                                 - d1Scratch(dir1,k-kp2,j-jp2,i-ip2) )*idx[dir2];
      });

  par_for(
     loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
     0, ndim-1, 0, ndim-1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(const int dir2, const int dir1, const int k, const int j, const int i)
     {
      const int ip1 = dir1 == 0 ? 1 : 0;
      const int jp1 = dir1 == 1 ? 1 : 0;
      const int kp1 = dir1 == 2 ? 1 : 0;
      const int ip2 = dir2 == 0 ? 1 : 0;
      const int jp2 = dir2 == 1 ? 1 : 0;
      const int kp2 = dir2 == 2 ? 1 : 0;

      const int dir = dir1 + (ndim)*dir2;

      d2Scratch(1,dir,k,j,i) = std::abs(d1Scratch(dir1,k+kp2,j+jp2,i+ip2)) 
                               + std::abs(d1Scratch(dir1,k-kp2,j-jp2,i-ip2))*idx[dir2];
      });

  par_for(
     loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
     0, ndim-1, 0, ndim-1, ks, ke, js, je, is, ie,
     KOKKOS_LAMBDA(const int dir2, const int dir1, const int k, const int j, const int i)
     {
      const int ip1 = dir1 == 0 ? 1 : 0;
      const int jp1 = dir1 == 1 ? 1 : 0;
      const int kp1 = dir1 == 2 ? 1 : 0;
      const int ip2 = dir2 == 0 ? 1 : 0;
      const int jp2 = dir2 == 1 ? 1 : 0;
      const int kp2 = dir2 == 2 ? 1 : 0;

      const int dir = dir1 + (ndim)*dir2;

      d2Scratch(2,dir,k,j,i) = std::abs(q(k+kp2+kp1,j+jp2+jp1,i+ip2+ip1)) + std::abs(q(k-kp2+kp1,j-jp2+jp1,i-ip2+ip1))
                             + std::abs(q(k+kp2-kp1,j+jp2-jp1,i+ip2-ip1)) + std::abs(q(k-kp2-kp1,j-jp2-jp1,i-ip2-ip1));
      d2Scratch(2,dir,k,j,i) *= idx[dir1]*idx[dir2];
      });

  Kokkos::fence();
  Real error = 0.0;
  par_reduce(
      loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(),
      ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int k, int j, int i, Real &error) {
         Real numer = 0., denom = std::numeric_limits<Real>::min();
         for (int dir=0; dir < d2Scratch.GetDim(4); dir++) {
           numer += std::pow(d2Scratch(0,dir,k,j,i),2);
           denom += std::pow(d2Scratch(1,dir,k,j,i) + refine_filter*d2Scratch(3,k,j,i), 2);
         }
         Real err = numer/denom;
         error = (err > error ? err : error);
      },
      Kokkos::Max<Real>(error));

  error = std::sqrt(error);

  if (error > refine_criteria) return AmrTag::refine;
  if (error < derefine_criteria) return AmrTag::derefine;
  return AmrTag::same;
}

void SetRefinement_(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  pmb->pmr->SetRefinement(CheckAllRefinement(rc));
}

template <>
TaskStatus Tag(MeshBlockData<Real> *rc) {
  PARTHENON_INSTRUMENT
  SetRefinement_(rc);
  return TaskStatus::complete;
}

template <>
TaskStatus Tag(MeshData<Real> *rc) {
  PARTHENON_INSTRUMENT
  for (int i = 0; i < rc->NumBlocks(); i++) {
    SetRefinement_(rc->GetBlockData(i).get());
  }
  return TaskStatus::complete;
}

} // namespace Refinement
} // namespace parthenon

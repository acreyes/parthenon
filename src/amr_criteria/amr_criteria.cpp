//========================================================================================
// (C) (or copyright) 2020-2021. Triad National Security, LLC. All rights reserved.
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
#include "amr_criteria/amr_criteria.hpp"

#include <memory>

#include "amr_criteria/refinement_package.hpp"
#include "coordinates/coordinates.hpp"
#include "globals.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

namespace parthenon {

AMRCriteria::AMRCriteria(ParameterInput *pin, std::string &block_name)
    : comp6(0), comp5(0), comp4(0) {
  field = pin->GetOrAddString(block_name, "field", "NO FIELD WAS SET");
  if (field == "NO FIELD WAS SET") {
    std::cerr << "Error in " << block_name << ": no field set" << std::endl;
    exit(1);
  }
  if (pin->DoesParameterExist(block_name, "tensor_ijk")) {
    auto index = pin->GetVector<int>(block_name, "tensor_ijk");
    PARTHENON_REQUIRE_THROWS(
        index.size() == 3, "tensor_ijk requires three values, e.g. tensor_ijk = 2, 1, 3");
    comp6 = index[0];
    comp5 = index[1];
    comp4 = index[2];
  } else if (pin->DoesParameterExist(block_name, "tensor_ij")) {
    auto index = pin->GetVector<int>(block_name, "tensor_ij");
    PARTHENON_REQUIRE_THROWS(index.size() == 2,
                             "tensor_ij requires two values, e.g. tensor_ij = 2, 1");
    comp5 = index[0];
    comp4 = index[1];
  } else if (pin->DoesParameterExist(block_name, "vector_i")) {
    auto index = pin->GetVector<int>(block_name, "vector_i");
    PARTHENON_REQUIRE_THROWS(index.size() == 1,
                             "vector_i requires one value, e.g. vector_i = 2");
    comp4 = index[0];
  }
  refine_criteria = pin->GetOrAddReal(block_name, "refine_tol", 0.5);
  derefine_criteria = pin->GetOrAddReal(block_name, "derefine_tol", 0.05);
  int global_max_level = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1);
  max_level = pin->GetOrAddInteger(block_name, "max_level", global_max_level);
  if (max_level > global_max_level) {
    std::cerr << "WARNING: max_level in " << block_name
              << " exceeds numlevel (the global maximum number of levels) set in "
                 "<parthenon/mesh>."
              << std::endl
              << std::endl
              << "Setting max_level = numlevel, but this may not be what you want."
              << std::endl
              << std::endl;
    max_level = global_max_level;
  }
}

AMRLoehnerEstimator::AMRLoehnerEstimator(ParameterInput *pin, std::string &block_name)
   : AMRCriteria(pin, block_name) {
      PARTHENON_REQUIRE_THROWS(Globals::nghost > 1, "Loehner estimator needs at least 2 ghost cells")
      refine_filter = pin->GetOrAddReal(block_name, "refine_filter", 0.01);
      const int meshx2 = pin->GetInteger("parthenon/mesh", "nx2") > 1 
                         ? Globals::nghost : 0;
      const int meshx3 = pin->GetInteger("parthenon/mesh", "nx3") > 1 
                         ? Globals::nghost : 0;

      const int nx1 = pin->GetInteger("parthenon/meshblock", "nx1") + 2*Globals::nghost;
      const int nx2 = pin->GetInteger("parthenon/meshblock", "nx2") + 2*meshx2;
      const int nx3 = pin->GetInteger("parthenon/meshblock", "nx3") + 2*meshx3;

      const int ndim = 1 + (meshx2 > 1) + (meshx3 > 1);
      // work array for first derivatives
      d1Scratch = ParArray4D<Real>("d1Scratch", 1 + meshx2 + meshx3, nx3, nx2, nx1);
      // work array for second derivatives & averaging
      d2Scratch = ParArray5D<Real>("d2Scratch", 3, ndim*ndim, nx3, nx2, nx1);

   }

std::shared_ptr<AMRCriteria> AMRCriteria::MakeAMRCriteria(std::string &criteria,
                                                          ParameterInput *pin,
                                                          std::string &block_name) {
  if (criteria == "derivative_order_1")
    return std::make_shared<AMRFirstDerivative>(pin, block_name);
  if (criteria == "derivative_order_2")
    return std::make_shared<AMRSecondDerivative>(pin, block_name);
  if (criteria == "loehner")
    return std::make_shared<AMRLoehnerEstimator>(pin, block_name);
  throw std::invalid_argument("\n  Invalid selection for refinment method in " +
                              block_name + ": " + criteria);
}

AMRBounds AMRCriteria::GetBounds(const MeshBlockData<Real> *rc) const {
  auto ib = rc->GetBoundsI(IndexDomain::interior);
  auto jb = rc->GetBoundsJ(IndexDomain::interior);
  auto kb = rc->GetBoundsK(IndexDomain::interior);
  return AMRBounds(ib, jb, kb);
}

AmrTag AMRFirstDerivative::operator()(const MeshBlockData<Real> *rc) const {
  if (!rc->HasVariable(field) || !rc->IsAllocated(field)) {
    return AmrTag::same;
  }
  auto bnds = GetBounds(rc);
  auto q = Kokkos::subview(rc->Get(field).data, comp6, comp5, comp4, Kokkos::ALL(),
                           Kokkos::ALL(), Kokkos::ALL());
  return Refinement::FirstDerivative(bnds, q, refine_criteria, derefine_criteria);
}

AmrTag AMRSecondDerivative::operator()(const MeshBlockData<Real> *rc) const {
  if (!rc->HasVariable(field) || !rc->IsAllocated(field)) {
    return AmrTag::same;
  }
  auto bnds = GetBounds(rc);
  auto q = Kokkos::subview(rc->Get(field).data, comp6, comp5, comp4, Kokkos::ALL(),
                           Kokkos::ALL(), Kokkos::ALL());
  return Refinement::SecondDerivative(bnds, q, refine_criteria, derefine_criteria);
}

Coordinates_t AMRLoehnerEstimator::GetCoords(const MeshBlockData<Real> *rc) const {
   return rc->GetBlockPointer()->coords;
}

AmrTag AMRLoehnerEstimator::operator()(const MeshBlockData<Real> *rc) const {
  if (!rc->HasVariable(field) || !rc->IsAllocated(field)) {
    return AmrTag::same;
  }
  auto bnds = GetBounds(rc);
  auto coords = GetCoords(rc);
  auto q = Kokkos::subview(rc->Get(field).data, comp6, comp5, comp4, Kokkos::ALL(),
                           Kokkos::ALL(), Kokkos::ALL());
  return Refinement::LoehnerEstimator(bnds, coords, q, d1Scratch, d2Scratch, 
                                      refine_criteria, derefine_criteria, refine_filter);
}
} // namespace parthenon

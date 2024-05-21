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

#include <sstream>
#include <string>

#include <parthenon/package.hpp>

#include "Kokkos_Macros.hpp"
#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "fieldloop_driver.hpp"
#include "fieldloop_package.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "parameter_input.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;
using namespace parthenon;
using TE = parthenon::TopologicalElement;

// *************************************************//
// redefine some weakly linked parthenon functions *//
// *************************************************//

namespace fieldloop_example {

KOKKOS_FORCEINLINE_FUNCTION
Real vectorPotential(const Real x, const Real y, const Real R) {
  const Real r = Kokkos::sqrt(x * x + y * y);
  if (r > R) {
    return 0.;
  }
  return 1.e-3 * (R - r);
}

KOKKOS_FORCEINLINE_FUNCTION
Real vectorPotential(const Real xx, const Real y, const Real zz, const Real R,
                     const Real xOff = 0., const Real zOff = 0.) {
  const Real x = xx - xOff;
  const Real z = zz - zOff;
  const Real d = std::sqrt(((y) * (y) + (z - x) * (z - x) + (-y) * (-y)) / 3.);
  return vectorPotential(d, 0., R);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  using parthenon::MetadataFlag;

  auto &data = pmb->meshblock_data.Get();

  auto pkg = pmb->packages.Get("fieldloop_package");

  const auto &R = pkg->Param<Real>("R");
  const auto tilt = pkg->Param<bool>("tilt");
  if (tilt) {
    PARTHENON_REQUIRE(tilt && (pmb->pmy_mesh->ndim > 2),
                      "tilting axis of field-loop only in 3D")
  }

  auto cellbounds = pmb->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto coords = pmb->coords;
  PackIndexMap index_map, index_mapCC, imapE;
  /* auto edge = data->PackVariables({"edgeE"}, imapE); */
  auto edge = data->PackVariables(
      std::vector<MetadataFlag>({Metadata::Edge, Metadata::Flux}), imapE);
  auto face = data->PackVariables(
      std::vector<MetadataFlag>{Metadata::Independent, Metadata::Face}, index_map);
  auto cc = data->PackVariables(std::vector<MetadataFlag>{Metadata::Cell}, index_mapCC);
  const auto idx_B = index_map["faceB"].first;
  const auto idx_P = index_mapCC["magP"].first;
  const auto idx_Bcc = index_mapCC["B"].first;
  const auto idx_div = index_mapCC["div"].first;

  const int nDim = 1 + (jb.e > jb.s) + (kb.e > kb.s);
  const int K3D = (nDim < 3) ? 0 : 1;
  const int K2D = (nDim < 2) ? 0 : 1;
  const Real idx = 1. / coords.Dxf<1>();
  const Real idy = 1. / coords.Dxf<2>();
  const Real idz = 1. / coords.Dxf<3>();
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e + K3D, jb.s, jb.e + K2D, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (!tilt) {
          const Real Azmm = vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j), R);
          const Real Azmp = vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j + 1), R);
          const Real Azpm = vectorPotential(coords.Xf<1>(i + 1), coords.Xf<2>(j), R);
          const Real Azpp = vectorPotential(coords.Xf<1>(i + 1), coords.Xf<2>(j + 1), R);

          face(TE::F1, idx_B, k, j, i) = idy * (Azmp - Azmm);
          face(TE::F2, idx_B, k, j, i) = -idx * (Azpm - Azmm);
        } else {
          // magnitude of vector potential is the same but we tilt to be pointing along
          // (1,0,1) evaluated at edges of cell ijk goes from 0,0,0 to 1,0,1
          Real Amnm =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k), R);
          Real Amnp =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k + 1), R);
          Real Ammn =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j), coords.Xc<3>(k), R);
          Real Ampn =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j + 1), coords.Xc<3>(k), R);

          Real Apmn =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i + 1), coords.Xf<2>(j), coords.Xc<3>(k), R);
          Real Anmm =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k), R);
          Real Anmp =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k + 1), R);

          Real Anpm =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j + 1), coords.Xf<3>(k), R);
          Real Apnm =
              1. / Kokkos::sqrt(2.) *
              vectorPotential(coords.Xf<1>(i + 1), coords.Xc<2>(j), coords.Xf<3>(k), R);

          // Ay = 0
          // Bx = dyAz - dzAy
          face(TE::F1, idx_B, k, j, i) = idy * (Ampn - Ammn);
          // By = dzAx - dxAz
          face(TE::F2, idx_B, k, j, i) = idz * (Anmp - Anmm) - idx * (Apmn - Ammn);
          // Bz = dxAy - dyAx
          face(TE::F3, idx_B, k, j, i) = -idy * (Anpm - Anmm);

          Amnm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k), R, 2.,
                                 0.);
          Amnp = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k + 1), R,
                                 2., 0.);
          Ammn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j), coords.Xc<3>(k), R, 2.,
                                 0.);
          Ampn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j + 1), coords.Xc<3>(k), R,
                                 2., 0.);

          Apmn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i + 1), coords.Xf<2>(j), coords.Xc<3>(k), R,
                                 2., 0.);
          Anmm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k), R, 2.,
                                 0.);
          Anmp = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k + 1), R,
                                 2., 0.);

          Anpm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j + 1), coords.Xf<3>(k), R,
                                 2., 0.);
          Apnm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i + 1), coords.Xc<2>(j), coords.Xf<3>(k), R,
                                 2., 0.);

          // Ay = 0
          // Bx = dyAz - dzAy
          face(TE::F1, idx_B, k, j, i) += idy * (Ampn - Ammn);
          // By = dzAx - dxAz
          face(TE::F2, idx_B, k, j, i) += idz * (Anmp - Anmm) - idx * (Apmn - Ammn);
          // Bz = dxAy - dyAx
          face(TE::F3, idx_B, k, j, i) += -idy * (Anpm - Anmm);

          Amnm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k), R, 0.,
                                 2.);
          Amnp = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xc<2>(j), coords.Xf<3>(k + 1), R,
                                 0., 2.);
          Ammn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j), coords.Xc<3>(k), R, 0.,
                                 2.);
          Ampn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i), coords.Xf<2>(j + 1), coords.Xc<3>(k), R,
                                 0., 2.);

          Apmn = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i + 1), coords.Xf<2>(j), coords.Xc<3>(k), R,
                                 0., 2.);
          Anmm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k), R, 0.,
                                 2.);
          Anmp = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j), coords.Xf<3>(k + 1), R,
                                 0., 2.);

          Anpm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xc<1>(i), coords.Xf<2>(j + 1), coords.Xf<3>(k), R,
                                 0., 2.);
          Apnm = 1. / Kokkos::sqrt(2.) *
                 vectorPotential(coords.Xf<1>(i + 1), coords.Xc<2>(j), coords.Xf<3>(k), R,
                                 0., 2.);

          // Ay = 0
          // Bx = dyAz - dzAy
          face(TE::F1, idx_B, k, j, i) += idy * (Ampn - Ammn);
          // By = dzAx - dxAz
          face(TE::F2, idx_B, k, j, i) += idz * (Anmp - Anmm) - idx * (Apmn - Ammn);
          // Bz = dxAy - dyAx
          face(TE::F3, idx_B, k, j, i) += -idy * (Anpm - Anmm);
        }
      });

  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real Bx =
            0.5 * (face(TE::F1, idx_B, k, j, i + 1) + face(TE::F1, idx_B, k, j, i));
        const Real By =
            0.5 * (face(TE::F2, idx_B, k, j + 1, i) + face(TE::F2, idx_B, k, j, i));
        Real Bz;
        if (nDim > 2) {
          Bz = 0.5 * (face(TE::F3, idx_B, k + 1, j, i) + face(TE::F3, idx_B, k, j, i));
        } else {
          Bz = cc(idx_Bcc + 2, k, j, i);
        }
        cc(idx_Bcc, k, j, i) = Bx;
        cc(idx_Bcc + 1, k, j, i) = By;
        cc(idx_Bcc + 2, k, j, i) = Bz;
        cc(idx_P, k, j, i) = 0.5 * (Bx * Bx + By * By + Bz * Bz);
      });
}

Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  auto pkg = fieldloop_package::Initialize(pin.get());
  packages.Add(pkg);

  auto app = std::make_shared<StateDescriptor>("fieldloop_app");
  packages.Add(app);

  return packages;
}

} // namespace fieldloop_example

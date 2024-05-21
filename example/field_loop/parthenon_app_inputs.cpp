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

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin, SimTime &tm)
//  \brief Compute L1 error in fieldloop test and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, SimTime &tm) {
  if (!pin->GetOrAddBoolean("FieldLoop", "compute_error", false)) return;

  // Initialize errors to zero
  Real l1_err = 0.0;
  Real max_err = 0.0;

  for (auto &pmb : mesh->block_list) {
    auto pkg = pmb->packages.Get("fieldloop_package");

    auto rc = pmb->meshblock_data.Get(); // get base container
    const auto &amp = pkg->Param<Real>("amp");
    const auto &vel = pkg->Param<Real>("vel");
    const auto &k_par = pkg->Param<Real>("k_par");
    const auto &cos_a2 = pkg->Param<Real>("cos_a2");
    const auto &cos_a3 = pkg->Param<Real>("cos_a3");
    const auto &sin_a2 = pkg->Param<Real>("sin_a2");
    const auto &sin_a3 = pkg->Param<Real>("sin_a3");
    const auto &profile = pkg->Param<std::string>("profile");

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // calculate error on host
    auto q = rc->Get("advected").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real ref_val;
          if (profile == "wave") {
            Real x =
                cos_a2 * (pmb->coords.Xc<1>(i) * cos_a3 + pmb->coords.Xc<2>(j) * sin_a3) +
                pmb->coords.Xc<3>(k) * sin_a2;
            Real sn = std::sin(k_par * x);
            ref_val = 1.0 + amp * sn * vel;
          } else if (profile == "smooth_gaussian") {
            Real rsq = pmb->coords.Xc<1>(i) * pmb->coords.Xc<1>(i) +
                       pmb->coords.Xc<2>(j) * pmb->coords.Xc<2>(j) +
                       pmb->coords.Xc<3>(k) * pmb->coords.Xc<3>(k);
            ref_val = 1. + amp * exp(-100.0 * rsq);
          } else if (profile == "hard_sphere") {
            Real rsq = pmb->coords.Xc<1>(i) * pmb->coords.Xc<1>(i) +
                       pmb->coords.Xc<2>(j) * pmb->coords.Xc<2>(j) +
                       pmb->coords.Xc<3>(k) * pmb->coords.Xc<3>(k);
            ref_val = (rsq < 0.15 * 0.15 ? 1.0 : 0.0);
          } else {
            ref_val = 1e9; // use an artificially large error
          }

          // Weight l1 error by cell volume
          Real vol = pmb->coords.CellVolume(k, j, i);

          l1_err += std::abs(ref_val - q(k, j, i)) * vol;
          max_err = std::max(static_cast<Real>(std::abs(ref_val - q(k, j, i))), max_err);
        }
      }
    }
  }

  Real max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &l1_err, 1, MPI_PARTHENON_REAL, MPI_SUM,
                                   0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &max_err, 1, MPI_PARTHENON_REAL, MPI_MAX,
                                   0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(
        MPI_Reduce(&l1_err, &l1_err, 1, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&max_err, &max_err, 1, MPI_PARTHENON_REAL, MPI_MAX, 0,
                                   MPI_COMM_WORLD));
  }
#endif

  // only the root process outputs the data
  if (Globals::my_rank == 0) {
    // normalize errors by number of cells
    auto mesh_size = mesh->mesh_size;
    Real vol = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR)) *
               (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR)) *
               (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
    l1_err /= vol;
    // compute rms error
    max_max_over_l1 = std::max(max_max_over_l1, (max_err / l1_err));

    // open output file and write out errors
    std::string fname;
    fname.assign("fieldloop-errors.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg.str().c_str());
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg.str().c_str());
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      std::fprintf(pfile, "L1 max_error/L1  max_error ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx(X1DIR), mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %d  %d", mesh_size.nx(X3DIR), tm.ncycle);
    std::fprintf(pfile, "  %e ", l1_err);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

void UserMeshWorkBeforeOutput(Mesh *mesh, ParameterInput *pin, SimTime const &) {
  // loop over blocks
  for (auto &pmb : mesh->block_list) {
    auto rc = pmb->meshblock_data.Get(); // get base container
    auto pkg = pmb->packages.Get("fieldloop_package");

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    PackIndexMap index_map;
    auto face = rc->PackVariables(
        std::vector<MetadataFlag>{Metadata::Independent, Metadata::Face}, index_map);
    const auto idx_B = index_map["faceB"].first;
    auto data = rc->Get("B").data;
    auto divB = rc->Get("div").data;
    auto coords = pmb->coords;
    const Real idx = 1. / coords.Dxf<1>();
    const Real idy = 1. / coords.Dxf<2>();
    const Real idz = 1. / coords.Dxf<3>();
    const int nDim = pmb->pmy_mesh->ndim;

    pmb->par_for(
        "FieldLoop::FillDerived", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          divB(0, k, j, i) =
              idx * (face(TE::F1, idx_B, k, j, i + 1) - face(TE::F1, idx_B, k, j, i)) +
              idy * (face(TE::F2, idx_B, k, j + 1, i) - face(TE::F2, idx_B, k, j, i));
          if (nDim > 2)
            divB(0, k, j, i) +=
                idz * (face(TE::F3, idx_B, k + 1, j, i) - face(TE::F3, idx_B, k, j, i));
        });
  }
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

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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <coordinates/coordinates.hpp>
#include <globals.hpp>
#include <parthenon/package.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "fieldloop_package.hpp"
#include "kokkos_abstraction.hpp"
#include "outputs/outputs.hpp"
#include "prolong_restrict/pr_divfree.hpp"
#include "reconstruct/dc_inline.hpp"
#include "utils/error_checking.hpp"
#include "utils/instrument.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

// *************************************************//
// define the "physics" package Advect, which      *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace fieldloop_package {
using parthenon::UserHistoryOperation;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("fieldloop_package");

  Real cfl = pin->GetOrAddReal("fieldloop", "cfl", 0.45);
  pkg->AddParam<>("cfl", cfl);
  bool tilt = pin->GetOrAddBoolean("fieldloop", "tilt", false);
  pkg->AddParam<>("tilt", tilt);

  Real refine_tol = pin->GetOrAddReal("fieldloop", "refine_tol", 0.3);
  pkg->AddParam<>("refine_tol", refine_tol);
  Real derefine_tol = pin->GetOrAddReal("fieldloop", "derefine_tol", 0.03);
  pkg->AddParam<>("derefine_tol", derefine_tol);

  Real vx = pin->GetOrAddReal("fieldloop", "vx", 1.0);
  Real vy = pin->GetOrAddReal("fieldloop", "vy", 1.0);
  Real vz = pin->GetOrAddReal("fieldloop", "vz", 0.0);
  Real R = pin->GetOrAddReal("fieldloop", "radius", 0.3);

  pkg->AddParam<>("R", R);
  pkg->AddParam<>("vx", vx);
  pkg->AddParam<>("vy", vy);
  pkg->AddParam<>("vz", vz);
  pkg->AddParam<>("divergence", 0.);

  // Give a custom labels to advected in the data output
  Metadata m;
  m = Metadata({Metadata::Edge, Metadata::Flux, Metadata::Derived});
  pkg->AddField(std::string("edgeE"), m);

  m = Metadata(
      {Metadata::Face, Metadata::Independent, Metadata::WithFluxes, Metadata::FillGhost});
  m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedMinMod,
                          parthenon::refinement_ops::RestrictAverage,
                          parthenon::refinement_ops::ProlongateInternalBalsara>();
  pkg->AddField(std::string("faceB"), m);

  m = Metadata(
      {Metadata::Cell, Metadata::Independent, Metadata::WithFluxes, Metadata::FillGhost},
      std::vector<int>({3}), std::vector<std::string>{"Bx", "By", "Bz"});
  pkg->AddField("B", m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({1}), std::vector<std::string>{"divB"});
  pkg->AddField("div", m);

  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({1}), std::vector<std::string>{"magP"});
  pkg->AddField("magP", m);

  pkg->FillDerivedBlock = magP;
  pkg->EstimateTimestepBlock = EstimateTimestepBlock;
  pkg->UserWorkBeforeLoopMesh = FieldLoopGreetings;

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
           UserHistoryOperation::sum, FieldLoopHst<Kokkos::Sum<Real, HostExecSpace>>,
           "total_divergence"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
           UserHistoryOperation::max, FieldLoopHst<Kokkos::Max<Real, HostExecSpace>>,
           "max_divergence"));

  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return pkg;
}

void FieldLoopGreetings(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Hello from the field-loop fieldloop package in the field-loop "
                 "fieldloop example!\n"
              << "This run is a restart: " << pmesh->is_restart << "\n"
              << std::endl;
  }
}

template <typename T>
Real FieldLoopHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Packing variable over MeshBlock as the function is called for MeshData, i.e., a
  // collection of blocks
  const auto &div_pack = md->PackVariables(std::vector<std::string>{"div"});

  Real result = 0.0;
  T reducer(result);

  // We choose to apply volume weighting when using the sum reduction.
  // Downstream this choice will be done on a variable by variable basis and volume
  // weighting needs to be applied in the reduction region.
  const bool volume_weighting = std::is_same<T, Kokkos::Sum<Real, HostExecSpace>>::value;
  const int nDim = pmb->pmy_mesh->ndim;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, PARTHENON_AUTO_LABEL, DevExecSpace(), 0,
      div_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lresult) {
        const auto &coords = div_pack.GetCoords(b);
        // `join` is a function of the Kokkos::ReducerConecpt that allows to use the same
        // call for different reductions
        const Real vol = volume_weighting ? coords.CellVolume(k, j, i) : 1.0;
        reducer.join(lresult, Kokkos::abs(div_pack(b, 0, k, j, i)) * vol);
      },
      reducer);

  return result;
}
// this is the package registered function to fill derived
void magP(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  // packing in principle unnecessary/convoluted here and just done for demonstration
  std::vector<std::string> vars({"magP", "B"});
  PackIndexMap imap;
  const auto &v = rc->PackVariables(vars, imap);

  const int iB = imap.get("B").first;
  const int iP = imap.get("magP").first;
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        /* v(out + n, k, j, i) = v(in + n, k, j, i) * v(in + n, k, j, i); */
        v(iP, k, j, i) = 0.5 * (v(iP, k, j, i) * v(iP, k, j, i) +
                                v(iP + 1, k, j, i) * v(iP + 1, k, j, i) +
                                v(iP + 2, k, j, i) * v(iP + 2, k, j, i));
      });
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestepBlock(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages.Get("fieldloop_package");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;

  // this is obviously overkill for this constant velocity problem
  Real min_dt;
  pmb->par_reduce(
      PARTHENON_AUTO_LABEL, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmin_dt) {
        if (vx != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X1DIR>(k, j, i) / std::abs(vx));
        if (vy != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X2DIR>(k, j, i) / std::abs(vy));
        if (vz != 0.0)
          lmin_dt = std::min(lmin_dt, coords.Dxc<X3DIR>(k, j, i) / std::abs(vz));
      },
      Kokkos::Min<Real>(min_dt));

  return cfl * min_dt;
}

// Compute fluxes at faces given the constant velocity field and
// upwinds the fluxes to get the "emf" at the edges
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(std::shared_ptr<MeshBlockData<Real>> &rc) {
  using parthenon::MetadataFlag;

  PARTHENON_INSTRUMENT
  auto pmb = rc->GetBlockPointer();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int K3D = (kb.e > kb.s);
  const int K2D = (jb.e > jb.s);

  auto pkg = pmb->packages.Get("fieldloop_package");
  const auto &vx = pkg->Param<Real>("vx");
  const auto &vy = pkg->Param<Real>("vy");
  const auto &vz = pkg->Param<Real>("vz");

  PackIndexMap index_map;
  auto face = rc->PackVariablesAndFluxes({"faceB"}, index_map);

  const auto idx_face = index_map["faceB"].first;

  PackIndexMap index_mapCC;
  auto cc = rc->PackVariablesAndFluxes(
      std::vector<MetadataFlag>{Metadata::Cell, Metadata::WithFluxes}, index_mapCC);
  const auto idx_Bx = index_mapCC["B"].first;
  const int n = idx_Bx;

  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  const int nvar = cc.GetDim(4);
  size_t scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nvar, nx1);
  // get x-fluxes
  pmb->par_for_outer(
      PARTHENON_AUTO_LABEL, 2 * scratch_size_in_bytes, scratch_level, kb.s - K3D,
      kb.e + K3D, jb.s - K2D, jb.e + K2D,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
        parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
        // get reconstructed state on faces
        parthenon::DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, cc, ql, qr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        par_for_inner(member, ib.s - 1, ib.e + 1, [&](const int i) {
          // Fx_Bx = 0
          // Fx_By =  Ez =  uxBy - uyBx
          // Fx_Bz =  Ey =  uzBx - uxBz
          cc.flux(X1DIR, n, k, j, i) = 0.;
          if (vx > 0.0) {
            cc.flux(X1DIR, n + 1, k, j, i) = ql(n + 1, i) * vx;
            cc.flux(X1DIR, n + 2, k, j, i) = ql(n + 2, i) * vx;
          } else {
            cc.flux(X1DIR, n + 1, k, j, i) = qr(n + 1, i) * vx;
            cc.flux(X1DIR, n + 2, k, j, i) = qr(n + 2, i) * vx;
          }
          cc.flux(X1DIR, n + 1, k, j, i) -= vy * face(TE::F1, 0, k, j, i);
          cc.flux(X1DIR, n + 2, k, j, i) -= vz * face(TE::F1, 0, k, j, i);
        });
      });

  // get y-fluxes
  if (pmb->pmy_mesh->ndim >= 2) {
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 3 * scratch_size_in_bytes, scratch_level, kb.s - K3D,
        kb.e + K3D, jb.s - K2D, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // the overall algorithm/use of scratch pad here is clear inefficient and kept
          // just for demonstrating purposes. The key point is that we cannot reuse
          // reconstructed arrays for different `j` with `j` being part of the outer
          // loop given that this loop can be handled by multiple threads simultaneously.

          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // get reconstructed state on faces
          parthenon::DonorCellX2(member, k, j - 1, ib.s - 1, ib.e + 1, cc, ql, q_unused);
          parthenon::DonorCellX2(member, k, j, ib.s - 1, ib.e + 1, cc, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          par_for_inner(member, ib.s - 1, ib.e + 1, [&](const int i) {
            // Fy_Bx = Ez = uxBy - uyBx
            // Fy_By = 0
            // Fy_Bz =-Ex =-uyBz + uzBy
            cc.flux(X2DIR, n + 1, k, j, i) = 0.;
            if (vy > 0.0) {
              cc.flux(X2DIR, n, k, j, i) = ql(n, i) * vy;
              cc.flux(X2DIR, n + 2, k, j, i) = ql(n + 2, i) * vy;
            } else {
              cc.flux(X2DIR, n, k, j, i) = qr(n, i) * vy;
              cc.flux(X2DIR, n + 2, k, j, i) = qr(n + 2, i) * vy;
            }
            cc.flux(X2DIR, n, k, j, i) -= vx * face(TE::F2, 0, k, j, i);
            cc.flux(X2DIR, n + 2, k, j, i) -= vz * face(TE::F2, 0, k, j, i);
          });
        });
  }

  // get z-fluxes
  if (pmb->pmy_mesh->ndim == 3) {
    pmb->par_for_outer(
        PARTHENON_AUTO_LABEL, 3 * scratch_size_in_bytes, scratch_level, kb.s - 1,
        kb.e + 1, jb.s - 1, jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          // the overall algorithm/use of scratch pad here is clear inefficient and kept
          // just for demonstrating purposes. The key point is that we cannot reuse
          // reconstructed arrays for different `j` with `j` being part of the outer
          // loop given that this loop can be handled by multiple threads simultaneously.

          parthenon::ScratchPad2D<Real> ql(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> qr(member.team_scratch(scratch_level), nvar, nx1);
          parthenon::ScratchPad2D<Real> q_unused(member.team_scratch(scratch_level), nvar,
                                                 nx1);
          // get reconstructed state on faces
          parthenon::DonorCellX3(member, k - 1, j, ib.s - 1, ib.e + 1, cc, ql, q_unused);
          parthenon::DonorCellX3(member, k, j, ib.s - 1, ib.e + 1, cc, q_unused, qr);
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          const int n = idx_Bx;
          par_for_inner(member, ib.s - 1, ib.e + 1, [&](const int i) {
            // Fz_Bx =-Ey = uzBx - uxBz
            // Fz_By = Ex = uzBy - uyBz
            // Fz_Bz = 0
            if (vz > 0.0) {
              cc.flux(X3DIR, n, k, j, i) = ql(n, i) * vz;
              cc.flux(X3DIR, n + 1, k, j, i) = ql(n + 1, i) * vz;
            } else {
              cc.flux(X3DIR, n, k, j, i) = qr(n, i) * vz;
              cc.flux(X3DIR, n + 1, k, j, i) = qr(n + 1, i) * vz;
            }
            cc.flux(X3DIR, n, k, j, i) -= vx * face(TE::F3, 0, k, j, i);
            cc.flux(X3DIR, n + 1, k, j, i) -= vz * face(TE::F3, 0, k, j, i);
          });
        });
  }

  Kokkos::fence();
  // get EMF on cell edges
  const int nDim = pmb->pmy_mesh->ndim;
  using TT = parthenon::TopologicalType;
  pmb->par_for(
      PARTHENON_AUTO_LABEL, kb.s, kb.e + K3D, jb.s, jb.e + K2D, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Ez =-Fx_By = Fy_Bx
        face.flux<TT::Edge>(X3DIR, 0, k, j, i) = (vy > 0.)
                                                     ? -cc.flux(X1DIR, n + 1, k, j - 1, i)
                                                     : -cc.flux(X1DIR, n + 1, k, j, i);
        face.flux<TT::Edge>(X3DIR, 0, k, j, i) +=
            (vx > 0.) ? cc.flux(X2DIR, n, k, j, i - 1) : cc.flux(X2DIR, n, k, j, i);
        face.flux<TT::Edge>(X3DIR, 0, k, j, i) *= 0.5;

        if (nDim >= 3) {
          // Ex = Fz_By = -Fy_Bz
          // Ey =-Fz_Bx =  Fx_Bz
          face.flux<TT::Edge>(X1DIR, 0, k, j, i) =
              (vz > 0.) ? -cc.flux(X2DIR, n + 2, k - 1, j, i)
                        : -cc.flux(X2DIR, n + 2, k, j, i);
          face.flux<TT::Edge>(X1DIR, 0, k, j, i) +=
              (vy > 0.) ? cc.flux(X3DIR, n + 1, k, j - 1, i)
                        : cc.flux(X3DIR, n + 1, k, j, i);
          face.flux<TT::Edge>(X1DIR, 0, k, j, i) *= 0.5;

          face.flux<TT::Edge>(X2DIR, 0, k, j, i) =
              (vx > 0.) ? -cc.flux(X3DIR, n, k, j, i - 1) : -cc.flux(X3DIR, n, k, j, i);
          face.flux<TT::Edge>(X2DIR, 0, k, j, i) +=
              (vz > 0.) ? cc.flux(X1DIR, n + 2, k - 1, j, i)
                        : cc.flux(X1DIR, n + 2, k, j, i);
          face.flux<TT::Edge>(X2DIR, 0, k, j, i) *= 0.5;
        }
      });

  return TaskStatus::complete;
}

template <>
TaskStatus StaggeredUpdate(MeshData<Real> *base, MeshData<Real> *in, const Real beta,
                           const Real dt, MeshData<Real> *out) {
  using TT = parthenon::TopologicalType;

  const IndexDomain interior = IndexDomain::interior;
  PackIndexMap fid_base, fid_in, fid_out;
  const auto &fbase = base->PackVariables(std::vector<std::string>({"faceB"}), fid_base);
  const auto &fin =
      in->PackVariablesAndFluxes(std::vector<std::string>({"faceB"}), fid_in);
  auto fout = out->PackVariables(std::vector<std::string>({"faceB"}), fid_out);

  PackIndexMap cid_base, cid_in, cid_out;
  const auto &Bbase = base->PackVariables(std::vector<std::string>({"B"}), cid_base);
  const auto &Bin = in->PackVariablesAndFluxes(std::vector<std::string>({"B"}), cid_in);
  auto Bout = out->PackVariables(std::vector<std::string>({"B", "div"}), cid_out);

  const auto idB = cid_out["B"].first;
  const auto idfB = fid_out["faceB"].first;
  const auto idiv = cid_out["div"].first;

  const IndexRange ib = in->GetBoundsI(interior);
  const IndexRange jb = in->GetBoundsJ(interior);
  const IndexRange kb = in->GetBoundsK(interior);

  const int ndim = fin.GetNdim();
  const int K3D = (kb.e > kb.s);
  const int K2D = (jb.e > jb.s);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, fin.GetDim(5) - 1,
      kb.s, kb.e + K3D, jb.s, jb.e + K2D, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        auto &coords = fin.GetCoords(m);
        const Real dtidx = beta * dt / coords.Dxf<1>();
        const Real dtidy = beta * dt / coords.Dxf<2>();
        const auto &fin_block = fin(m);
        fout(m, TE::F1, idfB, k, j, i) =
            beta * fbase(m, TE::F1, idfB, k, j, i) +
            (1. - beta) * fin_block(TE::F1, idfB, k, j, i) -
            dtidy * (fin_block.flux<TT::Edge>(X3DIR, idfB, k, j + 1, i) -
                     fin_block.flux<TT::Edge>(X3DIR, idfB, k, j, i));
        fout(m, TE::F2, idfB, k, j, i) =
            beta * fbase(m, TE::F2, idfB, k, j, i) +
            (1. - beta) * fin_block(TE::F2, idfB, k, j, i) -
            dtidx * (fin_block.flux<TT::Edge>(X3DIR, idfB, k, j, i) -
                     fin_block.flux<TT::Edge>(X3DIR, idfB, k, j, i + 1));

        if (ndim < 3) {
          const auto &Bin_block = Bin(m);
          Bout(m, idB + 2, k, j, i) =
              beta * Bbase(m, idB + 2, k, j, i) +
              (1. - beta) * Bin_block(idB + 2, k, j, i) -
              dtidx * (Bin_block.flux(X1DIR, idfB + 2, k, j, i + 1) -
                       Bin_block.flux(X1DIR, idfB + 2, k, j, i)) -
              dtidy * (Bin_block.flux(X2DIR, idfB + 2, k, j + 1, i) -
                       Bin_block.flux(X2DIR, idfB + 2, k, j, i));
        } else {
          const auto &Bin_block = Bin(m);
          const Real dtidz = beta * dt / coords.Dxf<3>();
          fout(m, TE::F1, idfB, k, j, i) -=
              dtidz * (fin_block.flux<TT::Edge>(X2DIR, idfB, k, j, i) -
                       fin_block.flux<TT::Edge>(X2DIR, idfB, k + 1, j, i));
          fout(m, TE::F2, idfB, k, j, i) -=
              dtidz * (fin_block.flux<TT::Edge>(X1DIR, idfB, k + 1, j, i) -
                       fin_block.flux<TT::Edge>(X1DIR, idfB, k, j, i));
          fout(m, TE::F3, idfB, k, j, i) =
              beta * fbase(m, TE::F3, idfB, k, j, i) +
              (1. - beta) * fin_block(TE::F3, idfB, k, j, i) -
              dtidy * (fin_block.flux<TT::Edge>(X1DIR, idfB, k, j, i) -
                       fin_block.flux<TT::Edge>(X1DIR, idfB, k, j + 1, i)) -
              dtidx * (fin_block.flux<TT::Edge>(X2DIR, idfB, k, j, i + 1) -
                       fin_block.flux<TT::Edge>(X2DIR, idfB, k, j, i));
        }
      });

  // average face-fields back down to CC
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, PARTHENON_AUTO_LABEL, DevExecSpace(), 0, fin.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        auto &coords = fout.GetCoords(m);

        Bout(m, idB, k, j, i) =
            0.5 * (fout(m, TE::F1, idfB, k, j, i + 1) + fout(m, TE::F1, idfB, k, j, i));
        Bout(m, idB + 1, k, j, i) =
            0.5 * (fout(m, TE::F2, idfB, k, j + 1, i) + fout(m, TE::F2, idfB, k, j, i));
        
        Bout(m, idiv, k, j, i) =
            1./coords.Dxf<1>() * (fout(m, TE::F1, idfB, k, j, i + 1) - fout(m, TE::F1, idfB, k, j, i)) +
            1./coords.Dxf<2>() * (fout(m, TE::F2, idfB, k, j + 1, i) - fout(m, TE::F2, idfB, k, j, i));
        if (ndim > 2) {
           Bout(m, idB + 2, k, j, i) =
            0.5 * (fout(m, TE::F3, idfB, k + 1, j, i) + fout(m, TE::F3, idfB, k, j, i));
           Bout(m, idiv, k, j, i) +=
            1./coords.Dxf<3>() * (fout(m, TE::F3, idfB, k + 1, j, i) - fout(m, TE::F3, idfB, k, j, i));
        }
      });

  return TaskStatus::complete;
}

} // namespace fieldloop_package

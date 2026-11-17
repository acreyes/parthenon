//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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

#ifndef BVALS_COMMS_BND_OPS_HPP_
#define BVALS_COMMS_BND_OPS_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "basic_types.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "kokkos_abstraction.hpp"

namespace parthenon {
template <typename T>
class MeshData;

namespace boundary_ops {
// Tags selecting Parthenon's default copy for one side of the exchange.
struct DefaultPack {};
struct DefaultUnpack {};
} // namespace boundary_ops

// Indices into the BndInfo arrays of one bucket (all boundaries of a cache that use the
// same registered op). See RegisterBoundaryOps in metadata.hpp for the order.
using BndOpIdx_t = ParArray1D<std::size_t>;
using BndOpIdxHost_t = typename BndOpIdx_t::host_mirror_type;

// A registered pack or unpack op. Runs on host once per bucket and launches its own
// kernels. Arguments: the MeshData being exchanged, the cache's BndInfo array (device
// and host), the bucket's indices into it (device and host) and the bucket size. A
// bucket holds same-level boundaries only; boundaries to coarser or finer neighbors take
// the default copy and the variable's prolongation/restriction ops. Ready-made ops are
// in bvals/comms/bnd_ops_library.hpp.
using BoundaryOp_t =
    std::function<void(MeshData<Real> *, const BndInfoArr_t &, const BndInfoArrHost_t &,
                       const BndOpIdx_t &, const BndOpIdxHost_t &, std::size_t)>;

// Type-erased pack/unpack ops of a variable, identified (and hashed) by the typeids of
// the registered op types, like refinement::RefinementFunctions_t. An empty op means the
// default copy on that side.
struct BoundaryFunctions_t {
  BoundaryFunctions_t() = default;
  explicit BoundaryFunctions_t(const std::string &label) : label_(label) {}

  template <class UnpackOp, class PackOp = boundary_ops::DefaultPack>
  static BoundaryFunctions_t RegisterOps() {
    BoundaryFunctions_t funcs(std::string(typeid(UnpackOp).name()) + " and " +
                              std::string(typeid(PackOp).name()));
    if constexpr (!std::is_same_v<UnpackOp, boundary_ops::DefaultUnpack>) {
      funcs.unpack = [](MeshData<Real> *md, const BndInfoArr_t &info,
                        const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                        const BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
        UnpackOp::Unpack(md, info, info_h, idx, idx_h, nbuffers);
      };
    }
    if constexpr (!std::is_same_v<PackOp, boundary_ops::DefaultPack>) {
      funcs.pack = [](MeshData<Real> *md, const BndInfoArr_t &info,
                      const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                      const BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
        PackOp::Pack(md, info, info_h, idx, idx_h, nbuffers);
      };
    }
    return funcs;
  }

  const std::string &label() const { return label_; }
  bool HasPack() const { return static_cast<bool>(pack); }
  bool HasUnpack() const { return static_cast<bool>(unpack); }
  bool operator==(const BoundaryFunctions_t &other) const {
    return label() == other.label();
  }

  BoundaryOp_t pack;
  BoundaryOp_t unpack;

 private:
  std::string label_;
};

// Type-erased flux-correction combine op of a flux variable (see
// Metadata::RegisterFluxCorrectionOps), identified (and hashed) by the typeid of the op.
// The op runs in SetBounds<flxcor_recv> on the same-level flux-correction boundaries of
// the variable, in place of the default overwrite.
struct FluxCorrectionFunctions_t {
  FluxCorrectionFunctions_t() = default;
  explicit FluxCorrectionFunctions_t(const std::string &label) : label_(label) {}

  template <class CombineOp>
  static FluxCorrectionFunctions_t RegisterOps() {
    FluxCorrectionFunctions_t funcs(typeid(CombineOp).name());
    funcs.combine = [](MeshData<Real> *md, const BndInfoArr_t &info,
                       const BndInfoArrHost_t &info_h, const BndOpIdx_t &idx,
                       const BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
      CombineOp::Combine(md, info, info_h, idx, idx_h, nbuffers);
    };
    return funcs;
  }

  const std::string &label() const { return label_; }
  bool operator==(const FluxCorrectionFunctions_t &other) const {
    return label() == other.label();
  }

  BoundaryOp_t combine;

 private:
  std::string label_;
};

// Hashes any of the op sets above by their label
struct BoundaryFunctionsHasher {
  template <class Funcs>
  auto operator()(const Funcs &f) const {
    return std::hash<std::string>{}(f.label());
  }
};

} // namespace parthenon

#endif // BVALS_COMMS_BND_OPS_HPP_

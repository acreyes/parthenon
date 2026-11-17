//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
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
// This file was made in part with generative AI.

#ifndef BVALS_COMMS_BND_INFO_HPP_
#define BVALS_COMMS_BND_INFO_HPP_

#include <memory>
#include <string>
#include <vector>

#include "basic_types.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable_state.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/forest/logical_coordinate_transformation.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/indexer.hpp"
#include "utils/object_pool.hpp"

namespace parthenon {

template <typename T>
class MeshData;
class IndexRange;
class NeighborBlock;
template <typename T>
class Variable;

enum class IndexRangeType {
  BoundaryInteriorSend,
  BoundaryExteriorRecv,
  InteriorSend,
  InteriorRecv
};

struct BndInfo {
  int ntopological_elements = 1;
  using TE = TopologicalElement;
  TE topo_idx[3]{TE::CC, TE::CC, TE::CC};
  SpatiallyMaskedIndexer6D idxer[3];
  forest::LogicalCoordinateTransformation lcoord_trans;

  // Number of points contained in this boundary region
  KOKKOS_FORCEINLINE_FUNCTION
  std::size_t size() const {
    std::size_t s = 0;
    for (int n = 0; n < ntopological_elements; ++n) {
      s += idxer[n].size();
    }
    return s;
  }

  CoordinateDirection dir{CoordinateDirection::X0DIR};
  bool allocated = true;
  bool buf_allocated = true;
  int alloc_status;
  bool same_to_same = false;

  buf_pool_t<Real>::weak_t buf;        // comm buffer from pool
  ParArrayND<Real, VariableState> var; // data variable used for comms
  Coordinates_t coords;

  // Which exchange this is, for registered boundary ops: index of the local block in the
  // MeshData (the sender in send caches, the receiver in receive caches), the gids of
  // both ends and the offset of the neighbor in the local block's frame.
  int block = -1;
  int send_gid = -1;
  int recv_gid = -1;
  CellCentOffsets offsets{0, 0, 0};
  // Provider lanes only: the channel ids of the exchange on the local block and on the
  // neighbor (offsets are zero there)
  int channel = -1;
  int nb_channel = -1;

  KOKKOS_DEFAULTED_FUNCTION
  BndInfo() = default;
  KOKKOS_DEFAULTED_FUNCTION
  BndInfo(const BndInfo &) = default;
  BndInfo(MeshBlock *pmb, const NeighborBlock &nb, std::shared_ptr<Variable<Real>> v,
          CommBuffer<buf_pool_t<Real>::owner_t> *combuf, IndexRangeType idx_range_type);
  KOKKOS_DEFAULTED_FUNCTION
  ~BndInfo() = default;

  // These are are used to generate the BndInfo struct for various
  // kinds of boundary types and operations.
  static BndInfo GetSendBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                std::shared_ptr<Variable<Real>> v,
                                CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetSetBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                               std::shared_ptr<Variable<Real>> v,
                               CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  // Provider lanes: buffer and variable only, no index set (ops do the packing)
  static BndInfo GetLaneSendBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                    std::shared_ptr<Variable<Real>> v,
                                    CommBuffer<buf_pool_t<Real>::owner_t> *buf);
  static BndInfo GetLaneSetBndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                   std::shared_ptr<Variable<Real>> v,
                                   CommBuffer<buf_pool_t<Real>::owner_t> *buf);
};

struct ProResInfo {
  int ntopological_elements = 1;
  // Has to be large enough to allow for maximum integer
  // conversion of TopologicalElements
  bool include_el[10]{false, false, false, false, false,
                      false, false, false, false, false};
  KOKKOS_FORCEINLINE_FUNCTION
  bool &IncludeTopoEl(TopologicalElement te) { return include_el[static_cast<int>(te)]; }
  KOKKOS_FORCEINLINE_FUNCTION
  bool IncludeTopoEl(TopologicalElement te) const {
    return include_el[static_cast<int>(te)];
  }
  SpatiallyMaskedIndexer6D idxer[10];

  CoordinateDirection dir{CoordinateDirection::X0DIR};
  bool allocated = true;
  int alloc_status;
  RefinementOp_t refinement_op = RefinementOp_t::None;
  Coordinates_t coords, coarse_coords; // coords

  ParArrayND<Real, VariableState> fine, coarse;
  KOKKOS_DEFAULTED_FUNCTION
  ProResInfo() = default;
  KOKKOS_DEFAULTED_FUNCTION
  ProResInfo(const ProResInfo &) = default;
  ProResInfo(MeshBlock *pmb, const NeighborBlock &nb, std::shared_ptr<Variable<Real>> v);
  // These are are used to generate the BndInfo struct for various
  // kinds of boundary types and operations.
  static ProResInfo GetNull(MeshBlock *pmb, const NeighborBlock &nb,
                            std::shared_ptr<Variable<Real>> v) {
    return ProResInfo();
  }
  static ProResInfo GetSend(MeshBlock *pmb, const NeighborBlock &nb,
                            std::shared_ptr<Variable<Real>> v);
  static ProResInfo GetSet(MeshBlock *pmb, const NeighborBlock &nb,
                           std::shared_ptr<Variable<Real>> v);
  static ProResInfo GetInteriorProlongate(MeshBlock *pmb, const NeighborBlock &nb,
                                          std::shared_ptr<Variable<Real>> v);
  static ProResInfo GetInteriorRestrict(MeshBlock *pmb, const NeighborBlock &nb,
                                        std::shared_ptr<Variable<Real>> v);
};

int GetBufferSize(const MeshBlock *const pmb, const NeighborBlock &nb,
                  std::shared_ptr<Variable<Real>> v);
// Buffer size on a provider lane, from the registered pack op's BufferSize
int GetLaneBufferSize(const MeshBlock *const pmb, const NeighborBlock &nb,
                      std::shared_ptr<Variable<Real>> v, bool flux);

using BndInfoArr_t = ParArray1DRaw<BndInfo>;
using BndInfoArrHost_t = typename BndInfoArr_t::host_mirror_type;

using ProResInfoArr_t = ParArray1DRaw<ProResInfo>;
using ProResInfoArrHost_t = typename ProResInfoArr_t::host_mirror_type;
class StateDescriptor;
struct ProResCache_t {
  ProResInfoArr_t prores_info{};
  ProResInfoArr_t::host_mirror_type prores_info_h{};
  std::vector<std::size_t> buffer_subset_sizes;
  ParArray2D<std::size_t> buffer_subsets{};
  ParArray2D<std::size_t>::host_mirror_type buffer_subsets_h{};

  void clear() {
    prores_info = ProResInfoArr_t{};
    prores_info_h = ProResInfoArr_t::host_mirror_type{};
    buffer_subset_sizes.clear();
    buffer_subsets = ParArray2D<std::size_t>{};
    buffer_subsets_h = ParArray2D<std::size_t>::host_mirror_type{};
  }

  void Initialize(int n_regions, StateDescriptor *pkg);

  void RegisterRegionHost(int region, ProResInfo pri, Variable<Real> *v,
                          StateDescriptor *pkg);

  void CopyToDevice() {
    Kokkos::deep_copy(prores_info, prores_info_h);
    Kokkos::deep_copy(buffer_subsets, buffer_subsets_h);
  }
};

// Payload lengths (number of buffer elements) of the boundaries of a cache, indexed like
// its BndInfo array; passed to variable-length boundary ops (see bnd_ops.hpp).
using BndPayload_t = ParArray1D<int>;
using BndPayloadHost_t = typename BndPayload_t::host_mirror_type;

// Buckets of the boundaries of one cache that use a registered boundary op (pack in send
// caches, unpack in receive caches), indexed by StateDescriptor::BoundaryFuncID, and the
// remaining boundaries that take the default copy. Empty (nuser == 0) unless a variable
// in the exchange registered ops, in which case the default kernels run over the
// default subset only.
struct BndOpsCache_t {
  std::size_t nuser = 0;
  std::size_t ndefault = 0;
  ParArray1D<std::size_t> default_subset{};
  std::vector<std::size_t> buffer_subset_sizes;
  ParArray2D<std::size_t> buffer_subsets{};
  ParArray2D<std::size_t>::host_mirror_type buffer_subsets_h{};
  // Payload lengths, allocated when nuser > 0. Entries of buckets whose pack op is
  // variable-length are set each send (send caches) or receive (receive caches), the
  // others stay -1 (send the whole buffer).
  bool variable_length = false;
  BndPayload_t payload{};
  BndPayloadHost_t payload_h{};

  void clear() {
    nuser = 0;
    ndefault = 0;
    default_subset = ParArray1D<std::size_t>{};
    buffer_subset_sizes.clear();
    buffer_subsets = ParArray2D<std::size_t>{};
    buffer_subsets_h = ParArray2D<std::size_t>::host_mirror_type{};
    variable_length = false;
    payload = BndPayload_t{};
    payload_h = BndPayloadHost_t{};
  }
};

// This is just a struct to cleanly hold all of the information it is useful to cache
// for the block boundary communication routines. A copy of it is contained in MeshData.
struct BvarsSubCache_t {
  void clear() {
    buf_vec.clear();
    idx_vec.clear();
    if (sending_non_zero_flags.KokkosView().is_allocated())
      sending_non_zero_flags = ParArray1D<bool>{};
    if (sending_non_zero_flags_h.KokkosView().is_allocated())
      sending_non_zero_flags_h = ParArray1D<bool>::host_mirror_type{};
    bnd_info = BndInfoArr_t{};
    bnd_info_h = BndInfoArr_t::host_mirror_type{};
    prores_cache.clear();
    bnd_ops.clear();
  }

  bool RequiresReinitialize(Mesh *pmesh) const;

  // Stores prolongation and restriction information for boundary regions
  ProResCache_t prores_cache;

  // Buckets of boundaries handled by registered boundary ops
  BndOpsCache_t bnd_ops;

  std::vector<std::size_t> idx_vec;
  std::vector<CommBuffer<buf_pool_t<Real>::owner_t> *> buf_vec;
  ParArray1D<bool> sending_non_zero_flags;
  // Cache both host and device buffer info. Reduces mallocs, and
  // also means the bounds values are available on host if needed.
  ParArray1D<bool>::host_mirror_type sending_non_zero_flags_h;

  BndInfoArr_t bnd_info{};
  BndInfoArr_t::host_mirror_type bnd_info_h{};
  std::size_t epoch{0};
};

struct BvarsCache_t {
  std::array<BvarsSubCache_t, NUM_BNDRY_TYPES * 2> caches;
  auto &GetSubCache(BoundaryType boundType, bool send) {
    return caches[2 * static_cast<int>(boundType) + send];
  }
  void clear() {
    for (std::size_t i = 0; i < caches.size(); ++i)
      caches[i].clear();
  }
};

} // namespace parthenon

#endif // BVALS_COMMS_BND_INFO_HPP_

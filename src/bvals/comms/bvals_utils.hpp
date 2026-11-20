//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2022-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2022-2026. Triad National Security, LLC. All rights reserved.
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

#ifndef BVALS_COMMS_BVALS_UTILS_HPP_
#define BVALS_COMMS_BVALS_UTILS_HPP_

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bnd_info.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include "utils/loop_utils.hpp"

namespace parthenon {
inline auto &GetSenderGid(Mesh::channel_key_t &key) { return std::get<0>(key); }
inline auto &GetReceiverGid(Mesh::channel_key_t &key) { return std::get<1>(key); }
inline auto &GetVariable(Mesh::channel_key_t &key) { return std::get<2>(key); }
inline auto &GetLocIdx(Mesh::channel_key_t &key) { return std::get<3>(key); }
inline auto &GetOther(Mesh::channel_key_t &key) { return std::get<4>(key); }
inline std::string GetLabel(Mesh::channel_key_t &key) {
  return "sender: " + std::to_string(GetSenderGid(key)) +
         ", receiver: " + std::to_string(GetReceiverGid(key)) +
         ", var: " + GetVariable(key) + ", location: " + std::to_string(GetLocIdx(key)) +
         ", other:" + std::to_string(GetOther(key));
}
// On provider lanes the location of a channel is its pair of channel ids
inline int LaneLocIdx(int send_channel, int recv_channel) {
  return send_channel * MaxLaneChannels + recv_channel;
}
// SendKey/ReceiveKey are templated on the field type since they only need
// field->label().
template <class Field>
inline Mesh::channel_key_t SendKey(const MeshBlock *pmb, const NeighborBlock &nb,
                                   const Field &pcv, BoundaryType btype, int id) {
  const int sender_id = pmb->gid;
  const int receiver_id = nb.gid;
  if (IsLaneBoundary(btype))
    return {sender_id, receiver_id, pcv->label(),
            LaneLocIdx(nb.lane_channel, nb.lane_nb_channel), 2 * id};
  const int location_idx = nb.offsets.GetIdx();

  int gmg_self_comm = (nb.gid == pmb->gid && (btype == BoundaryType::gmg_restrict_recv ||
                                              btype == BoundaryType::gmg_restrict_send))
                          ? 1
                          : 0;
  int other =
      gmg_self_comm + 2 * id; // Combine the id and gmg_self_comm into a single tag
  return {sender_id, receiver_id, pcv->label(), location_idx, other};
}

template <class Field>
inline Mesh::channel_key_t ReceiveKey(const MeshBlock *pmb, const NeighborBlock &nb,
                                      const Field &pcv, BoundaryType btype, int id) {
  const int receiver_id = pmb->gid;
  const int sender_id = nb.gid;
  if (IsLaneBoundary(btype))
    return {sender_id, receiver_id, pcv->label(),
            LaneLocIdx(nb.lane_nb_channel, nb.lane_channel), 2 * id};
  const int location_idx = nb.lcoord_trans.Transform(nb.offsets).GetReverseIdx();
  int gmg_self_comm = (nb.gid == pmb->gid && (btype == BoundaryType::gmg_restrict_recv ||
                                              btype == BoundaryType::gmg_restrict_send))
                          ? 1
                          : 0;
  int other =
      gmg_self_comm + 2 * id; // Combine the id and gmg_self_comm into a single tag
  return {sender_id, receiver_id, pcv->label(), location_idx, other};
}

inline Mesh::channel_key_t GetChannelKey(BndId &in) {
  return {in.send_gid(), in.recv_gid(), Variable<Real>::GetLabel(in.var_id()),
          in.loc_idx(), in.extra_id()};
}

// Build a vector of pointers to all of the sending or receiving communication buffers on
// MeshData md. This cache is important for performance, since this elides a map look up
// for the buffer every time the bvals code iterates over boundaries.
//
// The buffers in the cache do not necessarily need to be in the same order as the
// sequential order of the ForEachBoundary iteration. Therefore, this also builds a vector
// for indexing from the sequential boundary index defined by the iteration pattern of
// ForEachBoundary to the index of the buffer corresponding to this boundary in the buffer
// cache. This allows for reordering the calls to send and receive on the buffers, so that
// MPI_Isends and MPI_Irecvs get posted in the same order (approximately, due to the
// possibility of multiple MeshData per rank) on the sending and receiving ranks. In
// simple tests, this did not have a big impact on performance but I think it is useful to
// leave the machinery here since it doesn't seem to have a big overhead associated with
// it (LFR).
//
// The sub-cache of a provider lane holds the boundaries of its lane (pcache->lane) only.
template <BoundaryType bound_type, class COMM_MAP, class F>
void InitializeBufferCache(std::shared_ptr<MeshData<Real>> &md, COMM_MAP *comm_map,
                           BvarsSubCache_t *pcache, F KeyFunc) {
  using namespace loops;
  using namespace loops::shorthands;
  Mesh *pmesh = md->GetMeshPointer();

  pcache->clear();

  std::vector<std::tuple<int, int, Mesh::channel_key_t>> key_order;

  int boundary_idx = 0;
  ForEachBoundary<bound_type>(
      md, pcache->lane, [&](auto pmb, sp_mbd_t rc, const nb_t &nb, const sp_cv_t v) {
        auto key = KeyFunc(pmb, nb, v, bound_type, md->GetBoundBufferId(bound_type));
        PARTHENON_DEBUG_REQUIRE(comm_map->count(key) > 0,
                                "Boundary communicator does not exist");
        // Create a unique index by combining receiver gid (second element of the key
        // tuple) and geometric element index (fourth element of the key tuple)
        int recvr_idx = 27 * GetReceiverGid(key) + GetLocIdx(key);
        key_order.push_back({recvr_idx, boundary_idx, key});
        ++boundary_idx;
      });

  // If desired, sort the keys and boundary indices by receiver_idx
  // std::sort(key_order.begin(), key_order.end(),
  //          [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });

  // Or, what the hell, you could put them in random order if you want, which
  // frighteningly seems to run faster in some cases
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(key_order.begin(), key_order.end(), g);

  int buff_idx = 0;
  pcache->buf_vec.clear();
  pcache->idx_vec = std::vector<std::size_t>(key_order.size());
  std::for_each(std::begin(key_order), std::end(key_order), [&](auto &t) {
    if (comm_map->count(std::get<2>(t)) == 0) {
      auto key = std::get<2>(t);
      PARTHENON_FAIL(std::string("Asking for buffer that doesn't exist") + " (" +
                     GetLabel(key) + ")");
    }
    pcache->buf_vec.push_back(&((*comm_map)[std::get<2>(t)]));
    (pcache->idx_vec)[std::get<1>(t)] = buff_idx++;
  });
  pcache->epoch = comm_map->GetCurrentEpoch();
  if (pcache->lane >= 0) pcache->lane_epoch = comm_map->GetLaneEpoch(pcache->lane);
  ++pcache->initializations;
}

// The sub-cache of md for a boundary type; on provider lanes, of one lane
template <BoundaryType BOUND_TYPE, bool SENDER>
inline BvarsSubCache_t &GetBvarsSubCache(MeshData<Real> *md, [[maybe_unused]] int lane) {
  if constexpr (IsLaneBoundary(BOUND_TYPE)) {
    PARTHENON_DEBUG_REQUIRE(lane >= 0, "Lane exchanges need a lane.");
    return md->GetBvarsCache().GetLaneSubCache(BOUND_TYPE, SENDER, lane);
  } else {
    return md->GetBvarsCache().GetSubCache(BOUND_TYPE, SENDER);
  }
}

template <BoundaryType BOUND_TYPE, bool SENDER>
inline auto CheckSendBufferCacheForRebuild(std::shared_ptr<MeshData<Real>> md,
                                           int lane = -1) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = GetBvarsSubCache<BOUND_TYPE, SENDER>(md.get(), lane);

  bool rebuild = false;
  bool other_communication_unfinished = false;
  int nbound = 0;
  ForEachBoundary<BOUND_TYPE>(
      md, cache.lane, [&](auto pmb, sp_mbd_t rc, const nb_t &nb, const sp_cv_t v) {
        const std::size_t ibuf = cache.idx_vec[nbound];
        auto &buf = *(cache.buf_vec[ibuf]);

        if (!buf.IsAvailableForWrite()) other_communication_unfinished = true;

        if (v->IsAllocated()) {
          buf.Allocate();
        } else {
          buf.Free();
        }

        if (ibuf < cache.bnd_info_h.size()) {
          if (cache.bnd_info_h(ibuf).alloc_status != v->GetAllocationStatus())
            rebuild = true;
          rebuild =
              rebuild || !UsingSameResource(cache.bnd_info_h(ibuf).buf, buf.buffer());
        } else {
          rebuild = true;
        }
        ++nbound;
      });
  return std::make_tuple(rebuild, nbound, other_communication_unfinished);
}

template <BoundaryType BOUND_TYPE, bool SENDER>
inline auto CheckReceiveBufferCacheForRebuild(std::shared_ptr<MeshData<Real>> md,
                                              int lane = -1) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = GetBvarsSubCache<BOUND_TYPE, SENDER>(md.get(), lane);

  bool rebuild = false;
  int nbound = 0;

  ForEachBoundary<BOUND_TYPE>(
      md, cache.lane, [&](auto pmb, sp_mbd_t rc, const nb_t &nb, const sp_cv_t v) {
        const std::size_t ibuf = cache.idx_vec[nbound];
        auto &buf = *cache.buf_vec[ibuf];
        if (ibuf < cache.bnd_info_h.size()) {
          if (cache.bnd_info_h(ibuf).alloc_status != v->GetAllocationStatus())
            rebuild = true;
          rebuild =
              rebuild || !UsingSameResource(cache.bnd_info_h(ibuf).buf, buf.buffer());

          if ((buf.GetState() == BufferState::received) &&
              !cache.bnd_info_h(ibuf).buf_allocated) {
            rebuild = true;
          }

          if ((buf.GetState() == BufferState::received_null) &&
              cache.bnd_info_h(ibuf).buf_allocated) {
            rebuild = true;
          }
        } else {
          rebuild = true;
        }
        ++nbound;
      });
  return std::make_tuple(rebuild, nbound);
}

// Fill the op buckets of a cache from (op id, sort keys..., ibuf) tuples. Each bucket is
// sorted by its keys, so ops see their boundaries in a partition-independent order.
template <class KEY>
inline void BuildBndOpsCache(BndOpsCache_t *pops, std::vector<KEY> &op_keys,
                             const std::vector<std::size_t> &default_bufs,
                             std::size_t nfuncs, int nbound) {
  auto &ops = *pops;
  ops.clear();
  if (op_keys.empty()) return;
  std::sort(op_keys.begin(), op_keys.end());
  ops.nuser = op_keys.size();
  ops.ndefault = default_bufs.size();
  ops.buffer_subset_sizes.resize(nfuncs, 0);
  ops.buffer_subsets = ParArray2D<std::size_t>("bnd_ops_subsets", nfuncs, nbound);
  ops.buffer_subsets_h = Kokkos::create_mirror_view(ops.buffer_subsets);
  for (const auto &key : op_keys) {
    const std::size_t id = std::get<0>(key);
    ops.buffer_subsets_h(id, ops.buffer_subset_sizes[id]++) = std::get<6>(key);
  }
  Kokkos::deep_copy(ops.buffer_subsets, ops.buffer_subsets_h);
  ops.default_subset = ParArray1D<std::size_t>("bnd_ops_default_subset",
                                               std::max<std::size_t>(1, ops.ndefault));
  auto default_h = Kokkos::create_mirror_view(ops.default_subset);
  for (std::size_t i = 0; i < ops.ndefault; ++i)
    default_h(i) = default_bufs[i];
  Kokkos::deep_copy(ops.default_subset, default_h);
  ops.payload = BndPayload_t("bnd_ops_payload", nbound);
  ops.payload_h = Kokkos::create_mirror_view(ops.payload);
  Kokkos::deep_copy(ops.payload_h, -1);
  Kokkos::deep_copy(ops.payload, -1);
}

using F_BND_INFO = std::function<BndInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                         std::shared_ptr<Variable<Real>> v,
                                         CommBuffer<buf_pool_t<Real>::owner_t> *buf)>;

using F_PRORES_INFO = std::function<ProResInfo(MeshBlock *pmb, const NeighborBlock &nb,
                                               std::shared_ptr<Variable<Real>> v)>;

template <BoundaryType BOUND_TYPE, bool SENDER>
inline void RebuildBufferCache(std::shared_ptr<MeshData<Real>> md, int nbound,
                               F_BND_INFO BndInfoCreator, F_PRORES_INFO ProResInfoCreator,
                               int lane = -1) {
  using namespace loops;
  using namespace loops::shorthands;
  BvarsSubCache_t &cache = GetBvarsSubCache<BOUND_TYPE, SENDER>(md.get(), lane);
  cache.bnd_info = BndInfoArr_t(ViewOfViewAlloc("bnd_info"), nbound);
  cache.bnd_info_h = create_view_of_view_mirror(cache.bnd_info);

  // Lane neighbors are same-level by contract, so lane caches build no
  // prolongation/restriction info
  constexpr bool on_lane = IsLaneBoundary(BOUND_TYPE);

  // prolongation/restriction sub-sets
  // TODO(JMM): Right now I exclude fluxcorrection boundaries but if
  // we eventually had custom fluxcorrection ops, you could remove
  // this.
  Mesh *pmesh = md->GetParentPointer();
  StateDescriptor *pkg = (pmesh->resolved_packages).get();
  if constexpr (on_lane) {
    cache.prores_cache.clear();
  } else {
    cache.prores_cache.Initialize(nbound, pkg);
  }

  // Registered boundary ops act on the variable's own buffers; flux correction has
  // combine ops on the receiving side only
  constexpr bool flx_send = BOUND_TYPE == BoundaryType::flxcor_send;
  constexpr bool flx_recv = BOUND_TYPE == BoundaryType::flxcor_recv;
  // On lanes the flux exchange has registered ops on both sides (pack and combine)
  constexpr bool lane_flx = BOUND_TYPE == BoundaryType::lane_flxcor_send ||
                            BOUND_TYPE == BoundaryType::lane_flxcor_recv;
  std::size_t nfuncs = 0;
  if constexpr (flx_recv || lane_flx) {
    nfuncs = pkg->NumFluxCorrectionFuncs();
  } else if constexpr (!flx_send) {
    nfuncs = pkg->NumBoundaryFuncs();
  }
  const bool use_ops = nfuncs > 0;
  std::unordered_map<int, int> block_index;
  for (int b = 0; b < md->NumBlocks(); ++b)
    block_index[md->GetBlockData(b)->GetBlockPointer()->gid] = b;
  // (op id, block, variable label, offset index, neighbor gid, iteration order, ibuf)
  using op_key_t = std::tuple<std::size_t, int, std::string, int, int, int, std::size_t>;
  std::vector<op_key_t> op_keys;
  std::vector<std::size_t> default_bufs;
  bool variable_length = false;

  int ibound = 0;
  ForEachBoundary<BOUND_TYPE>(
      md, cache.lane, [&](auto pmb, sp_mbd_t rc, const nb_t &nb, const sp_cv_t v) {
        // bnd_info
        const std::size_t ibuf = cache.idx_vec[ibound];
        cache.bnd_info_h(ibuf) = BndInfoCreator(pmb, nb, v, cache.buf_vec[ibuf]);
        auto &bi = cache.bnd_info_h(ibuf);
        bi.block = block_index.at(pmb->gid);
        bi.send_gid = SENDER ? pmb->gid : nb.gid;
        bi.recv_gid = SENDER ? nb.gid : pmb->gid;
        bi.offsets = nb.offsets;
        if constexpr (on_lane) {
          bi.channel = nb.lane_channel;
          bi.nb_channel = nb.lane_nb_channel;
        }

        if (use_ops) {
          int id = -1;
          if constexpr (lane_flx) {
            id = pkg->FluxCorrectionFuncID(v->GetFluxCorrectionFunctions());
          } else if constexpr (flx_recv) {
            // Combine ops get the same-level boundaries only
            if (v->HasFluxCorrectionOps() && nb.loc.level() == pmb->loc.level())
              id = pkg->FluxCorrectionFuncID(v->GetFluxCorrectionFunctions());
          } else {
            const auto &funcs = v->GetBoundaryFunctions();
            // Ops get the same-level boundaries only (lane neighbors are same-level by
            // contract); coarse-fine boundaries keep the default copy and the variable's
            // prolongation/restriction ops
            if (v->HasBoundaryOps() && (SENDER ? funcs.HasPack() : funcs.HasUnpack()) &&
                (on_lane || nb.loc.level() == pmb->loc.level())) {
              id = pkg->BoundaryFuncID(funcs);
              variable_length = variable_length || (SENDER && funcs.variable_length);
            }
          }
          if (id >= 0) {
            op_keys.emplace_back(id, bi.block, v->label(),
                                 on_lane ? nb.lane_channel : nb.offsets.GetIdx(), nb.gid,
                                 ibound, ibuf);
          } else {
            default_bufs.push_back(ibuf);
          }
        }

        // subsets ordering is same as in cache.bnd_info
        // RefinementFunctions_t owns all relevant functionality, so
        // only one ParArray2D needed.
        if constexpr (!on_lane)
          cache.prores_cache.RegisterRegionHost(ibuf, ProResInfoCreator(pmb, nb, v),
                                                v.get(), pkg);

        ++ibound;
      });
  Kokkos::deep_copy(cache.bnd_info, cache.bnd_info_h);
  if constexpr (!on_lane) cache.prores_cache.CopyToDevice();
  BuildBndOpsCache(&cache.bnd_ops, op_keys, default_bufs, nfuncs, nbound);
  cache.bnd_ops.variable_length = variable_length;
  PARTHENON_REQUIRE_THROWS(!variable_length || !pmesh->do_coalesced_comms,
                           "Variable-length boundary ops need coalesced comms off");
}

} // namespace parthenon

#endif // BVALS_COMMS_BVALS_UTILS_HPP_

//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/comms/coalesced_buffers.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/comms_trace.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/instrument.hpp"

namespace parthenon {

namespace {
// All comm lanes of a sender BoundaryType share a communicator, so combine the partition
// tag and the lane (and on provider lanes the provider lane) into a unique MPI tag for
// the data message (the sparse status message uses this tag + 1)
int CoalescedDataTag(Mesh *pmesh, BoundaryType b_type, int partition_tag, int lane,
                     int provider_lane) {
  const int nlanes = pmesh->GetNumberOfCommLanes(GetAssociatedSender(b_type));
  if (provider_lane < 0) return 2 * (nlanes * partition_tag + lane);
  return 2 *
         ((nlanes * partition_tag + lane) * kMaxCoalescedProviderLanes + provider_lane);
}

// Tag of the lane info messages of one (sender type, comm lane, provider lane)
int LaneInfoTagOf(BoundaryType b_type, int lane, int provider_lane) {
  const bool flux = GetAssociatedSender(b_type) == BoundaryType::lane_flxcor_send;
  return 2 * (lane * kMaxCoalescedProviderLanes + provider_lane) + (flux ? 1 : 0);
}

// The entries of a CoalescedComms map with one (BoundaryType, comm lane, provider lane)
template <class MAP>
auto EntriesBegin(MAP &m, BoundaryType b_type, int lane, int provider_lane) {
  return m.lower_bound(CoalescedComms::rank_key_t{b_type, lane, provider_lane, INT_MIN});
}
template <class IT>
bool InEntries(const IT &it, const IT &end, BoundaryType b_type, int lane,
               int provider_lane) {
  return it != end && std::get<0>(it->first) == b_type &&
         std::get<1>(it->first) == lane && std::get<2>(it->first) == provider_lane;
}

void WaitForWrite(CoalescedBuffer &cb) {
  constexpr std::int64_t max_it = 1e10;
  std::int64_t it = 0;
  while (!cb.IsAvailableForWrite() && ++it < max_it) {
  }
  PARTHENON_REQUIRE(it < max_it, "Waited too long for a coalesced send to complete.");
}

// Calls f(pmb, nb, v) for every boundary of type BTYPE of provider lane provider_lane
// with a block on another rank
template <BoundaryType BTYPE, class F>
void ForEachOffRankLaneBoundary(std::shared_ptr<MeshData<Real>> &md, int provider_lane,
                                F f) {
  using namespace loops::shorthands;
  loops::ForEachBoundary<BTYPE>(md, provider_lane,
                                [&](auto pmb, sp_mbd_t, const nb_t &nb, const sp_cv_t v) {
                                  if (nb.rank != Globals::my_rank) f(pmb, nb, v);
                                });
}

// Copy between the var-boundary buffers and the coalesced buffer, splitting each buffer
// among enough teams that the kernel launches at least
// minimum_number_of_teams_for_boundary_kernel teams
template <bool TO_COALESCED>
void CopyCoalesced(Mesh *pmesh, ParArray1DRaw<BndId> &bids) {
  const int nbound = bids.size();
  if (nbound == 0) return;
  const int nteams_per_buffer = GetNteamsPerBoundaryBuffer(pmesh, nbound);
  Kokkos::parallel_for(
      PARTHENON_AUTO_LABEL,
      Kokkos::TeamPolicy<>(parthenon::DevExecSpace(), nbound * nteams_per_buffer,
                           Kokkos::AUTO),
      KOKKOS_LAMBDA(parthenon::team_mbr_t team_member) {
        const int b = team_member.league_rank() / nteams_per_buffer;
        const int bteam = team_member.league_rank() % nteams_per_buffer;
        if (bids[b].buf_allocated) {
          const SplitFlatIndexRangeAmongTeams split(nteams_per_buffer, 1, bids[b].size());
          Real *com_buf = &(bids[b].coalesced_buf(bids[b].start_idx()));
          Real *buf = &(bids[b].buf(0));
          Kokkos::parallel_for(Kokkos::TeamThreadRange<>(team_member,
                                                         split.GetStart(bteam),
                                                         split.GetEnd(bteam)),
                               [&](const int idx) {
                                 if constexpr (TO_COALESCED) {
                                   com_buf[idx] = buf[idx];
                                 } else {
                                   buf[idx] = com_buf[idx];
                                 }
                               });
        }
      });
}
} // namespace

//----------------------------------------------------------------------------------------
CoalescedBuffer::CoalescedBuffer(bool sender, int partition_tag, int lane,
                                 int provider_lane, int other_rank, BoundaryType b_type,
                                 mpi_comm_t comm, Mesh *pmesh)
    : sender(sender), partition_tag(partition_tag), lane(lane),
      provider_lane(provider_lane), other_rank(other_rank), b_type(b_type), comm(comm),
      pmesh(pmesh), current_size(0),
      coalesced_comm_buffer(
          CoalescedDataTag(pmesh, b_type, partition_tag, lane, provider_lane),
          sender ? Globals::my_rank : other_rank, sender ? other_rank : Globals::my_rank,
          comm, [](int size) { return buf_t("Combined Buffer", 2 * size); }, true),
      sparse_status_buffer(
          CoalescedDataTag(pmesh, b_type, partition_tag, lane, provider_lane) + 1,
          sender ? Globals::my_rank : other_rank, sender ? other_rank : Globals::my_rank,
          comm, [](int size) { return std::vector<int>(size); }, true) {
  // Provider lane messages can be empty; the combined buffer keeps its storage
  if (OnProviderLane()) coalesced_comm_buffer.KeepStorageOnNull();
}

//----------------------------------------------------------------------------------------
ParArray1DRaw<BndId> &CoalescedBuffer::GetBndIdsOnDevice(const std::set<Uid_t> &vars,
                                                         int *pcomb_size) {
  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  // On provider lanes a buffer takes its payload length, otherwise its size
  const bool by_payload = OnProviderLane();
  int nbnd_id{0};
  std::size_t comb_size{0};
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    nbnd_id += coalesced_info_buf.at(uid).size();
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      auto buf_state = pvbbuf->GetState();
      if ((buf_state == BufferState::sending) || (buf_state == BufferState::received))
        comb_size += by_payload ? pvbbuf->GetPayloadSize() : bnd_id.size();
    }
  }
  if (pcomb_size != nullptr) *pcomb_size = comb_size;

  bool updated = false;
  if (comb_size > coalesced_comm_buffer.buffer().size()) {
    PARTHENON_REQUIRE(
        sender, "Something bad is going on if we are doing this on a receiving buffer.");
    coalesced_comm_buffer.Allocate(comb_size);
    updated = true;
  }

  if (bnd_ids_device_map.count(var_set) == 0)
    bnd_ids_device_map.emplace(std::make_pair(
        var_set, ParArray1DRaw<BndId>(parthenon::ViewOfViewAlloc("bnd_id"), nbnd_id)));
  auto &bnd_ids_device = bnd_ids_device_map.at(var_set);
  if (bnd_ids_host_map.count(var_set) == 0)
    bnd_ids_host_map.emplace(
        std::make_pair(var_set, create_view_of_view_mirror(bnd_ids_device)));
  auto &bnd_ids_host = bnd_ids_host_map.at(var_set);

  if (nbnd_id != bnd_ids_device.size()) {
    bnd_ids_device = ParArray1DRaw<BndId>(parthenon::ViewOfViewAlloc("bnd_id"), nbnd_id);
    bnd_ids_host = create_view_of_view_mirror(bnd_ids_device);
    updated = true;
  }

  int idx{0};
  std::size_t c_buf_idx{0}; // Index at which v-b buffer starts in combined buffer
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      auto &bid_h = bnd_ids_host[idx];
      auto buf_state = pvbbuf->GetState();
      PARTHENON_REQUIRE(buf_state != BufferState::stale,
                        "Trying to work with a stale buffer.");

      const bool alloc =
          (buf_state == BufferState::sending) || (buf_state == BufferState::received);
      const int len = alloc && by_payload ? pvbbuf->GetPayloadSize() : bnd_id.size();

      // Test if this boundary has changed (the channel, ignoring the size, then the size)
      bool same_channel = true;
      for (int i = 0; i < 8; ++i)
        same_channel = same_channel && bid_h.data[i] == bnd_id.data[i];
      if (!same_channel || bid_h.size() != len || (bid_h.buf_allocated != alloc) ||
          (bid_h.start_idx() != c_buf_idx) ||
          !UsingSameResource(bid_h.buf, pvbbuf->buffer()) ||
          bid_h.coalesced_buf.data() != coalesced_comm_buffer.buffer().data()) {
        updated = true;
        bid_h = bnd_id;
        bid_h.size() = len;
        bid_h.buf_allocated = alloc;
        bid_h.start_idx() = c_buf_idx;
        bid_h.coalesced_buf = coalesced_comm_buffer.buffer();
        if (bid_h.buf_allocated) bid_h.buf = pvbbuf->buffer();
      }
      if (bid_h.buf_allocated) c_buf_idx += bid_h.size();
      idx++;
    }
  }
  if (updated) Kokkos::deep_copy(bnd_ids_device, bnd_ids_host);
  return bnd_ids_device;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::TraceOpen(double t0) {
  trace_rec = comms_trace::ExchangeRecord();
  trace_rec.cycle = comms_trace::CurrentCycle();
  trace_rec.build = comms_trace::CurrentBuild();
  trace_rec.in_step = comms_trace::InStep();
  trace_rec.exchange = trace_nexchange++;
  trace_rec.is_send = sender;
  trace_rec.peer = other_rank;
  // Record the sender type on both sides so that sends and receives can be matched
  trace_rec.btype = static_cast<int>(GetAssociatedSender(b_type));
  trace_rec.lane = lane;
  trace_rec.partition_tag = partition_tag;
  trace_rec.t[0] = t0;
  trace_last_poll = t0;
  trace_open = true;
}

void CoalescedBuffer::TraceClose() {
  auto *stored = comms_trace::RecordExchange(trace_rec);
  if (sender) trace_send_rec = stored;
  trace_open = false;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::PackAndSend(const std::set<Uid_t> &vars, int epoch) {
  // The completion of the previous send was never observed
  trace_send_rec = nullptr;
  PARTHENON_REQUIRE(coalesced_comm_buffer.IsAvailableForWrite(),
                    "Trying to write to a buffer that is in use.");
  if (comms_trace::Active()) TraceOpen(comms_trace::Now());
  int comb_size;
  auto &bids = GetBndIdsOnDevice(vars, &comb_size);
  CopyCoalesced<true>(pmesh, bids);
#ifdef MPI_PARALLEL
  const double t_fence = comms_trace::Now();
  Kokkos::fence();
  const double t_fence_done = comms_trace::Now();
  comms_trace::AddFenceTime(t_fence_done - t_fence);
  if (trace_open) trace_rec.t[1] = t_fence_done;
#endif
  coalesced_comm_buffer.Send(false, comb_size);
  if (trace_open) {
    trace_rec.t[2] = comms_trace::Now();
    trace_rec.bytes = static_cast<std::int64_t>(comb_size) * sizeof(Real);
    trace_rec.nbuf = bids.size();
  }

  // Send the sparse null info as well. This has an entry for every buffer, not just the
  // ones in var_set, since the receiver may unpack this message while it is working on
  // a MeshData with a different set of variables and so must learn var_set from here
  const int head = StatusHeader();
  const std::size_t nbuf_total = TotalBuffers() + head;
  if (nbuf_total != sparse_status_buffer.buffer().size()) {
    sparse_status_buffer.Allocate(nbuf_total);
  }

  const auto &var_set = vars.size() == 0 ? all_vars : vars;
  auto &stat = sparse_status_buffer.buffer();
  if (head > 0) stat[0] = epoch;
  int idx{head};
  for (auto &[uid, v] : coalesced_info_buf) {
    const bool in_message = var_set.count(uid) > 0;
    for (auto &[bnd_id, pvbbuf] : v) {
      int send_type = not_in_message;
      if (in_message) {
        const auto state = pvbbuf->GetState();
        PARTHENON_REQUIRE(state == BufferState::sending ||
                              state == BufferState::sending_null,
                          "Bad state.");
        if (OnProviderLane()) {
          send_type = state == BufferState::sending ? pvbbuf->GetPayloadSize() : 0;
        } else {
          send_type = (state == BufferState::sending);
        }
      }
      stat[idx] = send_type;
      ++idx;
    }
  }
  sparse_status_buffer.Send();
  if (trace_open) {
    trace_rec.t[3] = comms_trace::Now();
    TraceClose();
  }

  // Information in these send buffers is no longer required
  for (auto uid : var_set) {
    // Skip this variable if it is not communicated in this BoundaryType
    if (coalesced_info_buf.count(uid) == 0) continue;
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      pvbbuf->Stale();
    }
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffer::TryReceiveStatus() {
  if (sparse_status_buffer.GetState() == BufferState::received) return true;
  return sparse_status_buffer.TryReceive();
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffer::TryReceiveAndUnpack(int epoch) {
  if ((sparse_status_buffer.GetState() == BufferState::received) &&
      (coalesced_comm_buffer.GetState() == BufferState::received))
    return true;

  if (trace_open || comms_trace::Active()) {
    const double now = comms_trace::Now();
    if (!trace_open) {
      TraceOpen(now);
    } else {
      trace_rec.max_poll_gap = std::max(trace_rec.max_poll_gap, now - trace_last_poll);
      trace_last_poll = now;
    }
    trace_rec.npolls++;
  }

  // The status message tells us which variables the sender put in this message
  const bool received_sparse = sparse_status_buffer.TryReceive();
  if (trace_open) {
    const double now = comms_trace::Now();
    auto &t = trace_rec.t;
    if (t[1] < 0.0 && (received_sparse || sparse_status_buffer.ReceiveStarted()))
      t[1] = now;
    if (t[3] < 0.0 && received_sparse) t[3] = now;
  }
  if (!received_sparse) return false;
  auto &stat = sparse_status_buffer.buffer();
  const int head = StatusHeader();
  // A provider lane message from a newer layout waits for its lane info
  if (head > 0 && stat[0] != epoch) return false;
  PARTHENON_REQUIRE(sparse_status_buffer.GetPayloadSize() ==
                        static_cast<int>(TotalBuffers()) + head,
                    "Coalesced status message does not cover all buffers.");

  std::set<Uid_t> var_set;
  int idx{head};
  for (auto &[uid, v] : coalesced_info_buf) {
    if (v.size() > 0 && stat[idx] != not_in_message) var_set.insert(uid);
    idx += v.size();
  }

  // Make sure the var-boundary buffers are available to write to
  for (auto uid : var_set) {
    for (auto &[bnd_id, pvbbuf] : coalesced_info_buf.at(uid)) {
      if (pvbbuf->GetState() != BufferState::stale) return false;
    }
  }

  const bool received = coalesced_comm_buffer.TryReceive();
  if (trace_open) {
    const double now = comms_trace::Now();
    auto &t = trace_rec.t;
    if (t[2] < 0.0 && (received || coalesced_comm_buffer.ReceiveStarted())) t[2] = now;
    if (t[4] < 0.0 && received) t[4] = now;
  }
  if (!received) return false;

  // Allocate and free buffers as required
  idx = head;
  for (auto &[uid, v] : coalesced_info_buf) {
    for (auto &[bnd_id, pvbbuf] : v) {
      if (OnProviderLane()) {
        // The entry is the payload length; an empty buffer keeps its storage
        const int len = stat[idx++];
        if (len == not_in_message) continue;
        pvbbuf->SetReceived();
        if (!pvbbuf->IsActive()) pvbbuf->Allocate();
        PARTHENON_REQUIRE(len <= static_cast<int>(pvbbuf->buffer().size()),
                          "Coalesced provider lane payload exceeds the receive buffer.");
        pvbbuf->SetPayloadSize(len);
        continue;
      }
      if (stat[idx] == 1) {
        pvbbuf->SetReceived();
        if (!pvbbuf->IsActive()) pvbbuf->Allocate();
      } else if (stat[idx] == 0) {
        pvbbuf->SetReceivedNull();
        if (pvbbuf->IsActive()) pvbbuf->Free();
      }
      idx++;
    }
  }

  // An empty var_set would mean all variables to GetBndIdsOnDevice
  int comb_size{0};
  std::size_t nbids{0};
  if (var_set.size() > 0) {
    auto &bids = GetBndIdsOnDevice(var_set, &comb_size);
    CopyCoalesced<false>(pmesh, bids);
    nbids = bids.size();
  }
  PARTHENON_REQUIRE(!OnProviderLane() ||
                        comb_size == coalesced_comm_buffer.GetPayloadSize(),
                    "Coalesced provider lane message length differs from its payloads.");
  coalesced_comm_buffer.Stale();
  sparse_status_buffer.Stale();
  if (trace_open) {
    trace_rec.t[5] = comms_trace::Now();
    trace_rec.bytes = static_cast<std::int64_t>(comb_size) * sizeof(Real);
    trace_rec.nbuf = nbids;
    TraceClose();
  }

  return true;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffer::AddVarBoundary(BndId &bnd_id) {
  auto key = GetChannelKey(bnd_id);
  PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(key), "Buffer doesn't exist.");
  var_buf_t *pbuf = &(pmesh->boundary_comm_map.at(key));
  coalesced_info_buf[bnd_id.var_id()].push_back(std::make_pair(bnd_id, pbuf));
  current_size += bnd_id.size(); // This will be the maximum size of communication since
                                 // it includes all variables
  all_vars.insert(bnd_id.var_id());
}

void CoalescedBuffer::ClearVarBoundaries() {
  coalesced_info_buf.clear();
  all_vars.clear();
  current_size = 0;
}

void CoalescedBuffer::AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                                     const std::shared_ptr<Variable<Real>> &var) {
  // Store both the variable-boundary buffer information and a pointer to the v-b buffer
  // itself associated with var ids
  BndId bnd_id = BndId::GetSend(pmb, nb, var, b_type, partition_tag, -1, lane);
  PARTHENON_REQUIRE(!OnProviderLane() || bnd_id.size() > 0,
                    "Provider lane buffers must have a positive BufferSize.");
  AddVarBoundary(bnd_id);
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CoalescedBuffersRank::CoalescedBuffersRank(int o_rank, BoundaryType b_type, int lane,
                                           bool send, mpi_comm_t comm, Mesh *pmesh,
                                           int provider_lane, mpi_comm_t info_comm)
    : other_rank(o_rank), b_type(b_type), lane(lane), provider_lane(provider_lane),
      sender(send), buffers_built(false), comm(comm), pmesh(pmesh), info_comm(info_comm) {
  if (provider_lane >= 0) info_tag = LaneInfoTagOf(b_type, lane, provider_lane);

  // There is a separate communicator for each sender BoundaryType, so the comm lane is
  // sufficient to make the info message tag unique
  int tag = lane;
  if (sender) {
    message = com_buf_t(tag, Globals::my_rank, other_rank, comm,
                        [](int size) { return std::vector<int>(size); });
  } else {
    message = com_buf_t(
        tag, other_rank, Globals::my_rank, comm,
        [](int size) { return std::vector<int>(size); }, true);
  }
  PARTHENON_REQUIRE(other_rank != Globals::my_rank, "Should only build for other ranks.");
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::AddSendBuffer(int partition_tag, MeshBlock *pmb,
                                         const NeighborBlock &nb,
                                         const std::shared_ptr<Variable<Real>> &var) {
  if (coalesced_bufs.count(partition_tag) == 0)
    coalesced_bufs.emplace(std::make_pair(
        partition_tag, CoalescedBuffer(true, partition_tag, lane, provider_lane,
                                       other_rank, b_type, comm, pmb->pmy_mesh)));

  auto &coal_buf = coalesced_bufs.at(partition_tag);
  coal_buf.AddVarBoundary(pmb, nb, var);
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveBufInfo() {
  PARTHENON_REQUIRE(!sender, "Trying to receive on a combined sender.");
  if (buffers_built) return buffers_built;

  bool received = message.TryReceive();
  if (received) {
    auto &mess_buf = message.buffer();
    int npartitions = mess_buf[0];
    // Unpack into per combined buffer information
    int idx{nglobal};

    for (int p = 0; p < npartitions; ++p) {
      const int partition_tag = mess_buf[idx++];
      const int nbuf = mess_buf[idx++];
      const int total_size = mess_buf[idx++];

      // Create the new partition
      coalesced_bufs.emplace(std::make_pair(
          partition_tag, CoalescedBuffer(false, partition_tag, lane, provider_lane,
                                         other_rank, b_type, comm, pmesh)));
      auto &coal_buf = coalesced_bufs.at(partition_tag);

      for (int b = 0; b < nbuf; ++b) {
        BndId bnd_id(&(mess_buf[idx]));
        coal_buf.AddVarBoundary(bnd_id);
        idx += BndId::NDAT;
      }
    }
    message.Stale();

    buffers_built = true;
    return true;
  }
  return false;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::ResolveAndSendBufInfo() {
  // First calculate the total size of the message
  int total_buffers{0};
  for (auto &[partition_tag, coalesced_buf] : coalesced_bufs)
    total_buffers += coalesced_buf.TotalBuffers();
  std::size_t total_partitions = coalesced_bufs.size();

  std::size_t mesg_size =
      nglobal + nper_part * total_partitions + BndId::NDAT * total_buffers;
  message.Allocate(mesg_size);

  auto &mess_buf = message.buffer();
  mess_buf[0] = total_partitions;

  // Pack the data
  int idx{nglobal};
  for (auto &[partition_tag, coalesced_buf] : coalesced_bufs) {
    mess_buf[idx++] = partition_tag;                // Used as the comm tag
    mess_buf[idx++] = coalesced_buf.TotalBuffers(); // Number of buffers
    mess_buf[idx++] =
        coalesced_buf.current_size; // combined size of buffers (now probably unused)
    for (auto &[uid, v] : coalesced_buf.coalesced_info_buf) {
      for (auto &[bnd_id, pbvbuf] : v) {
        bnd_id.Serialize(&(mess_buf[idx]));
        idx += BndId::NDAT;
      }
    }
  }

  message.Send();

  buffers_built = true;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::PackAndSend(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to send combined buffers before they have been built");
  // A registered partition may have no boundaries shared with other_rank, in which case
  // there is nothing to send
  const int partition_tag = pmesh->pcoalesced_comms->GetPartitionTag(pmd);
  if (coalesced_bufs.count(partition_tag))
    coalesced_bufs.at(partition_tag).PackAndSend(pmd->GetUids(), info_epoch);
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::IsAvailableForWrite(MeshData<Real> *pmd) {
  PARTHENON_REQUIRE(sender, "Shouldn't be checking this on non-sender.");
  const int partition_tag = pmesh->pcoalesced_comms->GetPartitionTag(pmd);
  if (coalesced_bufs.count(partition_tag) == 0) return true;
  return coalesced_bufs.at(partition_tag).IsAvailableForWrite();
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveAndUnpack(MeshData<Real> *pmd, int partition_tag) {
  PARTHENON_REQUIRE(buffers_built,
                    "Trying to recv combined buffers before they have been built");
  PARTHENON_REQUIRE(coalesced_bufs.count(partition_tag) > 0,
                    "Trying to receive on a non-existent combined receive buffer.");
  return coalesced_bufs.at(partition_tag).TryReceiveAndUnpack();
}

//----------------------------------------------------------------------------------------
CoalescedBuffer &CoalescedBuffersRank::GetPartition(int partition_tag) {
  auto it = coalesced_bufs.find(partition_tag);
  if (it == coalesced_bufs.end())
    it = coalesced_bufs
             .emplace(partition_tag,
                      CoalescedBuffer(sender, partition_tag, lane, provider_lane,
                                      other_rank, b_type, comm, pmesh))
             .first;
  return it->second;
}

//----------------------------------------------------------------------------------------
std::vector<int> CoalescedBuffersRank::SerializeLaneInfo() const {
  std::vector<int> out{static_cast<int>(coalesced_bufs.size())};
  for (const auto &[partition_tag, cb] : coalesced_bufs) {
    out.insert(out.end(),
               {partition_tag, static_cast<int>(cb.TotalBuffers()), cb.current_size});
    for (const auto &[uid, v] : cb.coalesced_info_buf) {
      for (const auto &[bnd_id, pbuf] : v)
        out.insert(out.end(), bnd_id.data, bnd_id.data + BndId::NDAT);
    }
  }
  return out;
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::DrainLaneInfos() {
#ifdef MPI_PARALLEL
  while (true) {
    int flag;
    MPI_Status status;
    PARTHENON_MPI_CHECK(MPI_Iprobe(other_rank, info_tag, info_comm, &flag, &status));
    if (!flag) return;
    int n;
    PARTHENON_MPI_CHECK(MPI_Get_count(&status, MPI_INT, &n));
    std::vector<int> info(n);
    PARTHENON_MPI_CHECK(MPI_Recv(info.data(), n, MPI_INT, other_rank, info_tag, info_comm,
                                 MPI_STATUS_IGNORE));
    const int epoch = info[0];
    // Superseded by the layout in use
    if (epoch <= info_epoch) continue;
    // Poll the status messages of its partitions from now on
    int idx = 2;
    for (int p = 0; p < info[1]; ++p) {
      GetPartition(info[idx]);
      idx += nper_part + BndId::NDAT * info[idx + 1];
    }
    pending_infos[epoch] = std::move(info);
  }
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::ApplyLaneInfo(int epoch) {
  const std::vector<int> info = std::move(pending_infos.at(epoch));
  pending_infos.erase(pending_infos.begin(), pending_infos.upper_bound(epoch));
  for (auto &[partition_tag, cb] : coalesced_bufs)
    cb.ClearVarBoundaries();
  int idx = 2;
  for (int p = 0; p < info[1]; ++p) {
    auto &cb = GetPartition(info[idx]);
    const int nbuf = info[idx + 1];
    idx += nper_part;
    for (int b = 0; b < nbuf; ++b) {
      BndId bnd_id(&info[idx]);
      idx += BndId::NDAT;
      PARTHENON_REQUIRE(pmesh->boundary_comm_map.count(GetChannelKey(bnd_id)),
                        "Rank " + std::to_string(other_rank) + " sent provider lane " +
                            std::to_string(provider_lane) +
                            " data for a buffer this rank does not have (asymmetric "
                            "lane graph, or RebuildLane missing on this rank).");
      cb.AddVarBoundary(bnd_id);
    }
  }
  // Partitions that no longer send to this rank
  for (auto it = coalesced_bufs.begin(); it != coalesced_bufs.end();) {
    if (it->second.TotalBuffers() == 0) {
      PARTHENON_REQUIRE(it->second.sparse_status_buffer.GetState() !=
                            BufferState::received,
                        "Status message from a partition without buffers.");
      it = coalesced_bufs.erase(it);
    } else {
      ++it;
    }
  }
  info_epoch = epoch;
  buffers_built = true;
  CheckLaneLayout();
  PARTHENON_REQUIRE(info_valid, "Provider lane " + std::to_string(provider_lane) +
                                    " buffers from rank " + std::to_string(other_rank) +
                                    " differ from this rank's receive buffers "
                                    "(asymmetric lane graph or buffer sizes).");
}

//----------------------------------------------------------------------------------------
void CoalescedBuffersRank::CheckLaneLayout() {
  if (info_epoch < 0) {
    info_valid = false;
    return;
  }
  bool found = true;
  std::vector<Mesh::channel_key_t> keys;
  for (auto &[partition_tag, cb] : coalesced_bufs) {
    for (auto &[uid, v] : cb.coalesced_info_buf) {
      for (auto &[bnd_id, pbuf] : v) {
        auto key = GetChannelKey(bnd_id);
        if (pmesh->boundary_comm_map.count(key)) {
          pbuf = &(pmesh->boundary_comm_map.at(key));
        } else {
          pbuf = nullptr;
          found = false;
        }
        keys.push_back(std::move(key));
      }
    }
  }
  std::sort(keys.begin(), keys.end());
  info_valid = found && keys == expected_keys;
}

//----------------------------------------------------------------------------------------
bool CoalescedBuffersRank::TryReceiveLane() {
  if (!info_valid) DrainLaneInfos();
  // The epoch of the sender's layout, from any status message that has arrived
  int epoch = -1;
  for (auto &[partition_tag, cb] : coalesced_bufs)
    if (cb.TryReceiveStatus()) epoch = std::max(epoch, cb.StatusEpoch());
  if (epoch < 0) return false;
  if (epoch != info_epoch) {
    PARTHENON_REQUIRE(epoch > info_epoch, "Provider lane message from rank " +
                                              std::to_string(other_rank) +
                                              " is older than the layout in use.");
    if (pending_infos.count(epoch) == 0) DrainLaneInfos();
    if (pending_infos.count(epoch) == 0) return false;
    ApplyLaneInfo(epoch);
  }
  PARTHENON_REQUIRE(info_valid, "Rank " + std::to_string(other_rank) +
                                    " kept its provider lane " +
                                    std::to_string(provider_lane) +
                                    " layout, but this rank's receive buffers from it "
                                    "changed (asymmetric lane graph or buffer sizes, or "
                                    "RebuildLane missing on the other rank).");
  bool all_received = true;
  for (auto &[partition_tag, cb] : coalesced_bufs)
    all_received = cb.TryReceiveAndUnpack(info_epoch) && all_received;
  return all_received;
}

//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
CoalescedComms::CoalescedComms(Mesh *pmesh) : pmesh(pmesh) {
  // TODO(LFR): Switch to a different communicator for each BoundaryType pair
  std::vector<BoundaryType> types{BoundaryType::any, BoundaryType::flxcor_send,
                                  BoundaryType::gmg_same, BoundaryType::gmg_restrict_send,
                                  BoundaryType::gmg_prolongate_send};
  // Provider lanes get their own communicators only when they can coalesce
  if (pmesh->do_coalesced_comms) {
    types.push_back(BoundaryType::lane);
    types.push_back(BoundaryType::lane_flxcor_send);
  }
  for (auto b_type : types) {
    auto &comm = comms[b_type];
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &comm));
#else
    comm = 0;
#endif
  }
#ifdef MPI_PARALLEL
  lane_info_comm = MPI_COMM_NULL;
  if (pmesh->do_coalesced_comms)
    PARTHENON_MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &lane_info_comm));
#else
  lane_info_comm = 0;
#endif
}

//----------------------------------------------------------------------------------------
CoalescedComms::~CoalescedComms() {
#ifdef MPI_PARALLEL
  // Lane infos nobody read (the lane was not exchanged again) are let go
  for (auto &[info, req] : lane_infos_in_flight)
    if (req != MPI_REQUEST_NULL) PARTHENON_MPI_CHECK(MPI_Request_free(&req));
  for (auto &[b_type, comm] : comms)
    PARTHENON_MPI_CHECK(MPI_Comm_free(&comm));
  if (lane_info_comm != MPI_COMM_NULL)
    PARTHENON_MPI_CHECK(MPI_Comm_free(&lane_info_comm));
#endif
}

//----------------------------------------------------------------------------------------
void CoalescedComms::clear() {
  bool can_delete;
  int iter{0};
  const int max_iters = 1e8;
  do {
    can_delete = true;
    for (auto &[p, cbr] : coalesced_send_buffers) {
      can_delete = cbr.message.IsAvailableForWrite() && can_delete;
      for (auto &[r, cbrp] : cbr.coalesced_bufs) {
        can_delete = cbrp.IsAvailableForWrite() && can_delete;
      }
    }
    iter++;
  } while (!can_delete && iter < max_iters);
  if (iter >= max_iters) PARTHENON_FAIL("Waited too long to clear CoalescedComms.");

  coalesced_send_buffers.clear();
  coalesced_recv_buffers.clear();
  partition_tags.clear();
}

//----------------------------------------------------------------------------------------
void CoalescedComms::RegisterPartition(const BlockListPartition *partition) {
  if (partition_tags.count(partition) == 0) {
    const int tag = partition_tags.size();
    partition_tags[partition] = tag;
  }
}

//----------------------------------------------------------------------------------------
int CoalescedComms::GetPartitionTag(MeshData<Real> *pmd) const {
  const auto it = partition_tags.find(pmd->partition.get());
  if (it == partition_tags.end())
    PARTHENON_FAIL("Trying to do coalesced communication using a partition that is "
                   "unregistered (i.e. not a default or multigrid partition).");
  return it->second;
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddSendBuffer(const BlockListPartition *partition, MeshBlock *pmb,
                                   const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>> &var,
                                   BoundaryType b_type, int lane) {
  PARTHENON_REQUIRE(partition_tags.count(partition),
                    "Partition must be registered before adding buffers.");
  const rank_key_t key{b_type, lane, -1, nb.rank};
  if (coalesced_send_buffers.count(key) == 0)
    coalesced_send_buffers.emplace(std::make_pair(
        key, CoalescedBuffersRank(nb.rank, b_type, lane, true,
                                  comms[GetAssociatedSender(b_type)], pmesh)));
  coalesced_send_buffers.at(key).AddSendBuffer(partition_tags.at(partition), pmb, nb,
                                               var);
}

//----------------------------------------------------------------------------------------
void CoalescedComms::AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                                   const std::shared_ptr<Variable<Real>>,
                                   BoundaryType b_type, int lane) {
  // We don't actually know enough here to register this particular buffer, but we do
  // know that it's existence implies that we need to receive a message from the
  // neighbor block rank eventually telling us the details
  const rank_key_t key{b_type, lane, -1, nb.rank};
  if (coalesced_recv_buffers.count(key) == 0)
    coalesced_recv_buffers.emplace(std::make_pair(
        key, CoalescedBuffersRank(nb.rank, b_type, lane, false,
                                  comms[GetAssociatedSender(b_type)], pmesh)));
}

//----------------------------------------------------------------------------------------
void CoalescedComms::ResolveAndSendSendBuffers() {
  // Provider lanes send their layouts in SyncProviderLane
  for (auto &[id, buf] : coalesced_send_buffers)
    if (!buf.OnProviderLane()) buf.ResolveAndSendBufInfo();
}

//----------------------------------------------------------------------------------------
int CoalescedComms::LaneInfoTag(BoundaryType b_type, int lane, int provider_lane) const {
  return LaneInfoTagOf(b_type, lane, provider_lane);
}

//----------------------------------------------------------------------------------------
void CoalescedComms::SendLaneInfo(CoalescedBuffersRank *cbr, std::vector<int> info) {
  const rank_key_t key{cbr->b_type, cbr->lane, cbr->provider_lane, cbr->other_rank};
  auto eit = lane_info_epochs.find(key);
  const int epoch = eit == lane_info_epochs.end() ? 0 : eit->second + 1;
  lane_info_epochs[key] = epoch;
  cbr->info_epoch = epoch;
  info.insert(info.begin(), epoch);
#ifdef MPI_PARALLEL
  for (auto it = lane_infos_in_flight.begin(); it != lane_infos_in_flight.end();) {
    int flag;
    PARTHENON_MPI_CHECK(MPI_Test(&(it->second), &flag, MPI_STATUS_IGNORE));
    it = flag ? lane_infos_in_flight.erase(it) : std::next(it);
  }
  lane_infos_in_flight.emplace_back(std::move(info), MPI_REQUEST_NULL);
  auto &[buf, req] = lane_infos_in_flight.back();
  PARTHENON_MPI_CHECK(MPI_Isend(buf.data(), buf.size(), MPI_INT, cbr->other_rank,
                                cbr->info_tag, lane_info_comm, &req));
#endif
  lane_infos_sent[cbr->provider_lane]++;
}

//----------------------------------------------------------------------------------------
void CoalescedComms::SyncProviderLane(int provider_lane) {
  PARTHENON_INSTRUMENT
  PARTHENON_REQUIRE(provider_lane >= 0 && provider_lane < kMaxCoalescedProviderLanes,
                    "Provider lane id out of range for coalesced comms.");
  // Senders: refill the layouts of the existing entries (keeping their combined buffers)
  // and add the new ones; send a lane info wherever a layout changed
  for (auto b_type : {BoundaryType::lane, BoundaryType::lane_flxcor_send}) {
    const int ncl = pmesh->GetNumberOfCommLanes(b_type);
    std::map<rank_key_t, std::vector<int>> old_info;
    for (int cl = 0; cl < ncl; ++cl) {
      for (auto it = EntriesBegin(coalesced_send_buffers, b_type, cl, provider_lane);
           InEntries(it, coalesced_send_buffers.end(), b_type, cl, provider_lane); ++it) {
        old_info[it->first] = it->second.SerializeLaneInfo();
        for (auto &[partition_tag, cb] : it->second.coalesced_bufs)
          cb.ClearVarBoundaries();
      }
    }
    for (auto &partition : pmesh->GetDefaultBlockPartitions()) {
      RegisterPartition(partition.get());
      const int partition_tag = partition_tags.at(partition.get());
      auto &md = pmesh->mesh_data.Add("base", partition);
      auto add = [&](auto pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &v) {
        for (int cl = 0; cl < ncl; ++cl) {
          const rank_key_t key{b_type, cl, provider_lane, nb.rank};
          auto it = coalesced_send_buffers.find(key);
          if (it == coalesced_send_buffers.end())
            it = coalesced_send_buffers
                     .emplace(key, CoalescedBuffersRank(nb.rank, b_type, cl, true,
                                                        comms.at(b_type), pmesh,
                                                        provider_lane, lane_info_comm))
                     .first;
          it->second.GetPartition(partition_tag).AddVarBoundary(pmb, nb, v);
        }
      };
      if (b_type == BoundaryType::lane) {
        ForEachOffRankLaneBoundary<BoundaryType::lane>(md, provider_lane, add);
      } else {
        ForEachOffRankLaneBoundary<BoundaryType::lane_flxcor_send>(md, provider_lane,
                                                                   add);
      }
    }
    for (int cl = 0; cl < ncl; ++cl) {
      auto it = EntriesBegin(coalesced_send_buffers, b_type, cl, provider_lane);
      while (InEntries(it, coalesced_send_buffers.end(), b_type, cl, provider_lane)) {
        auto &cbr = it->second;
        for (auto pit = cbr.coalesced_bufs.begin(); pit != cbr.coalesced_bufs.end();) {
          if (pit->second.TotalBuffers() > 0) {
            ++pit;
            continue;
          }
          WaitForWrite(pit->second);
          pit = cbr.coalesced_bufs.erase(pit);
        }
        if (cbr.coalesced_bufs.empty()) {
          it = coalesced_send_buffers.erase(it);
          continue;
        }
        auto info = cbr.SerializeLaneInfo();
        const auto old = old_info.find(it->first);
        if (old == old_info.end() || old->second != info)
          SendLaneInfo(&cbr, std::move(info));
        cbr.buffers_built = true;
        ++it;
      }
    }
  }

  // Receivers: one entry per rank with receive buffers on the lane, each holding the
  // keys of those buffers; layouts arrive with the next exchange
  auto &base = pmesh->mesh_data.Add("base", pmesh->GetBasePartition());
  for (auto b_type : {BoundaryType::lane, BoundaryType::lane_flxcor_recv}) {
    const int ncl = pmesh->GetNumberOfCommLanes(b_type);
    std::map<rank_key_t, std::vector<Mesh::channel_key_t>> expected;
    auto add = [&](auto pmb, const NeighborBlock &nb,
                   const std::shared_ptr<Variable<Real>> &v) {
      for (int cl = 0; cl < ncl; ++cl)
        expected[{b_type, cl, provider_lane, nb.rank}].push_back(
            ReceiveKey(pmb, nb, v, b_type, cl));
    };
    if (b_type == BoundaryType::lane) {
      ForEachOffRankLaneBoundary<BoundaryType::lane>(base, provider_lane, add);
    } else {
      ForEachOffRankLaneBoundary<BoundaryType::lane_flxcor_recv>(base, provider_lane,
                                                                 add);
    }
    for (int cl = 0; cl < ncl; ++cl) {
      auto it = EntriesBegin(coalesced_recv_buffers, b_type, cl, provider_lane);
      while (InEntries(it, coalesced_recv_buffers.end(), b_type, cl, provider_lane))
        it = expected.count(it->first) ? std::next(it) : coalesced_recv_buffers.erase(it);
    }
    for (auto &[key, keys] : expected) {
      auto it = coalesced_recv_buffers.find(key);
      if (it == coalesced_recv_buffers.end())
        it = coalesced_recv_buffers
                 .emplace(key, CoalescedBuffersRank(std::get<3>(key), b_type,
                                                    std::get<1>(key), false,
                                                    comms.at(GetAssociatedSender(b_type)),
                                                    pmesh, provider_lane, lane_info_comm))
                 .first;
      std::sort(keys.begin(), keys.end());
      it->second.expected_keys = std::move(keys);
      it->second.CheckLaneLayout();
    }
  }
}

//----------------------------------------------------------------------------------------
void CoalescedComms::ReceiveBufferInfo() {
  constexpr std::int64_t max_it = 1e10;
  std::vector<bool> received(coalesced_recv_buffers.size(), false);
  bool all_received;
  std::int64_t receive_iters = 0;
  do {
    all_received = true;
    for (auto &[id, buf] : coalesced_recv_buffers)
      if (!buf.OnProviderLane()) all_received = buf.TryReceiveBufInfo() && all_received;
    receive_iters++;
  } while (!all_received && receive_iters < max_it);
  PARTHENON_REQUIRE(
      receive_iters < max_it,
      "Too many iterations waiting to receive boundary communication buffers.");
}

//----------------------------------------------------------------------------------------
void CoalescedComms::PrintCensus() {
  const int build = comms_trace::CurrentBuild();

  // One row per coalesced message: src, dst, b_type, lane, partition_tag, nbuf, size
  // (in Reals, assuming every variable is communicated and allocated)
  constexpr int NR = 7;
  std::vector<int> local;
  for (auto &[key, cbr] : coalesced_send_buffers) {
    for (auto &[ptag, cb] : cbr.coalesced_bufs) {
      local.insert(local.end(),
                   {Globals::my_rank, cb.other_rank, static_cast<int>(cb.b_type), cb.lane,
                    ptag, static_cast<int>(cb.TotalBuffers()), cb.current_size});
    }
  }
  const int nranks = Globals::nranks;
  std::vector<int> all = local;
#ifdef MPI_PARALLEL
  int nlocal = local.size();
  std::vector<int> counts(nranks), displs(nranks);
  PARTHENON_MPI_CHECK(
      MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD));
  if (Globals::my_rank == 0) {
    int tot = 0;
    for (int r = 0; r < nranks; ++r) {
      displs[r] = tot;
      tot += counts[r];
    }
    all.resize(tot);
  }
  PARTHENON_MPI_CHECK(MPI_Gatherv(local.data(), nlocal, MPI_INT, all.data(),
                                  counts.data(), displs.data(), MPI_INT, 0,
                                  MPI_COMM_WORLD));
#endif
  if (Globals::my_rank != 0) return;

  auto btype_name = [](int b) {
    switch (static_cast<BoundaryType>(b)) {
    case BoundaryType::local:
      return "local";
    case BoundaryType::nonlocal:
      return "nonlocal";
    case BoundaryType::any:
      return "any";
    case BoundaryType::flxcor_send:
      return "flxcor_send";
    case BoundaryType::flxcor_recv:
      return "flxcor_recv";
    case BoundaryType::gmg_same:
      return "gmg_same";
    case BoundaryType::gmg_restrict_send:
      return "gmg_restrict_send";
    case BoundaryType::gmg_restrict_recv:
      return "gmg_restrict_recv";
    case BoundaryType::gmg_prolongate_send:
      return "gmg_prolongate_send";
    case BoundaryType::gmg_prolongate_recv:
      return "gmg_prolongate_recv";
    case BoundaryType::lane:
      return "lane";
    case BoundaryType::lane_flxcor_send:
      return "lane_flxcor_send";
    case BoundaryType::lane_flxcor_recv:
      return "lane_flxcor_recv";
    }
    return "unknown";
  };
  const double kib = sizeof(Real) / 1024.0;
  const int nmsg = all.size() / NR;
  auto row = [&](int m) { return &all[NR * m]; };
  auto on_node = [](const int *r) {
    return comms_trace::NodeOfRank(r[0]) == comms_trace::NodeOfRank(r[1]);
  };

  std::printf("\n==== Coalesced comms census (build %d): %d ranks on %d nodes, %d "
              "messages per exchange of all variables ====\n",
              build, nranks, comms_trace::NumNodes(), nmsg);
  std::printf(
      "Each message is accompanied by a status message of nbuf ints. Sizes assume "
      "every variable is communicated.\n");
  for (int n = 0; n < comms_trace::NumNodes(); ++n) {
    std::printf("  node %d:", n);
    std::string host;
    for (int r = 0; r < nranks; ++r) {
      if (comms_trace::NodeOfRank(r) == n) {
        std::printf(" %d", r);
        host = comms_trace::HostOfRank(r);
      }
    }
    std::printf("  (%s)\n", host.c_str());
  }

  // Per rank, per boundary type totals
  struct Tot {
    int nmsg[2]{0, 0};
    double kib[2]{0.0, 0.0};
    int nbuf[2]{0, 0};
    double min_kib{1e300}, max_kib{0.0};
  };
  std::map<std::pair<int, int>, Tot> tots;
  for (int m = 0; m < nmsg; ++m) {
    const int *r = row(m);
    auto &t = tots[{r[0], r[2]}];
    const int off = on_node(r) ? 0 : 1;
    const double sz = r[6] * kib;
    t.nmsg[off]++;
    t.kib[off] += sz;
    t.nbuf[off] += r[5];
    t.min_kib = std::min(t.min_kib, sz);
    t.max_kib = std::max(t.max_kib, sz);
  }
  std::printf("\n%6s %-20s | %8s %10s %8s | %8s %10s %8s | %10s %10s\n", "rank", "btype",
              "on_msgs", "on_KiB", "on_nbuf", "off_msgs", "off_KiB", "off_nbuf",
              "min_KiB", "max_KiB");
  for (const auto &[k, t] : tots) {
    std::printf("%6d %-20s | %8d %10.1f %8d | %8d %10.1f %8d | %10.2f %10.2f\n", k.first,
                btype_name(k.second), t.nmsg[0], t.kib[0], t.nbuf[0], t.nmsg[1], t.kib[1],
                t.nbuf[1], t.min_kib, t.max_kib);
  }

  // Histogram of message sizes in power of two KiB buckets
  std::map<int, std::array<int, 2>> hist;
  for (int m = 0; m < nmsg; ++m) {
    const int *r = row(m);
    const double sz = r[6] * kib;
    int b = 0;
    while (std::pow(2.0, b) < sz)
      b++;
    if (sz < 1.0) b = 0;
    hist[b][on_node(r) ? 0 : 1]++;
  }
  std::printf("\nMessage size histogram (KiB bucket upper bound)\n%12s %8s %8s\n",
              "<= KiB", "on_node", "off_node");
  for (const auto &[b, h] : hist)
    std::printf("%12.0f %8d %8d\n", std::pow(2.0, b), h[0], h[1]);

  if (comms_trace::CensusPrintMessages()) {
    std::printf("\n%6s %6s %4s %-20s %4s %6s %6s %10s\n", "src", "dst", "node", "btype",
                "lane", "ptag", "nbuf", "KiB");
    for (int m = 0; m < nmsg; ++m) {
      const int *r = row(m);
      std::printf("%6d %6d %4s %-20s %4d %6d %6d %10.2f\n", r[0], r[1],
                  on_node(r) ? "on" : "off", btype_name(r[2]), r[3], r[4], r[5],
                  r[6] * kib);
    }
  }
  std::printf("==== End census ====\n\n");
  std::fflush(stdout);

  const std::string fname =
      comms_trace::Prefix() + ".census." + std::to_string(build) + ".csv";
  FILE *f = std::fopen(fname.c_str(), "w");
  if (f != nullptr) {
    std::fprintf(f, "src,dst,src_node,dst_node,btype,lane,partition_tag,nbuf,bytes\n");
    for (int m = 0; m < nmsg; ++m) {
      const int *r = row(m);
      std::fprintf(f, "%d,%d,%d,%d,%s,%d,%d,%d,%lld\n", r[0], r[1],
                   comms_trace::NodeOfRank(r[0]), comms_trace::NodeOfRank(r[1]),
                   btype_name(r[2]), r[3], r[4], r[5],
                   static_cast<long long>(r[6]) * static_cast<long long>(sizeof(Real)));
    }
    std::fclose(f);
  }
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type,
                                         int provider_lane) {
  const int lane = pmd->GetCommLane(b_type);
  bool available{true};
  for (auto it = EntriesBegin(coalesced_send_buffers, b_type, lane, provider_lane);
       InEntries(it, coalesced_send_buffers.end(), b_type, lane, provider_lane); ++it)
    available = available && it->second.IsAvailableForWrite(pmd);
  return available;
}

//----------------------------------------------------------------------------------------
void CoalescedComms::PackAndSend(MeshData<Real> *pmd, BoundaryType b_type,
                                 int provider_lane) {
  const int lane = pmd->GetCommLane(b_type);
  for (auto it = EntriesBegin(coalesced_send_buffers, b_type, lane, provider_lane);
       InEntries(it, coalesced_send_buffers.end(), b_type, lane, provider_lane); ++it)
    it->second.PackAndSend(pmd);
}

//----------------------------------------------------------------------------------------
bool CoalescedComms::TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type,
                                   int provider_lane) {
#ifdef MPI_PARALLEL
  const int lane = pmd->GetCommLane(b_type);
  bool all_received = true;
  for (auto it = EntriesBegin(coalesced_recv_buffers, b_type, lane, provider_lane);
       InEntries(it, coalesced_recv_buffers.end(), b_type, lane, provider_lane); ++it) {
    auto &coal_bufs = it->second;
    if (coal_bufs.OnProviderLane()) {
      all_received = coal_bufs.TryReceiveLane() && all_received;
      continue;
    }
    for (auto &[partition_tag, coal_buf] : coal_bufs.coalesced_bufs) {
      bool received = coal_buf.TryReceiveAndUnpack();
      all_received = all_received && received;
    }
  }
  return all_received;
#else
  // There are no coalesced comms without MPI, so all buffers
  // are automatically received
  return true;
#endif
}
} // namespace parthenon

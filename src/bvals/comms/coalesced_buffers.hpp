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

#ifndef BVALS_COMMS_COALESCED_BUFFERS_HPP_
#define BVALS_COMMS_COALESCED_BUFFERS_HPP_

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "bvals/comms/bnd_id.hpp"
#include "bvals/comms/bvals_utils.hpp"
#include "bvals/neighbor_block.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/variable.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/comms_trace.hpp"
#include "utils/communication_buffer.hpp"

namespace parthenon {

// Coalesced comms support provider lane ids in [0, kMaxCoalescedProviderLanes)
constexpr int kMaxCoalescedProviderLanes = 64;

// Structure containing the information required for sending coalesced
// messages between ranks
struct CoalescedBuffer {
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;
  // Tag of the sending partition (see CoalescedComms::partition_tags), which is combined
  // with the comm lane to build the MPI tags of the messages
  int partition_tag;
  int lane;
  // Provider lane of the buffers, -1 off provider lanes. On provider lanes the status
  // message starts with the epoch of the sender's lane info, and each buffer's entry is
  // its payload length (0 for an empty or null buffer).
  int provider_lane;
  mpi_comm_t comm;
  Mesh *pmesh;
  bool sender;

  using var_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  std::map<Uid_t, std::vector<std::pair<BndId, var_buf_t *>>> coalesced_info_buf;
  std::set<Uid_t> all_vars;
  std::unordered_map<std::set<Uid_t>, ParArray1DRaw<BndId>> bnd_ids_device_map;
  std::unordered_map<std::set<Uid_t>, ParArray1DRaw<BndId>::host_mirror_type>
      bnd_ids_host_map;
  CommBuffer<buf_t> coalesced_comm_buffer;
  // One entry per buffer: 1 if allocated, 0 if null, and not_in_message if its variable
  // was not sent (messages can contain a subset of the variables, see PackAndSend).
  // On provider lanes the entries follow an epoch header and hold payload lengths.
  CommBuffer<std::vector<int>> sparse_status_buffer;
  static constexpr int not_in_message = -1;
  int current_size;

  // Tracing of the exchange in flight (see utils/comms_trace.hpp)
  comms_trace::ExchangeRecord trace_rec;
  // Stored record of the last send, whose completion time is filled in lazily
  comms_trace::ExchangeRecord *trace_send_rec{nullptr};
  bool trace_open{false};
  int trace_nexchange{0};
  double trace_last_poll{0.0};
  void TraceOpen(double t0);
  void TraceClose();

  CoalescedBuffer(bool sender, int partition_tag, int lane, int provider_lane,
                  int other_rank, BoundaryType b_type, mpi_comm_t comm, Mesh *pmesh);

  bool OnProviderLane() const { return provider_lane >= 0; }
  int StatusHeader() const { return OnProviderLane() ? 1 : 0; }

  std::size_t TotalBuffers() const {
    std::size_t total_buffers{0};
    for (const auto &[uid, v] : coalesced_info_buf)
      total_buffers += v.size();
    return total_buffers;
  }

  void AddVarBoundary(BndId &bnd_id);
  void ClearVarBoundaries();

  void AddVarBoundary(MeshBlock *pmb, const NeighborBlock &nb,
                      const std::shared_ptr<Variable<Real>> &var);

  bool IsAvailableForWrite() {
    const bool available = sparse_status_buffer.IsAvailableForWrite() &&
                           coalesced_comm_buffer.IsAvailableForWrite();
    if (available && trace_send_rec != nullptr) {
      if (comms_trace::Recording()) trace_send_rec->t[4] = comms_trace::Now();
      trace_send_rec = nullptr;
    }
    return available;
  }

  ParArray1DRaw<BndId> &GetBndIdsOnDevice(const std::set<Uid_t> &vars,
                                          int *pcomb_size = nullptr);

  // epoch: the sender's lane info epoch (provider lanes only)
  void PackAndSend(const std::set<Uid_t> &vars, int epoch = 0);

  // Provider lanes: receives the status message only; true once it has arrived
  bool TryReceiveStatus();
  int StatusEpoch() { return sparse_status_buffer.buffer()[0]; }

  // On provider lanes only if the status message carries epoch
  bool TryReceiveAndUnpack(int epoch = 0);
};

struct CoalescedBuffersRank {
  using coalesced_message_structure_t = std::vector<BndId>;
  using buf_t = BufArray1D<Real>;

  // Rank that these buffers communicate with
  BoundaryType b_type;
  int other_rank;
  int lane;
  int provider_lane{-1};

  bool buffers_built{false};

  // Map from the sender's partition tag (see CoalescedComms::partition_tags) to the
  // coalesced buffer associated with that partition
  std::map<int, CoalescedBuffer> coalesced_bufs;

  static constexpr int nglobal{1};
  static constexpr int nper_part{3};

  using com_buf_t = CommBuffer<std::vector<int>>;
  com_buf_t message;

  mpi_comm_t comm;
  Mesh *pmesh;
  bool sender{true};

  // Provider lanes have no info handshake: the sender sends a lane info message (the
  // epoch, then the partitions as in ResolveAndSendBufInfo) whenever its buffers to
  // other_rank change, and its status messages carry that epoch. The receiver applies
  // the info whose epoch a status message names, so it never waits in a rebuild.
  // Sender: epoch of the last info sent. Receiver: epoch of the info the current
  // layout comes from (-1 for none), and whether that layout lists exactly the receive
  // buffers of this rank (expected_keys).
  int info_epoch{-1};
  bool info_valid{false};
  std::map<int, std::vector<int>> pending_infos;
  std::vector<Mesh::channel_key_t> expected_keys;
  mpi_comm_t info_comm;
  int info_tag{0};

  explicit CoalescedBuffersRank(int o_rank, BoundaryType b_type, int lane, bool send,
                                mpi_comm_t comm, Mesh *pmesh, int provider_lane = -1,
                                mpi_comm_t info_comm = mpi_comm_t());

  bool OnProviderLane() const { return provider_lane >= 0; }
  CoalescedBuffer &GetPartition(int partition_tag);
  // The lane info message without its epoch
  std::vector<int> SerializeLaneInfo() const;
  void DrainLaneInfos();
  void ApplyLaneInfo(int epoch);
  // Re-resolves the buffers of the current layout after a lane rebuild
  void CheckLaneLayout();
  bool TryReceiveLane();

  void AddSendBuffer(int partition_tag, MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>> &var);

  bool TryReceiveBufInfo();

  void ResolveAndSendBufInfo();

  void PackAndSend(MeshData<Real> *pmd);

  bool TryReceiveAndUnpack(MeshData<Real> *pmd, int partition_tag);

  bool IsAvailableForWrite(MeshData<Real> *pmd);
};

struct CoalescedComms {
  // Combined buffers for each (BoundaryType, comm lane, provider lane, rank); the
  // provider lane is -1 off provider lanes
  using rank_key_t = std::tuple<BoundaryType, int, int, int>;
  std::map<rank_key_t, CoalescedBuffersRank> coalesced_send_buffers;
  std::map<rank_key_t, CoalescedBuffersRank> coalesced_recv_buffers;

  // Map from each registered partition to an integer tag that is unique on this rank.
  // Receiving ranks learn these tags from the buffer info messages.
  std::map<const BlockListPartition *, int> partition_tags;

  std::map<BoundaryType, mpi_comm_t> comms;
  // Lane info messages (provider lanes only)
  mpi_comm_t lane_info_comm;
  // Last lane info epoch sent per key; kept across clear() so that epochs only grow
  std::map<rank_key_t, int> lane_info_epochs;
  // Lane info sends in flight; their receivers read them lazily
  std::list<std::pair<std::vector<int>, mpi_request_t>> lane_infos_in_flight;
  // Lane info messages sent by this rank, per provider lane
  std::map<int, std::int64_t> lane_infos_sent;

  Mesh *pmesh;

  explicit CoalescedComms(Mesh *pmesh);

  ~CoalescedComms();

  void clear();

  void RegisterPartition(const BlockListPartition *partition);

  void AddSendBuffer(const BlockListPartition *partition, MeshBlock *pmb,
                     const NeighborBlock &nb, const std::shared_ptr<Variable<Real>> &var,
                     BoundaryType b_type, int lane);

  void AddRecvBuffer(MeshBlock *pmb, const NeighborBlock &nb,
                     const std::shared_ptr<Variable<Real>>, BoundaryType b_type,
                     int lane);

  // Brings the coalesced buffers of provider lane provider_lane in line with its
  // current buffers (after the lane is built or rebuilt), and sends a lane info message
  // to each rank whose buffers changed. Never blocks on other ranks.
  void SyncProviderLane(int provider_lane);
  void SendLaneInfo(CoalescedBuffersRank *cbr, std::vector<int> info);
  int LaneInfoTag(BoundaryType b_type, int lane, int provider_lane) const;

  void ResolveAndSendSendBuffers();

  void ReceiveBufferInfo();

  // Print the size and number of all coalesced messages sent by every rank. Collective.
  void PrintCensus();

  // provider_lane: -1 off provider lanes
  void PackAndSend(MeshData<Real> *pmd, BoundaryType b_type, int provider_lane = -1);

  bool TryReceiveAny(MeshData<Real> *pmd, BoundaryType b_type, int provider_lane = -1);

  bool IsAvailableForWrite(MeshData<Real> *pmd, BoundaryType b_type,
                           int provider_lane = -1);

  // Get the tag of the partition pmd was built from
  int GetPartitionTag(MeshData<Real> *pmd) const;
};

} // namespace parthenon

#endif // BVALS_COMMS_COALESCED_BUFFERS_HPP_

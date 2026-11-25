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
#ifndef EXAMPLE_PROVIDER_LANES_PROVIDER_LANES_HPP_
#define EXAMPLE_PROVIDER_LANES_PROVIDER_LANES_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <bvals/comms/bnd_ops.hpp>
#include <bvals/neighbor_block.hpp>
#include <parthenon/package.hpp>

namespace provider_lanes {
using namespace parthenon::package::prelude;
using parthenon::LaneId;

// Two lanes on a hyper-rectangular pool of blocks whose connectivity has nothing to do
// with the blocks' positions:
//  - groups: in each of kLayers layers the blocks are shuffled and cut into groups of
//    2 to 6 blocks (irregular "nodes"); a block is a lane neighbor of every other block
//    of each of its groups. Its channel for layer L is (L + gid) % kLayers, so the two
//    ends of a channel use different channel ids. The graph is reshuffled mid-run
//    (version 1) and the lane rebuilt with Mesh::RebuildLane.
//  - ring: the blocks are shuffled into a ring; channel 0 is the previous block,
//    channel 1 the next. Never rebuilt.
// One MeshData holds the variables of both lanes and an ordinary FillGhost variable u;
// it is exchanged lane by lane, and u by the geometric ghost exchange.
constexpr LaneId kGroups{0};
constexpr LaneId kRing{1};
constexpr int kLayers = 3;
constexpr int kMaxGroup = 6;

// The lane graphs, computed identically on every rank from (nblocks, version)
struct LaneGraph {
  LaneGraph(int nblocks, int version);
  int nblocks, version;
  // group[L][gid] = index of the gid's group in layer L; members[L][g] ascending gids
  std::vector<std::vector<int>> group;
  std::vector<std::vector<std::vector<int>>> members;
  // ring order: ring[p] = gid at position p, pos[gid] = its position
  std::vector<int> ring, pos;

  static int Channel(int layer, int gid) { return (layer + gid) % kLayers; }
  void GroupNeighbors(int gid, std::vector<parthenon::LaneNeighbor> &out) const;
  void RingNeighbors(int gid, std::vector<parthenon::LaneNeighbor> &out) const;
};

// Values each block sends, not exactly representable so sums depend on their order
KOKKOS_INLINE_FUNCTION Real RecordValue(int gid, int cycle) {
  return 1.0 / (gid + 3.0 + 0.37 * cycle);
}
KOKKOS_INLINE_FUNCTION Real RingValue(int gid, int cycle) {
  return 1.0 / (2.0 * gid + 5.0 + 0.13 * cycle);
}
KOKKOS_INLINE_FUNCTION Real FluxCandidate(int gid, int channel, int cycle) {
  return (channel + 1.0) / (gid + 7.0 + 0.11 * cycle);
}

// Pack op for per-block records: every buffer gets slot 0 of the sender's record.
struct RecordPack {
  static std::size_t BufferSize(const parthenon::MeshBlock &,
                                const parthenon::NeighborBlock &,
                                const parthenon::Variable<Real> &) {
    return 1;
  }
  static void Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                   const parthenon::BndInfoArrHost_t &info_h,
                   const parthenon::BndOpIdx_t &idx,
                   const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};

// Unpack op: slot 1 + c of the receiver's record gets the sum, in ascending sender gid,
// of all buffers that arrived on its channel c (one on the ring, one per other group
// member on the groups lane).
struct SumByChannel {
  static void Unpack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                     const parthenon::BndInfoArrHost_t &info_h,
                     const parthenon::BndOpIdx_t &idx,
                     const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};

// Lane flux exchange: every block holds one candidate flux per channel c, at the x1
// face (ks, js + c, is). The pack op sends it to the other group members; the combine
// op replaces it by the mean over the group, summed in ascending gid, so all members
// hold the same bits.
// Variable-length payloads on the groups lane: each sender sends RowsLength rows (at most
// the capacity, a package param that BufferSize reads and the driver raises before
// RebuildLane); the unpack op sums, per receiving channel, the rows and their number.
KOKKOS_INLINE_FUNCTION int RowsLength(int send_gid, int send_channel, int cycle,
                                      int capacity) {
  return 1 + (5 * send_gid + 3 * send_channel + 2 * cycle) % capacity;
}
KOKKOS_INLINE_FUNCTION Real RowValue(int send_gid, int row, int cycle) {
  return 1.0 / (send_gid + 2.0 + 0.1 * row + 0.37 * cycle);
}

struct RowsPack {
  static std::size_t BufferSize(const parthenon::MeshBlock &pmb,
                                const parthenon::NeighborBlock &,
                                const parthenon::Variable<Real> &);
  static void Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                   const parthenon::BndInfoArrHost_t &info_h,
                   const parthenon::BndOpIdx_t &idx,
                   const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers,
                   const parthenon::BndPayload_t &payload,
                   const parthenon::BndPayloadHost_t &payload_h);
};

struct RowsUnpack {
  static void Unpack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                     const parthenon::BndInfoArrHost_t &info_h,
                     const parthenon::BndOpIdx_t &idx,
                     const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers,
                     const parthenon::BndPayload_t &payload,
                     const parthenon::BndPayloadHost_t &payload_h);
};

struct FluxCandidatePack {
  static std::size_t BufferSize(const parthenon::MeshBlock &,
                                const parthenon::NeighborBlock &,
                                const parthenon::Variable<Real> &) {
    return 1;
  }
  static void Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                   const parthenon::BndInfoArrHost_t &info_h,
                   const parthenon::BndOpIdx_t &idx,
                   const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};
struct GroupMean {
  static void Combine(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                      const parthenon::BndInfoArrHost_t &info_h,
                      const parthenon::BndOpIdx_t &idx,
                      const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers);
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
// Sets the records' own slots, clears the received slots to -1, sets the flux
// candidates and u
TaskStatus SetValues(MeshData<Real> *md, int cycle);
void SetCycleAndCapacity(Mesh *pmesh, int cycle, int capacity);
int GetCapacity(Mesh *pmesh);

struct CheckResult {
  int nwrong = 0;
  int nchecked = 0;
  int group_sizes[kMaxGroup + 1] = {0};
  std::uint64_t hash = 0;
  std::int64_t rows_sent = 0, rows_capacity = 0;
};
// Checks the records, fluxes and u ghosts of every local block against the graph and
// the geometric neighbors (global sums)
CheckResult Check(Mesh *pmesh, const LaneGraph &graph, int cycle);

} // namespace provider_lanes

#endif // EXAMPLE_PROVIDER_LANES_PROVIDER_LANES_HPP_

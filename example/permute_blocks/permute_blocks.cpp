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

// A pool of blocks whose content is permuted over the gids every
// parthenon/loadbalancing/permute_interval cycles by a user callback
// (ApplicationInput::MeshBlockPermutation), with load-balance cuts kept on group
// boundaries (ApplicationInput::LoadBalanceGroups) and a provider lane rebuilt after
// every permutation. Each block content has an identity id; ids 3g, 3g+1, 3g+2 form
// group g (a "simplex"). A permutation shuffles the groups and reverses the order
// inside every group on every other permutation, so each group stays on consecutive
// gids. Every cycle checks bitwise, on every block, that
//  - the cell variable q still holds the values of its identity (data followed it),
//  - so does the ghost-less per-block record desc (Metadata::None), all of it,
//  - the ring lane (ids id-1 and id+1 are neighbors) delivered the right ids, i.e. the
//    lane was rebuilt for the new gids,
//  - no group is split across ranks.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <bvals/comms/bnd_ops.hpp>
#include <bvals/neighbor_block.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::BndInfo;
using parthenon::LaneId;

namespace {
constexpr LaneId kRing{0};
constexpr int kGroupSize = 3;
constexpr int kDescSize = 3;

std::uint64_t SplitMix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Content of identity id, not exactly representable
KOKKOS_INLINE_FUNCTION Real Value(int id, int k, int j, int i) {
  return 1.0 / (id + 3.0 + 0.01 * (i + 16 * j + 256 * k));
}

// The user's global state, identical on every rank: which identity sits on which gid
struct Pool {
  std::vector<int> id_of_gid, gid_of_id;
  int npermutations = 0;
  void EnsureSize(int n) {
    if (id_of_gid.size() == n) return;
    id_of_gid.resize(n);
    std::iota(id_of_gid.begin(), id_of_gid.end(), 0);
    gid_of_id = id_of_gid;
  }
  int Group(int gid) const { return id_of_gid[gid] / kGroupSize; }
  int nblocks() const { return id_of_gid.size(); }
};
Pool pool;

// Shuffles the groups (and reverses the members of each group on odd permutations);
// returns new_gid_of_old and updates the pool if update is set
std::vector<int> Permutation(int seed, bool update) {
  const int n = pool.nblocks();
  const int ngroups = (n + kGroupSize - 1) / kGroupSize;
  std::vector<int> order(ngroups);
  std::iota(order.begin(), order.end(), 0);
  std::uint64_t s = seed + 1;
  for (int i = ngroups - 1; i > 0; --i) {
    s = SplitMix64(s);
    std::swap(order[i], order[s % (i + 1)]);
  }
  const bool reverse = pool.npermutations % 2 == 0;
  std::vector<int> new_id_of_gid;
  for (int g : order) {
    std::vector<int> ids;
    for (int id = g * kGroupSize; id < std::min(n, (g + 1) * kGroupSize); ++id)
      ids.push_back(id);
    if (reverse) std::reverse(ids.begin(), ids.end());
    new_id_of_gid.insert(new_id_of_gid.end(), ids.begin(), ids.end());
  }
  std::vector<int> new_gid_of_id(n), new_gid_of_old(n);
  for (int g = 0; g < n; ++g)
    new_gid_of_id[new_id_of_gid[g]] = g;
  for (int o = 0; o < n; ++o)
    new_gid_of_old[o] = new_gid_of_id[pool.id_of_gid[o]];
  if (update) {
    pool.id_of_gid = std::move(new_id_of_gid);
    pool.gid_of_id = std::move(new_gid_of_id);
  }
  pool.npermutations++;
  return new_gid_of_old;
}

// Ring lane record: slot 0 the block's id, slot 1 + c the id received on channel c
// (0 = id - 1, 1 = id + 1)
struct RecordPack {
  static std::size_t BufferSize(const parthenon::MeshBlock &,
                                const parthenon::NeighborBlock &,
                                const parthenon::Variable<Real> &) {
    return 1;
  }
  static void Pack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                   const parthenon::BndInfoArrHost_t &info_h,
                   const parthenon::BndOpIdx_t &idx,
                   const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
    Kokkos::parallel_for(
        "RecordPack", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nbuffers),
        KOKKOS_LAMBDA(const int n) {
          const BndInfo &bi = info(idx(n));
          bi.buf(0) = bi.var(0, 0, 0, 0, 0, 0, 0);
        });
  }
};
struct RecordUnpack {
  static void Unpack(MeshData<Real> *md, const parthenon::BndInfoArr_t &info,
                     const parthenon::BndInfoArrHost_t &info_h,
                     const parthenon::BndOpIdx_t &idx,
                     const parthenon::BndOpIdxHost_t &idx_h, std::size_t nbuffers) {
    Kokkos::parallel_for(
        "RecordUnpack", Kokkos::RangePolicy<>(parthenon::DevExecSpace(), 0, nbuffers),
        KOKKOS_LAMBDA(const int n) {
          const BndInfo &bi = info(idx(n));
          bi.var(0, 0, 0, 0, 0, 0, 1 + bi.channel) = bi.buf(0);
        });
  }
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("permute_blocks");
  // The block content that must follow its identity (moved by PermuteBlocks)
  pkg->AddField("q", Metadata({Metadata::Cell, Metadata::Independent}));
  // A per-block record without ghost zones that must follow its identity too, set only
  // in the problem generator
  pkg->AddField("desc", Metadata({Metadata::None, Metadata::Independent},
                                 std::vector<int>{kDescSize}));
  Metadata mr({Metadata::None, Metadata::Derived, Metadata::OneCopy},
              std::vector<int>{3});
  mr.SetLane(kRing);
  mr.RegisterBoundaryOps<RecordUnpack, RecordPack>();
  pkg->AddField("ring_rec", mr);
  return pkg;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *) {
  pool.EnsureSize(pmb->pmy_mesh->nbtotal);
  const int id = pool.id_of_gid[pmb->gid];
  auto q = pmb->meshblock_data.Get()->Get("q").data.Get<3>();
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, q.extent_int(0) - 1, 0, q.extent_int(1) - 1, 0,
      q.extent_int(2) - 1, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        q(k, j, i) = Value(id, k, j, i);
      });
  auto desc = pmb->meshblock_data.Get()->Get("desc").data;
  parthenon::par_for(
      PARTHENON_AUTO_LABEL, 0, kDescSize - 1, KOKKOS_LAMBDA(const int c) {
        desc(0, 0, 0, 0, 0, 0, c) = Value(id, 0, 0, 1000 + c);
      });
}

TaskStatus SetRecords(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto &rc = md->GetBlockData(b);
    const Real id = pool.id_of_gid[rc->GetBlockPointer()->gid];
    auto rec = rc->Get("ring_rec").data.Get();
    parthenon::par_for(
        PARTHENON_AUTO_LABEL, 0, 0, KOKKOS_LAMBDA(const int) {
          rec(0, 0, 0, 0, 0, 0, 0) = id;
          rec(0, 0, 0, 0, 0, 0, 1) = -1.0;
          rec(0, 0, 0, 0, 0, 0, 2) = -1.0;
        });
  }
  return TaskStatus::complete;
}

class PermuteBlocksDriver : public EvolutionDriver {
 public:
  PermuteBlocksDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
      : EvolutionDriver(pin, app_in, pm) {
    pmesh->RegisterNeighborProvider(kRing, [](const parthenon::MeshBlock &pmb,
                                              std::vector<parthenon::LaneNeighbor> &nbs) {
      const int n = pool.nblocks();
      const int id = pool.id_of_gid[pmb.gid];
      nbs.push_back({pool.gid_of_id[(id + n - 1) % n], 0, 1});
      nbs.push_back({pool.gid_of_id[(id + 1) % n], 1, 0});
    });
  }

  TaskListStatus Step() override {
    TaskCollection tc;
    const int num_partitions = pmesh->DefaultNumPartitions();
    TaskRegion &region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &tl = region[i];
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      auto &md_ring = pmesh->mesh_data.Add("lane_ring", md, {"ring_rec"});
      auto set = tl.AddTask(TaskID(0), SetRecords, md.get());
      parthenon::AddLaneExchangeTasks(set, tl, md_ring, kRing);
    }
    const auto status = tc.Execute();
    Check();
    return status;
  }

  int nwrong = 0;

 private:
  void Check() {
    const int n = pool.nblocks();
    int counts[3] = {0, 0, 0}; // wrong values, checked values, split groups
    std::uint64_t hash = 0;
    auto check = [&](Real got, Real expected, std::uint64_t tag) {
      std::uint64_t bits;
      std::memcpy(&bits, &got, sizeof(bits));
      hash += SplitMix64(bits ^ SplitMix64(tag));
      counts[1]++;
      if (std::memcmp(&got, &expected, sizeof(Real)) != 0) counts[0]++;
    };
    for (auto &pmb : pmesh->block_list) {
      auto &rc = pmb->meshblock_data.Get();
      const int gid = pmb->gid;
      const int id = pool.id_of_gid[gid];
      const std::uint64_t tag = 1024 * static_cast<std::uint64_t>(gid);
      auto q = rc->Get("q").data.GetHostMirrorAndCopy();
      const IndexRange ib = rc->GetBoundsI(IndexDomain::interior);
      const IndexRange jb = rc->GetBoundsJ(IndexDomain::interior);
      const IndexRange kb = rc->GetBoundsK(IndexDomain::interior);
      for (int k = kb.s; k <= kb.e; ++k)
        for (int j = jb.s; j <= jb.e; ++j)
          for (int i = ib.s; i <= ib.e; ++i)
            check(q(0, 0, 0, 0, k, j, i), Value(id, k, j, i), tag + i + 16 * j);
      auto desc = rc->Get("desc").data.GetHostMirrorAndCopy();
      for (int c = 0; c < kDescSize; ++c)
        check(desc(0, 0, 0, 0, 0, 0, c), Value(id, 0, 0, 1000 + c), tag + 1010 + c);
      auto rec = rc->Get("ring_rec").data.GetHostMirrorAndCopy();
      check(rec(0, 0, 0, 0, 0, 0, 1), (id + n - 1) % n, tag + 1000);
      check(rec(0, 0, 0, 0, 0, 0, 2), (id + 1) % n, tag + 1001);
    }
    // Groups at the ends of this rank's gid range must not continue on another rank
    if (!pmesh->block_list.empty()) {
      const int first = pmesh->block_list.front()->gid;
      const int last = pmesh->block_list.back()->gid;
      if (first > 0 && pool.Group(first) == pool.Group(first - 1)) counts[2]++;
      if (last < n - 1 && pool.Group(last) == pool.Group(last + 1)) counts[2]++;
    }
#ifdef MPI_PARALLEL
    MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &hash, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    nwrong += counts[0] + counts[2];
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "permute_blocks: cycle " << tm.ncycle << ", " << pool.npermutations
                << " permutations, blocks per rank";
      for (int nb : pmesh->GetNbList())
        std::cout << " " << nb;
      std::cout << ", " << counts[1] << " values checked, " << counts[0] << " wrong, "
                << counts[2] << " split groups, hash " << std::hex << hash << std::dec
                << std::endl;
    }
  }
};
} // namespace

int main(int argc, char *argv[]) {
  ParthenonManager pman;
  pman.app_input->ProcessPackages = [](std::unique_ptr<ParameterInput> &pin) {
    Packages_t packages;
    packages.Add(Initialize(pin.get()));
    return packages;
  };
  pman.app_input->ProblemGenerator = ProblemGenerator;
  // Keep each group's blocks on one rank
  pman.app_input->LoadBalanceGroups =
      [](Mesh *, const std::vector<parthenon::LogicalLocation> &locs) {
        pool.EnsureSize(locs.size());
        std::vector<int> groups(locs.size());
        for (int g = 0; g < locs.size(); ++g)
          groups[g] = pool.Group(g);
        return groups;
      };
  // update_state = false permutes the blocks without updating the pool: the check fails
  pman.app_input->MeshBlockPermutation = [](Mesh *, ParameterInput *pin,
                                            const parthenon::SimTime &tm) {
    return Permutation(tm.ncycle,
                       pin->GetOrAddBoolean("permute_blocks", "update_state", true));
  };
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  pman.ParthenonInitPackagesAndMesh();

  int nwrong = 0;
  {
    PermuteBlocksDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());
    driver.Execute();
    nwrong = driver.nwrong;
  }
  pman.ParthenonFinalize();
  if (parthenon::Globals::my_rank == 0)
    std::cout << "permute_blocks: " << (nwrong == 0 ? "PASS" : "FAIL") << std::endl;
  return nwrong == 0 ? 0 : 1;
}

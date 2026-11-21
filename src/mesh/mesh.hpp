//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_MESH_HPP_
#define MESH_MESH_HPP_
//! \file mesh.hpp
//  \brief defines Mesh and MeshBlock classes, and various structs used in them
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.

// This file was made in part with generative AI.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bvals/boundary_conditions.hpp"
#include "bvals/comms/tag_map.hpp"
#include "bvals/neighbor_block.hpp"
#include "config.hpp"
#include "coordinates/coordinates.hpp"
#include "defs.hpp"
#include "domain.hpp"
#include "interface/data_collection.hpp"
#include "interface/mesh_data.hpp"
#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/meshblock_pack.hpp"
#include "outputs/io_wrapper.hpp"
#include "pack/sparse_pack/pack_descriptor.hpp"
#include "parameter_input.hpp"
#include "parthenon_arrays.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/hash.hpp"
#include "utils/object_pool.hpp"
#include "utils/partition_stl_containers.hpp"

#ifdef PARTHENON_ENABLE_FFT
#include "utils/fft_manager.hpp"
#include "utils/uniform_grid_helper.hpp"
#endif

namespace parthenon {

// Forward declarations
class ApplicationInput;
class CoalescedComms;
class MeshBlock;
class MeshRefinement;
class Packages_t;
class ParameterInput;
class RestartReader;

// Map from LogicalLocation to (gid, rank) pair of location
using LogicalLocMap_t = std::map<LogicalLocation, std::pair<int, int>>;

// Base class to allow cacheing of different types of PackDescriptors

//----------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
  friend class RestartOutput;
  friend class HistoryOutput;
  friend class MeshBlock;
  friend class MeshRefinement;
  friend class FFTManager;

  struct base_constructor_selector_t {};
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       base_constructor_selector_t);
  struct hyper_rectangular_constructor_selector_t {};
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       hyper_rectangular_constructor_selector_t);

 public:
#ifdef PARTHENON_ENABLE_FFT
  std::unique_ptr<parthenon::FFTManager> fft_manager;

  FFTManager *GetFFTManager() {
    if (!fft_manager) {
      fft_manager = std::make_unique<FFTManager>(this);
    }
    return fft_manager.get();
  }

  std::unique_ptr<parthenon::UniformGridHelper> uniform_grid_helper;

  UniformGridHelper *GetUniformGridHelper() {
    if (!uniform_grid_helper) {
      uniform_grid_helper = std::make_unique<UniformGridHelper>(this);
    }
    return uniform_grid_helper.get();
  }
#endif
  // 2x function overloads of ctor: normal and restarted simulation
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, RestartReader &resfile,
       Packages_t &packages, int test_flag = 0);
  Mesh(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
       forest::ForestDefinition &forest_def);
  static RegionSize GetBaseMeshBlockSize(ParameterInput *pin,
                                         const RegionSize &mesh_size);
  static std::pair<RegionSize, RegionSize> GetRegionSizes(ParameterInput *pin);
  ~Mesh();

  // accessors
  int GetNumMeshBlocksThisRank(int my_rank = Globals::my_rank) const {
    return nblist[my_rank];
  }
  int GetNumMeshThreads() const { return num_mesh_threads_; }
  std::int64_t GetTotalCells();
  // TODO(JMM): Move block_size into mesh.
  int GetNumberOfMeshBlockCells() const;
  const RegionSize &GetDefaultBlockSize() const { return base_block_size; }
  RegionSize GetBlockSize(const LogicalLocation &loc, std::size_t coarsenings = 0) const {
    return forest.GetBlockDomain(loc, coarsenings);
  }
  const IndexShape GetLeafBlockCellBounds(CellLevel level = CellLevel::same) const;

  ParArray1D<AmrTag> &GetAmrTags();

  const forest::Forest &Forest() const { return forest; }

  // data
  bool modified;
  RegionSize mesh_size;
  RegionSize base_block_size;

  BValNames_t mesh_bc_names;
  BValNames_t mesh_swarm_bc_names;

  // these are flags not boundary functions
  std::array<BoundaryFlag, BOUNDARY_NFACES> mesh_bcs;
  int ndim; // number of dimensions
  const bool adaptive, multilevel, multigrid;
  int nbtotal, nbnew, nbdel;
  std::uint64_t mbcnt;

  int step_since_lb;
  int gflag;
  int task_collection_timeout_in_seconds;

  const bool do_coalesced_comms;
  // Don't complete a coalesced receive task until this rank's coalesced sends for the
  // same partition and boundary type have completed
  const bool coalesced_wait_for_sends;

  BlockList_t block_list;
  Packages_t packages;
  std::shared_ptr<StateDescriptor> resolved_packages;

  DataCollection<MeshData<Real>> mesh_data;

  int GetGMGMaxLevel() const { return current_level; }
  int GetGMGMinLevel() const { return gmg_min_level_; }
  GridIdentifier GetGMGGrid(int gmg_level) {
    if (gmg_grids_.count(gmg_level)) return gmg_grids_[gmg_level];
    return GridIdentifier::none();
  }

  // functions
  void Initialize(bool init_problem, ParameterInput *pin, ApplicationInput *app_in);

  bool SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                 BoundaryFlag *block_bcs,
                                 std::size_t block_coarsenings = 0);
  void OutputCycleDiagnostics();
  void LoadBalancingAndAdaptiveMeshRefinement(ParameterInput *pin,
                                              ApplicationInput *app_in);

  // Block permutation. Moves the content of every block (the remeshed variables, i.e.
  // MeshBlock::vars_cc_, with their sparse allocation state, and the block's
  // load-balance cost) from gid old to gid new_gid_of_old[old]. Block locations, the
  // forest and the gids' coordinates stay put, so this is meant for pools of blocks
  // whose position on the mesh carries no physics (and for meshes without swarms). The
  // ranks are recomputed for the permuted costs (and LoadBalanceGroups), content moves
  // with the remeshing send/receive, and the neighbors, boundary buffers and provider
  // lanes are rebuilt (the providers are called again, so the user's graph must already
  // be in the new gids) and ghosts filled, as after load balancing. Collective; every
  // rank passes the same permutation, computed from global data.
  void PermuteBlocks(const std::vector<int> &new_gid_of_old, ParameterInput *pin,
                     ApplicationInput *app_in);
  // Called by the driver after load balancing: if BlockPermutation is set and
  // parthenon/loadbalancing/permute_interval > 0 divides tm.ncycle, permutes the blocks
  // with the permutation it returns (if not empty). Returns whether blocks were moved.
  bool PermuteBlocksIfDue(ParameterInput *pin, ApplicationInput *app_in,
                          SimTime const &tm);
  int DefaultPackSize() {
    if (use_pack_size_) {
      return default_pack_size_ < 1 ? std::max(static_cast<int>(block_list.size()), 1)
                                    : default_pack_size_;
    } else {
      return std::max(
          1, partition::partition_impl::IntCeil(block_list.size(), default_num_packs_));
    }
  }
  int DefaultNumPartitions() {
    if (use_pack_size_) {
      return partition::partition_impl::IntCeil(block_list.size(), DefaultPackSize());
    } else {
      return std::max(1,
                      static_cast<int>(std::min(default_num_packs_, block_list.size())));
    }
  }
  std::size_t CommBufferChunkSize() {
    // Might be worth discussing what a good default is.  The number
    // of blocks on a rank is a "greatest common denominator" of the
    // number of buffers required, assuming each block has similar
    // buffer configurations, which may or may not be a good
    // approximation. To minimize the memory footprint at the cost of
    // more allocations, the user may set this to "1."
    return (nbuf_add_ > 0) ? nbuf_add_ : std::max(std::size_t{1}, block_list.size());
  }

  const std::vector<std::shared_ptr<BlockListPartition>> &
  GetDefaultBlockPartitions() const {
    auto grid = GridIdentifier::leaf();
    PARTHENON_REQUIRE(
        block_partitions_.count(grid),
        "There isn't a block partition available for this grid for some reason.");
    return block_partitions_.at(grid);
  }

  const std::vector<std::shared_ptr<BlockListPartition>> &
  GetMultigridBlockPartitions(int gmg_level) const {
    auto grid = gmg_grids_.at(gmg_level);
    PARTHENON_REQUIRE(multigrid, "Asking for a partition of a multigrid grid when "
                                 "parthenon/mesh/multigrid = false.")
    PARTHENON_REQUIRE(
        block_partitions_.count(grid),
        "There isn't a block partition available for this grid for some reason.");
    return block_partitions_.at(grid);
  }

  auto GetBasePartition() const { return base_block_partition_; }

  std::shared_ptr<MeshBlock> FindMeshBlock(int tgid) const;

  void ApplyUserWorkBeforeOutput(Mesh *mesh, ParameterInput *pin, SimTime const &time);

  void ApplyUserWorkBeforeRestartOutput(Mesh *mesh, ParameterInput *pin,
                                        SimTime const &time, OutputParameters *pparams);

  // defined in either the prob file or default_pgen.cpp in ../pgen/
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> ProblemGenerator =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> PostProblemGenerator =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, MeshData<Real> *)> PostInitialization =
      nullptr;
  static void UserWorkAfterLoopDefault(Mesh *mesh, ParameterInput *pin,
                                       SimTime &tm); // called in main loop
  std::function<void(Mesh *, ParameterInput *, SimTime &)> UserWorkAfterLoop = nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime &)> PreStepUserWorkInLoop =
      nullptr;
  std::function<void(Mesh *, ParameterInput *, SimTime const &)> PostStepUserWorkInLoop =
      nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      UserMeshWorkBeforeOutput = nullptr;

  std::function<void(Mesh *, ParameterInput *, SimTime const &,
                     OutputParameters *pparams)>
      UserWorkBeforeRestartOutput = nullptr;

  static void PreStepUserDiagnosticsInLoopDefault(Mesh *, ParameterInput *,
                                                  SimTime const &);
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PreStepUserDiagnosticsInLoop = PreStepUserDiagnosticsInLoopDefault;
  static void PostStepUserDiagnosticsInLoopDefault(Mesh *, ParameterInput *,
                                                   SimTime const &);
  std::function<void(Mesh *, ParameterInput *, SimTime const &)>
      PostStepUserDiagnosticsInLoop = PostStepUserDiagnosticsInLoopDefault;
  // See ApplicationInput::MeshBlockPermutation and ApplicationInput::LoadBalanceGroups
  std::function<std::vector<int>(Mesh *, ParameterInput *, SimTime const &)>
      BlockPermutation = nullptr;
  std::function<std::vector<int>(Mesh *, const std::vector<LogicalLocation> &)>
      LoadBalanceGroups = nullptr;

  int GetRootLevel() const noexcept { return root_level; }
  int GetLegacyTreeRootLevel() const {
    return forest.root_level + forest.forest_level.value();
  }

  int GetMaxLevel() const noexcept { return max_level; }
  int GetCurrentLevel() const noexcept { return current_level; }
  std::vector<int> GetNbList() const noexcept { return nblist; }
  std::vector<LogicalLocation> GetLocList() const noexcept { return loclist; }

  std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>>
  GetLevelsAndLogicalLocationsFlat() const noexcept;

  void OutputMeshStructure(const int dim, const bool dump_mesh_structure = true);

  // Ordering here is important to prevent deallocation of pools before boundary
  // communication buffers
  // channel_key_t is tuple of (gid_sender, gid_receiver, variable_name,
  // block_location_idx, extra_delineater) which uniquely define a communication channel
  // between two blocks for a given variable
  using channel_key_t = std::tuple<int, int, std::string, int, int>;
  using comm_buf_t = CommBuffer<buf_pool_t<Real>::owner_t>;
  class comm_buf_map_t {
   public:
    using map_t = std::unordered_map<channel_key_t, comm_buf_t>;
    using key_type = map_t::key_type;
    using mapped_type = map_t::mapped_type;

   private:
    // On initial meshing and after remeshing, the comm buffer map is cleared and
    // rebuilt. The member epoch_ stores the number of times the comm buffers have
    // been built so that various boundary cache objects that point to the comm
    // buffers can easily check if they point to buffers from an old mesh
    // configuration that have been cleared.
    std::size_t epoch_{1};
    map_t m_;

   public:
    auto &operator[](const key_type &k) { return m_[k]; }
    auto &at(const key_type &k) { return m_.at(k); }
    auto count(const key_type &k) const { return m_.count(k); }

    auto begin() noexcept { return m_.begin(); }
    auto end() noexcept { return m_.end(); }
    auto begin() const noexcept { return m_.begin(); }
    auto end() const noexcept { return m_.end(); }

    auto GetCurrentEpoch() const { return epoch_; }
    void clear() {
      m_.clear();
      epoch_++;
    }
    // Provider lanes have their own epoch, bumped by Mesh::RebuildLane; a lane's boundary
    // caches re-initialize when either epoch changes
    std::size_t GetLaneEpoch(int lane) const {
      return lane < static_cast<int>(lane_epochs_.size()) ? lane_epochs_[lane] : 0;
    }
    // Removes the buffers of one lane (those whose key satisfies pred) and bumps that
    // lane's epoch only; other buffers and the boundary caches pointing to them are kept
    template <class Pred>
    void EraseLane(int lane, Pred &&pred) {
      for (auto it = m_.begin(); it != m_.end();) {
        it = pred(it->first) ? m_.erase(it) : std::next(it);
      }
      BumpLaneEpoch(lane);
    }
    auto erase(const key_type &k) { return m_.erase(k); }
    void BumpLaneEpoch(int lane) {
      if (lane >= static_cast<int>(lane_epochs_.size())) lane_epochs_.resize(lane + 1, 0);
      lane_epochs_[lane]++;
    }

   private:
    std::vector<std::size_t> lane_epochs_;
  };

  ObjectPoolMap<BufArray1D<Real>> pool_map;
  comm_buf_map_t boundary_comm_map;
  TagMap tag_map;
  int minimum_number_of_teams_for_boundary_kernel;
  int boundary_buffer_work_chunk_size;

  // Sets the number of comm lanes for a given boundary type. Each comm lane is an
  // independent set of communication buffers, so different MeshData (e.g. in different
  // stages) can communicate concurrently by using different lanes (see
  // MeshData::SetCommLane). This *must* be called before build boundary buffers
  // is called internally, so use beyond the defaults with care
  void SetNumberOfCommLanes(BoundaryType bound, std::size_t n_lanes) {
    if (locked_comm_lane_numbers_.count(bound))
      PARTHENON_FAIL("Trying to reset the number of comm lanes after boundary buffers "
                     "have been set up.");
    if (number_of_comm_lanes_.count(bound) && number_of_comm_lanes_[bound] > n_lanes)
      PARTHENON_WARN(
          "You are reducing the number of comm lanes from a previously set value.");
    number_of_comm_lanes_[bound] = n_lanes;

    // Need to set the complementary lanes to the same value
    if (!IsSender(bound)) number_of_comm_lanes_[GetAssociatedSender(bound)] = n_lanes;
    if (!IsReceiver(bound)) number_of_comm_lanes_[GetAssociatedReceiver(bound)] = n_lanes;
  }

  void LockCommLaneNumbers(BoundaryType bound) {
    locked_comm_lane_numbers_.insert(GetAssociatedSender(bound));
    locked_comm_lane_numbers_.insert(GetAssociatedReceiver(bound));
  }

  std::size_t GetNumberOfCommLanes(BoundaryType bound) const {
    if (number_of_comm_lanes_.count(bound)) return number_of_comm_lanes_.at(bound);
    // We default to only having a single comm lane
    return 1;
  }

  // Pre-PR 1473 names, kept for downstream codes
  [[deprecated("Use SetNumberOfCommLanes")]] void
  SetNumberOfCommChannels(BoundaryType bound, std::size_t n_channels) {
    SetNumberOfCommLanes(bound, n_channels);
  }
  [[deprecated("Use GetNumberOfCommLanes")]] std::size_t
  GetNumberOfCommChannels(BoundaryType bound) const {
    return GetNumberOfCommLanes(bound);
  }

  template <BoundaryType bound_type>
  void AddToTagMap(std::shared_ptr<MeshData<Real>> &md) {
    LockCommLaneNumbers(bound_type);
    int lanes = GetNumberOfCommLanes(bound_type);
    tag_map.AddMeshDataToMap<bound_type>(md, lanes);
  }

  std::shared_ptr<CoalescedComms> pcoalesced_comms;

  // Provider lanes. A lane is an exchange of the Metadata::Lane variables of one LaneId
  // (Metadata::SetLane) between the neighbors that a user provider lists for each local
  // block, instead of the geometric neighbors. Parthenon fills in the neighbors' ranks,
  // builds the lane's tag map and buffers and runs the exchange
  // (AddLaneExchangeTasks, AddLaneFluxCorrectionTasks). Nothing is built for a lane
  // without a provider. Coalesced comms are not supported on lanes.
  //
  // Registers (or replaces) the provider of a lane. On every rank, collectively in the
  // sense that both ends of each channel must register. If the boundary buffers already
  // exist the lane is built immediately (RebuildLane); afterwards it is rebuilt with the
  // other buffers whenever the mesh is (initialization, load balancing, remeshing).
  void RegisterNeighborProvider(LaneId lane, NeighborProvider_t provider);
  bool HasNeighborProvider(LaneId lane) const {
    return lane_providers_.count(static_cast<int>(lane)) > 0;
  }
  // Rank-local rebuild of one lane after its provider's graph or its buffer sizes
  // changed: reruns the provider for every local block and the lane's BufferSize ops,
  // and rebuilds only what changed: the neighbor lists of the blocks whose list changed,
  // the tag maps of the ranks whose channels changed, and the buffers of new channels,
  // of channels whose size changed and of every channel to a rank whose tags changed.
  // If anything was rebuilt it bumps the lane's comm-map epoch so that the lane's
  // sub-caches in every MeshData re-initialize at their next exchange. Other lanes and
  // the geometric exchanges keep their buffers and caches. With full, the lane's neighbor
  // lists, tag map and buffers are all rebuilt, as on remeshing. Every rank on either end
  // of a changed channel must call it before its next exchange on the lane. The lane
  // must be quiescent: every message of its previous exchanges completed and no receive
  // posted (i.e. call it between exchanges, not while lane tasks are in flight).
  void RebuildLane(LaneId lane, bool full = false);
  TagMap &GetLaneTagMap(int lane) { return lane_tag_maps_.at(lane); }

  bool TryReallocCommBufferPools();

#ifdef MPI_PARALLEL
  MPI_Comm GetMPIComm(const std::string &label) const { return mpi_comm_map_.at(label); }
#endif

  void SetAllVariablesToInitialized() {
    for (auto &sp_mb : block_list) {
      for (auto &pair : sp_mb->meshblock_data.Stages()) {
        auto &sp_mbd = pair.second;
        sp_mbd->SetAllVariablesToInitialized();
      }
    }
  }

  uint64_t GetBufferPoolSizeInBytes() const {
    std::uint64_t buffer_memory = 0;
    for (auto &p : pool_map.GetMap()) {
      buffer_memory += p.second.SizeInBytes();
    }
    return buffer_memory;
  }

  // expose a mesh-level call to get lists of variables from resolved_packages
  template <typename... Args>
  std::vector<std::string> GetVariableNames(Args &&...args) {
    return resolved_packages->GetVariableNames(std::forward<Args>(args)...);
  }

  forest::Forest forest;

 private:
  // data
  int root_level, max_level, current_level;
  int max_level_ref_; // the max level as interpreted by the input deck/user
  int num_mesh_threads_;
  int base_block_coarsenings;

  /// Maps Global Block IDs to which rank the block is mapped to.
  std::vector<int> ranklist;
  /// Maps rank to start of local block IDs.
  std::vector<int> nslist;
  /// Maps rank to count of local blocks.
  std::vector<int> nblist;
  /// Maps global block ID to its cost
  std::vector<double> costlist;
  // 8x arrays used exclusively for AMR (not SMR):
  /// Count of blocks to refine on each rank
  std::vector<int> nref;
  /// Count of blocks to de-refine on each rank
  std::vector<int> nderef;
  std::vector<int> rdisp, ddisp;
  std::vector<int> bnref, bnderef;
  std::vector<int> brdisp, bddisp;
  // the last 4x should be std::size_t, but are limited to int by MPI
  // Refinement tags used by MeshData checks
  ParArray1D<AmrTag> amr_tags;
  std::map<BoundaryType, std::size_t> number_of_comm_lanes_;
  std::set<BoundaryType> locked_comm_lane_numbers_;

  std::vector<LogicalLocation> loclist;

  // Block lists for internal nodes in the tree corresponding to multigrid levels
  std::map<int, GridIdentifier> gmg_grids_;
  std::map<int, BlockList_t> gmg_block_lists_; // maps from *GMG* level to blocks list

  // flags are false if using non-uniform or user meshgen function
  bool use_uniform_meshgen_fn_[4];

  // variables for load balancing control
  bool lb_flag_, lb_automatic_, lb_manual_;
  double lb_tolerance_;
  int lb_interval_;
  int permute_interval_ = 0;

  // size of default MeshBlockPacks
  bool use_pack_size_;
  int default_pack_size_;
  std::size_t default_num_packs_;

  // number of comm buffers to add when more need to be allocated
  // TODO(JMM): Stash this in globals or a param maybe?
  std::int64_t nbuf_add_;

  // Tracking for when to re-allocate comm-buffers to minimize memory
  // footprint.
  Real buffer_reset_frac_;

  int gmg_min_level_ = 0;

#ifdef MPI_PARALLEL
  // Global map of MPI comms for separate variables
  std::unordered_map<std::string, MPI_Comm> mpi_comm_map_;
#endif

  void SetBCNames_(ParameterInput *pin);
  std::array<BoundaryFlag, BOUNDARY_NFACES>
  GetBCsFromNames_(const BValNames_t &names) const;

  // functions
  void CheckMeshValidity() const;
  void BuildBlockList(ParameterInput *pin, ApplicationInput *app_in, Packages_t &packages,
                      int mesh_test,
                      const std::unordered_map<LogicalLocation, int> &dealloc_count = {});
  void DoStaticRefinement(ParameterInput *pin);
  void CalculateLoadBalance(std::vector<double> const &costlist,
                            const std::vector<LogicalLocation> &locs,
                            std::vector<int> &ranklist, std::vector<int> &nslist,
                            std::vector<int> &nblist);
  void ResetLoadBalanceVariables();

  // Mesh::LoadBalancingAndAdaptiveMeshRefinement() helper functions:
  void UpdateCostList();
  void UpdateMeshBlockTree(int &nnew, int &ndel);
  bool GatherCostListAndCheckBalance();
  void RedistributeAndRefineMeshBlocks(ParameterInput *pin, ApplicationInput *app_in,
                                       int ntot);
  void BuildGMGBlockLists(ParameterInput *pin, ApplicationInput *app_in);
  void SetGMGNeighbors();

  // Optionally defined in the problem file
  std::function<void(Mesh *, ParameterInput *)> InitUserMeshData = nullptr;

  // Re-used functionality in constructor
  void RegisterLoadBalancing_(ParameterInput *pin);
  void BuildAndRegisterCommBuffers_();

  void SetupMPIComms();
  void BuildTagMapAndBoundaryBuffers();
  // Provider lanes (see RegisterNeighborProvider)
  void SetLaneNeighbors_(int lane);
  void BuildLaneTagMap_(int lane);
  // The provider's sorted, checked neighbor list of one block, and one entry as a
  // NeighborBlock
  void ListLaneNeighbors_(const MeshBlock &pmb, int lane, std::vector<LaneNeighbor> &lnbs);
  NeighborBlock MakeLaneNeighbor_(const MeshBlock &pmb, const LaneNeighbor &l);
  // BufferSize of every (lane neighbor, lane variable) of a block, and of all local
  // blocks into MeshBlock::lane_buffer_sizes (what the lane's buffers were built with)
  void LaneBufferSizes_(MeshBlock *pmb, int lane, std::vector<int> &sizes);
  void RecordLaneBufferSizes_(int lane);
  int LaneCommLanes_();
  std::map<int, NeighborProvider_t> lane_providers_;
  std::map<int, TagMap> lane_tag_maps_;
  void CommunicateBoundaries(std::string md_name = "base",
                             const std::vector<std::string> &fields = {});
  void PreCommFillDerived();
  void FillDerived();

  void BuildBlockPartitions(GridIdentifier grid);
  std::map<GridIdentifier, std::vector<std::shared_ptr<BlockListPartition>>>
      block_partitions_;
  std::shared_ptr<BlockListPartition> base_block_partition_;
};

} // namespace parthenon

#endif // MESH_MESH_HPP_

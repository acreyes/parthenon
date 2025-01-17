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
#ifndef INTERFACE_MESH_DATA_HPP_
#define INTERFACE_MESH_DATA_HPP_

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bvals/comms/bnd_info.hpp"
#include "interface/sparse_pack_base.hpp"
#include "interface/swarm_pack_base.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "utils/communication_buffer.hpp"
#include "utils/error_checking.hpp"
#include "utils/object_pool.hpp"
#include "utils/unique_id.hpp"
#include "utils/utils.hpp"

namespace parthenon {

class Mesh;
template <typename T>
class MeshBlockData;

template <typename T>
using BlockDataList_t = std::vector<std::shared_ptr<MeshBlockData<T>>>;

namespace pack_on_mesh_impl {

// This function template takes a new key and adds it to a key collection
template <typename K>
inline void AppendKey(K *key_collection, const K *new_key);

// Specialization for variable packs where key is a vpack_types::VPackKey_t
template <>
inline void AppendKey<vpack_types::VPackKey_t>(vpack_types::VPackKey_t *key_collection,
                                               const vpack_types::VPackKey_t *new_key) {
  for (const auto &k : *new_key) {
    key_collection->push_back(k);
  }
}

// Specialization for flux-variable packs where key is a vpack_types::UidVecPair
template <>
inline void AppendKey<vpack_types::UidVecPair>(vpack_types::UidVecPair *key_collection,
                                               const vpack_types::UidVecPair *new_key) {
  for (const auto &k : new_key->first) {
    key_collection->first.push_back(k);
  }
  for (const auto &k : new_key->second) {
    key_collection->second.push_back(k);
  }
}

// This functor template takes a pack (VariablePack or VariableFluxPack) and appends
// all the allocation statuses to the given collection of allocation statuses. We have to
// use a functor instead of a template function because template function cannot be
// partially specialized
template <typename P>
struct AllocationStatusCollector {
  static inline void Append(std::vector<int> *alloc_status_collection, const P &pack);
};

// Specialization for VariablePack<T>
template <typename T>
struct AllocationStatusCollector<VariablePack<T>> {
  static inline void Append(std::vector<int> *alloc_status_collection,
                            const VariablePack<T> &var_pack) {
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_pack.alloc_status()->begin(),
                                    var_pack.alloc_status()->end());
  }
};

// Specialization for VariableFluxPack<T>
template <typename T>
struct AllocationStatusCollector<VariableFluxPack<T>> {
  static inline void Append(std::vector<int> *alloc_status_collection,
                            const VariableFluxPack<T> &var_flux_pack) {
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_flux_pack.alloc_status()->cbegin(),
                                    var_flux_pack.alloc_status()->cend());
    alloc_status_collection->insert(alloc_status_collection->end(),
                                    var_flux_pack.flux_alloc_status()->cbegin(),
                                    var_flux_pack.flux_alloc_status()->cend());
  }
};

// TODO(JMM): pass the coarse/fine option through the meshblockpack machinery
template <typename P, typename K, typename M, typename F>
const MeshBlockPack<P> &PackOnMesh(M &map, BlockDataList_t<Real> &block_data_,
                                   F &packing_function, PackIndexMap *map_out) {
  const auto nblocks = block_data_.size();

  // since the pack keys used by MeshBlockData includes the allocation status of each
  // variable, we cannot simply use the key from the first MeshBlockData, but we need to
  // get the keys from all MeshBlockData instances and concatenate them
  K total_key;
  K this_key;

  PackIndexMap pack_idx_map;
  PackIndexMap this_map;

  std::vector<int> alloc_status_collection;

  for (size_t i = 0; i < nblocks; i++) {
    const auto &pack = packing_function(block_data_[i], this_map, this_key);
    AppendKey(&total_key, &this_key);
    AllocationStatusCollector<P>::Append(&alloc_status_collection, pack);

    if (i == 0) {
      pack_idx_map = this_map;
    } else {
      assert(this_map == pack_idx_map);
    }
  }

  auto itr = map.find(total_key);
  bool make_new_pack = false;
  if (itr == map.end()) {
    // we don't have a cached pack, need to make a new one
    make_new_pack = true;
  } else {
    // we have a cached pack, check allocation status
    if (alloc_status_collection != itr->second.alloc_status) {
      // allocation statuses differ, need to make a new pack and remove outdated one
      make_new_pack = true;
      map.erase(itr);
    }
  }

  if (make_new_pack) {
    ParArray1DRaw<P> packs(ViewOfViewAlloc("MeshData::PackVariables::packs"), nblocks);
    auto packs_host = create_view_of_view_mirror(packs);

    for (size_t i = 0; i < nblocks; i++) {
      const auto &pack = packing_function(block_data_[i], this_map, this_key);
      packs_host(i) = pack;
    }

    std::array<int, 5> dims;
    for (int i = 0; i < 4; i++) {
      dims[i] = packs_host(0).GetDim(i + 1);
    }
    dims[4] = nblocks;

    Kokkos::deep_copy(packs, packs_host);

    typename M::mapped_type new_item;
    new_item.alloc_status = alloc_status_collection;
    new_item.map = pack_idx_map;
    new_item.pack = MeshBlockPack<P>(packs, dims);

    itr = map.insert({total_key, new_item}).first;
  }

  if (map_out != nullptr) {
    *map_out = itr->second.map;
  }

  return itr->second.pack;
}

} // namespace pack_on_mesh_impl

/// The MeshData class is a container for cached MeshBlockPacks, i.e., it
/// contains both the pointers to the MeshBlockData of the MeshBlocks contained
/// in the object as well as maps to the cached MeshBlockPacks of VariablePacks or
/// VariableFluxPacks.

template <typename T>
class MeshData {
 public:
  using parent_t = Mesh;
  MeshData() = default;
  explicit MeshData(const std::string &name) : stage_name_(name) {}

  GridIdentifier grid;
  int partition;

  const auto &StageName() const { return stage_name_; }

  Mesh *GetMeshPointer() const { return pmy_mesh_; }
  auto GetParentPointer() const { return GetMeshPointer(); }

  void SetMeshPointer(Mesh *pmesh) { pmy_mesh_ = pmesh; }
  void SetMeshPointer(const std::shared_ptr<MeshData<T>> &other) {
    pmy_mesh_ = other->GetMeshPointer();
  }

  void SetAllowedDt(const Real dt) const {
    for (const auto &pbd : block_data_) {
      pbd->SetAllowedDt(std::min(dt, pbd->GetBlockPointer()->NewDt()));
    }
  }

  auto &GetBvarsCache() { return bvars_cache_; }

  template <class... Ts>
  IndexRange GetBoundsI(Ts &&...args) const {
    if (block_data_.size() > 0)
      return block_data_[0]->GetBoundsI(std::forward<Ts>(args)...);
    return IndexRange{-1, -2};
  }

  template <class... Ts>
  IndexRange GetBoundsJ(Ts &&...args) const {
    if (block_data_.size() > 0)
      return block_data_[0]->GetBoundsJ(std::forward<Ts>(args)...);
    return IndexRange{-1, -2};
  }

  template <class... Ts>
  IndexRange GetBoundsK(Ts &&...args) const {
    if (block_data_.size() > 0)
      return block_data_[0]->GetBoundsK(std::forward<Ts>(args)...);
    return IndexRange{-1, -2};
  }

  template <class... Args>
  void Add(Args &&...args) {
    for (const auto &pbd : block_data_) {
      pbd->Add(std::forward<Args>(args)...);
    }
  }

  template <typename ID_t>
  void Initialize(const std::shared_ptr<BlockListPartition> &part,
                  const std::vector<ID_t> &vars, const bool shallow) {
    PARTHENON_REQUIRE(
        shallow == false,
        "Can't shallow copy when the source is not another MeshData object.");
    SetMeshProperties(part->pmesh);
    auto &bl = part->block_list;
    block_data_.resize(bl.size());
    for (int i = 0; i < bl.size(); ++i)
      block_data_[i] = bl[i]->meshblock_data.Add(stage_name_, bl[i], vars);
    grid = part->grid;
    partition = part->partition;
  }

  template <typename ID_t>
  void Initialize(std::shared_ptr<MeshData<T>> src, const std::vector<ID_t> &vars,
                  const bool shallow) {
    if (src == nullptr) {
      PARTHENON_THROW("src points at null");
    }
    SetMeshProperties(src->GetParentPointer());
    const int nblocks = src->NumBlocks();
    block_data_.resize(nblocks);
    for (int i = 0; i < nblocks; ++i) {
      auto pmbd = src->GetBlockData(i);
      block_data_[i] = pmbd->GetBlockSharedPointer()->meshblock_data.Add(
          stage_name_, pmbd, vars, shallow);
    }
    grid = src->grid;
    partition = src->partition;
  }

  void Initialize(BlockList_t blocks, Mesh *pmesh, std::optional<int> gmg_level = {});

  const std::shared_ptr<MeshBlockData<T>> &GetBlockData(int n) const {
    assert(n >= 0 && n < block_data_.size());
    return block_data_[n];
  }

  std::shared_ptr<MeshBlockData<T>> &GetBlockData(int n) {
    assert(n >= 0 && n < block_data_.size());
    return block_data_[n];
  }

  const auto &GetAllBlockData() const { return block_data_; }

  bool ContainsGid(int gid) const {
    bool contains = false;
    for (auto &b : block_data_) {
      if (b->GetBlockPointer()->gid == gid) contains = true;
    }
    return contains;
  }

  void SetAllVariablesToInitialized() {
    std::for_each(block_data_.begin(), block_data_.end(),
                  [](auto &sp_block) { sp_block->SetAllVariablesToInitialized(); });
  }

  bool AllVariablesInitialized() {
    bool all_initialized = true;
    std::for_each(block_data_.begin(), block_data_.end(), [&](auto &sp_block) {
      all_initialized = all_initialized && sp_block->AllVariablesInitialized();
    });
    return all_initialized;
  }

  std::vector<bool> AllocationStatus(const std::string &label) {
    std::vector<bool> status(NumBlocks());
    std::transform(
        block_data_.begin(), block_data_.end(), status.begin(),
        [&](std::shared_ptr<MeshBlockData<T>> mbd) { return mbd->IsAllocated(label); });
    return status;
  }

 private:
  template <typename... Args>
  const auto &PackVariablesAndFluxesImpl(PackIndexMap *map_out, Args &&...args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data,
                             PackIndexMap &map, vpack_types::UidVecPair &key) {
      return meshblock_data->PackVariablesAndFluxes(std::forward<Args>(args)..., map,
                                                    key);
    };

    return pack_on_mesh_impl::PackOnMesh<VariableFluxPack<T>, vpack_types::UidVecPair>(
        varFluxPackMap_, block_data_, pack_function, map_out);
  }

  template <typename... Args>
  const auto &PackVariablesImpl(PackIndexMap *map_out, bool coarse, Args &&...args) {
    auto pack_function = [&](std::shared_ptr<MeshBlockData<T>> meshblock_data,
                             PackIndexMap &map, vpack_types::VPackKey_t &key) {
      return meshblock_data->PackVariables(std::forward<Args>(args)..., map, key, coarse);
    };
    return pack_on_mesh_impl::PackOnMesh<VariablePack<T>, vpack_types::VPackKey_t>(
        varPackMap_, block_data_, pack_function, map_out);
  }

 public:
  // DO NOT use variatic templates here. They shadow each other

  // Pack by separate variable and flux names
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map, var_names, flx_names, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(nullptr, var_names, flx_names, sparse_ids);
  }
  // no sparse ids
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map, var_names, flx_names);
  }
  const auto &PackVariablesAndFluxes(const std::vector<std::string> &var_names,
                                     const std::vector<std::string> &flx_names) {
    return PackVariablesAndFluxesImpl(nullptr, var_names, flx_names);
  }
  // Pack by either the same variable and flux names, or by metadata flags
  template <typename Selector>
  const auto &PackVariablesAndFluxes(const Selector &names_or_flags,
                                     const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map, names_or_flags, sparse_ids);
  }
  template <typename Selector>
  const auto &PackVariablesAndFluxes(const Selector &names_or_flags,
                                     const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(nullptr, names_or_flags, sparse_ids);
  }
  // no sparse ids
  template <typename Selector>
  const auto &PackVariablesAndFluxes(const Selector &names_or_flags, PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map, names_or_flags);
  }
  template <typename Selector>
  const auto &PackVariablesAndFluxes(const Selector &names_or_flags) {
    return PackVariablesAndFluxesImpl(nullptr, names_or_flags);
  }
  // only sparse ids
  const auto &PackVariablesAndFluxes(const std::vector<int> &sparse_ids,
                                     PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map, sparse_ids);
  }
  const auto &PackVariablesAndFluxes(const std::vector<int> &sparse_ids) {
    return PackVariablesAndFluxesImpl(nullptr, sparse_ids);
  }
  // No nothing
  const auto &PackVariablesAndFluxes(PackIndexMap &map) {
    return PackVariablesAndFluxesImpl(&map);
  }
  const auto &PackVariablesAndFluxes() { return PackVariablesAndFluxesImpl(nullptr); }

  // As above, DO NOT use variadic templates here. They shadow each other.
  // covers names and metadata flags
  template <typename Selector>
  const auto &PackVariables(const Selector &names_or_flags,
                            const std::vector<int> &sparse_ids, PackIndexMap &map,
                            bool coarse = false) {
    return PackVariablesImpl(&map, coarse, names_or_flags, sparse_ids);
  }
  template <typename Selector>
  const auto &PackVariables(const Selector &names_or_flags,
                            const std::vector<int> &sparse_ids, bool coarse = false) {
    return PackVariablesImpl(nullptr, coarse, names_or_flags, sparse_ids);
  }
  // no sparse ids
  template <typename Selector>
  const auto &PackVariables(const Selector &names_or_flags, PackIndexMap &map,
                            bool coarse = false) {
    return PackVariablesImpl(&map, coarse, names_or_flags);
  }
  template <typename Selector>
  const auto &PackVariables(const Selector &names_or_flags, bool coarse = false) {
    return PackVariablesImpl(nullptr, coarse, names_or_flags);
  }
  // No names or flags
  const auto &PackVariables(const std::vector<int> &sparse_ids, PackIndexMap &map,
                            bool coarse = false) {
    return PackVariablesImpl(&map, coarse, sparse_ids);
  }
  const auto &PackVariables(const std::vector<int> &sparse_ids, bool coarse = false) {
    return PackVariablesImpl(nullptr, coarse, sparse_ids);
  }
  // no nothing
  const auto &PackVariables(PackIndexMap &map, bool coarse = false) {
    return PackVariablesImpl(&map, coarse);
  }
  const auto &PackVariables(bool coarse = false) {
    return PackVariablesImpl(nullptr, coarse);
  }

  void ClearCaches() {
    sparse_pack_cache_.clear();
    block_data_.clear();
    varPackMap_.clear();
    varFluxPackMap_.clear();
    bvars_cache_.clear();
  }

  int GetNDim() const { return ndim_; }
  int NumBlocks() const { return block_data_.size(); }

  bool operator==(MeshData<T> &cmp) const {
    const int nblocks = block_data_.size();
    const int nblocks_cmp = cmp.NumBlocks();
    if (nblocks != nblocks_cmp) return false;

    for (int i = 0; i < nblocks; i++) {
      if (!(*block_data_[i] == *(cmp.GetBlockData(i)))) return false;
    }
    return true;
  }

  // vars may be a subset of the MeshData object
  template <typename Vars_t>
  bool Contains(const Vars_t &vars) const noexcept {
    return std::all_of(block_data_.begin(), block_data_.end(),
                       [this, vars](const auto &b) { return b->Contains(vars); });
  }
  // MeshData object must contain these vars and only these vars
  template <typename Vars_t>
  bool ContainsExactly(const Vars_t &vars) const noexcept {
    return std::all_of(block_data_.begin(), block_data_.end(),
                       [this, vars](const auto &b) { return b->ContainsExactly(vars); });
  }

  // Checks that the same set of variables was requested to create this container
  // (which may be different than the set of variables in the container because of fluxes)
  template <typename Vars_t>
  bool CreatedFrom(const Vars_t &vars) const noexcept {
    return std::all_of(block_data_.begin(), block_data_.end(),
                       [this, vars](const auto &b) { return b->CreatedFrom(vars); });
  }

  std::shared_ptr<SwarmContainer> GetSwarmData(int n) {
    PARTHENON_REQUIRE(n >= 0 && n < block_data_.size(),
                      "MeshData::GetSwarmData requires n within [0, block_data_.size()]");
    return block_data_[n]->GetSwarmData();
  }

  SparsePackCache &GetSparsePackCache() { return sparse_pack_cache_; }

  template <typename TYPE>
  SwarmPackCache<TYPE> &GetSwarmPackCache() {
    if constexpr (std::is_same<TYPE, int>::value) {
      return swarm_pack_int_cache_;
    } else if constexpr (std::is_same<TYPE, Real>::value) {
      return swarm_pack_real_cache_;
    }
    PARTHENON_THROW("SwarmPacks only compatible with int and Real types");
  }

  void ClearSwarmCaches() {
    if (swarm_pack_real_cache_.size() > 0) swarm_pack_real_cache_.clear();
    if (swarm_pack_int_cache_.size() > 0) swarm_pack_int_cache_.clear();
  }

 private:
  void SetMeshProperties(Mesh *pmesh);

  int ndim_;
  Mesh *pmy_mesh_;
  BlockDataList_t<T> block_data_;
  std::string stage_name_;

  // caches for packs
  MapToMeshBlockVarPack<T> varPackMap_;
  MapToMeshBlockVarFluxPack<T> varFluxPackMap_;
  SparsePackCache sparse_pack_cache_;
  SwarmPackCache<int> swarm_pack_int_cache_;
  SwarmPackCache<Real> swarm_pack_real_cache_;
  // caches for boundary information
  BvarsCache_t bvars_cache_;
};

template <typename T, typename... Args>
std::vector<Uid_t> UidIntersection(MeshData<T> *md1, MeshData<T> *md2, Args &&...args) {
  if (md1->NumBlocks() == 0 || md2->NumBlocks() == 0) return std::vector<Uid_t>();
  return UidIntersection(md1->GetBlockData(0).get(), md2->GetBlockData(0).get(),
                         std::forward<Args>(args)...);
}

} // namespace parthenon

#endif // INTERFACE_MESH_DATA_HPP_

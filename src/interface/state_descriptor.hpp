//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
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
#ifndef INTERFACE_STATE_DESCRIPTOR_HPP_
#define INTERFACE_STATE_DESCRIPTOR_HPP_

#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "interface/metadata.hpp"
#include "interface/params.hpp"
#include "interface/swarm.hpp"
#include "refinement/amr_criteria.hpp"

namespace parthenon {

// Forward declarations
template <typename T>
class MeshBlockData;
template <typename T>
class MeshData;

/// The state metadata descriptor class.
///
/// Each State descriptor has a label, associated parameters, and
/// metadata for all fields within that state.
class StateDescriptor {
 public:
  // copy constructor throw error
  StateDescriptor(const StateDescriptor &s) = delete;

  // Preferred constructor
  explicit StateDescriptor(std::string label) : label_(label) {
    PostFillDerivedBlock = nullptr;
    PostFillDerivedMesh = nullptr;
    PreFillDerivedBlock = nullptr;
    PreFillDerivedMesh = nullptr;
    FillDerivedBlock = nullptr;
    FillDerivedMesh = nullptr;
    EstimateTimestepBlock = nullptr;
    EstimateTimestepMesh = nullptr;
    CheckRefinementBlock = nullptr;
  }

  template <typename T>
  void AddParam(const std::string &key, T value) {
    params_.Add<T>(key, value);
  }

  template <typename T>
  const T &Param(const std::string &key) {
    return params_.Get<T>(key);
  }

  // Set (if not set) and get simultaneously.
  // infers type correctly.
  template <typename T>
  const T &Param(const std::string &key, T value) {
    params_.Get(key, value);
  }

  Params &AllParams() { return params_; }
  // retrieve label
  const std::string &label() const { return label_; }

  bool AddSwarm(const std::string &swarm_name, const Metadata &m) {
    if (swarmMetadataMap_.count(swarm_name) > 0) {
      throw std::invalid_argument("Swarm " + swarm_name + " already exists!");
    }
    swarmMetadataMap_[swarm_name] = m;

    return true;
  }

  bool AddSwarmValue(const std::string &value_name, const std::string &swarm_name,
                     const Metadata &m);

  // field addition / retrieval routines
  // add a field with associated metadata
  bool AddField(const std::string &field_name, const Metadata &m);

  // retrieve number of fields
  int size() const { return metadataMap_.size(); }

  // Ensure all required bits are present
  // projective and can be called multiple times with no harm
  void ValidateMetadata();

  // retrieve all field names
  std::vector<std::string> Fields() {
    std::vector<std::string> names;
    names.reserve(metadataMap_.size());
    for (auto &x : metadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  // retrieve all swarm names
  std::vector<std::string> Swarms() {
    std::vector<std::string> names;
    names.reserve(swarmMetadataMap_.size());
    for (auto &x : swarmMetadataMap_) {
      names.push_back(x.first);
    }
    return names;
  }

  const std::map<std::string, Metadata> &AllFields() const { return metadataMap_; }
  const std::map<std::string, std::vector<Metadata>> &AllSparseFields() const {
    return sparseMetadataMap_;
  }
  const std::map<std::string, Metadata> &AllSwarms() const { return swarmMetadataMap_; }
  const std::map<std::string, Metadata> &
  AllSwarmValues(const std::string swarm_name) const {
    return swarmValueMetadataMap_.at(swarm_name);
  }
  bool SwarmPresent(const std::string swarm_name) const {
    return swarmMetadataMap_.count(swarm_name) > 0;
  }

  // retrieve metadata for a specific field
  Metadata &FieldMetadata(const std::string &field_name) {
    return metadataMap_[field_name];
  }

  // retrieve metadata for a specific swarm
  Metadata &SwarmMetadata(const std::string &swarm_name) {
    return swarmMetadataMap_[swarm_name];
  }

  template <typename F>
  void MetadataLoop(F func) {
    for (auto &pair : metadataMap_) {
      func(pair.second);
    }
    for (auto &pair : sparseMetadataMap_) {
      for (auto &metadata : pair.second) {
        func(metadata);
      }
    }
    for (auto &pair : swarmMetadataMap_) {
      func(pair.second);
    }
  }

  // get all metadata for this physics
  const std::map<std::string, Metadata> &AllMetadata() { return metadataMap_; }

  bool FlagsPresent(std::vector<MetadataFlag> const &flags, bool matchAny = false);

  void PreFillDerived(std::shared_ptr<MeshBlockData<Real>> &rc) const {
    if (PreFillDerivedBlock != nullptr) PreFillDerivedBlock(rc);
  }
  void PreFillDerived(std::shared_ptr<MeshData<Real>> &rc) const {
    if (PreFillDerivedMesh != nullptr) PreFillDerivedMesh(rc);
  }
  void PostFillDerived(std::shared_ptr<MeshBlockData<Real>> &rc) const {
    if (PostFillDerivedBlock != nullptr) PostFillDerivedBlock(rc);
  }
  void PostFillDerived(std::shared_ptr<MeshData<Real>> &rc) const {
    if (PostFillDerivedMesh != nullptr) PostFillDerivedMesh(rc);
  }
  void FillDerived(std::shared_ptr<MeshBlockData<Real>> &rc) const {
    if (FillDerivedBlock != nullptr) FillDerivedBlock(rc);
  }
  void FillDerived(std::shared_ptr<MeshData<Real>> &rc) const {
    if (FillDerivedMesh != nullptr) FillDerivedMesh(rc);
  }

  Real EstimateTimestep(std::shared_ptr<MeshBlockData<Real>> &rc) const {
    if (EstimateTimestepBlock != nullptr) return EstimateTimestepBlock(rc);
    return std::numeric_limits<Real>::max();
  }
  Real EstimateTimestep(std::shared_ptr<MeshData<Real>> &rc) const {
    if (EstimateTimestepMesh != nullptr) return EstimateTimestepMesh(rc);
    return std::numeric_limits<Real>::max();
  }

  AmrTag CheckRefinement(std::shared_ptr<MeshBlockData<Real>> &rc) const {
    if (CheckRefinementBlock != nullptr) return CheckRefinementBlock(rc);
    return AmrTag::derefine;
  }

  std::vector<std::shared_ptr<AMRCriteria>> amr_criteria;
  void (*PreFillDerivedBlock)(std::shared_ptr<MeshBlockData<Real>> &rc);
  void (*PreFillDerivedMesh)(std::shared_ptr<MeshData<Real>> &rc);
  void (*PostFillDerivedBlock)(std::shared_ptr<MeshBlockData<Real>> &rc);
  void (*PostFillDerivedMesh)(std::shared_ptr<MeshData<Real>> &rc);
  void (*FillDerivedBlock)(std::shared_ptr<MeshBlockData<Real>> &rc);
  void (*FillDerivedMesh)(std::shared_ptr<MeshData<Real>> &rc);
  Real (*EstimateTimestepBlock)(std::shared_ptr<MeshBlockData<Real>> &rc);
  Real (*EstimateTimestepMesh)(std::shared_ptr<MeshData<Real>> &rc);
  AmrTag (*CheckRefinementBlock)(std::shared_ptr<MeshBlockData<Real>> &rc);

  friend std::ostream &operator<<(std::ostream &os, const StateDescriptor &sd);

 private:
  Params params_;
  const std::string label_;
  std::map<std::string, Metadata> metadataMap_;
  std::map<std::string, std::vector<Metadata>> sparseMetadataMap_;
  std::map<std::string, Metadata> swarmMetadataMap_;
  std::map<std::string, std::map<std::string, Metadata>> swarmValueMetadataMap_;
};

using Packages_t = std::map<std::string, std::shared_ptr<StateDescriptor>>;

std::shared_ptr<StateDescriptor> ResolvePackages(Packages_t &packages);

} // namespace parthenon

#endif // INTERFACE_STATE_DESCRIPTOR_HPP_

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

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/variable.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_nanobind.hpp"

namespace parthenon {
namespace python {

namespace {
using NumpyArray1D = nb::ndarray<nb::numpy, Real, nb::ndim<1>>;
using NumpyArray2D = nb::ndarray<nb::numpy, Real, nb::ndim<2>>;
using NumpyArray4D = nb::ndarray<nb::numpy, Real, nb::ndim<4>>;
using NumpyArray5D = nb::ndarray<nb::numpy, Real, nb::ndim<5>>;

// Create a 1D numpy array that owns its data
template <typename F>
NumpyArray1D MakeArray(const std::size_t n, F &&f) {
  auto *data = new Real[n];
  for (std::size_t i = 0; i < n; ++i) {
    data[i] = f(static_cast<int>(i));
  }
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<Real *>(p); });
  return NumpyArray1D(data, {n}, owner);
}

// Create a 2D numpy array that owns its data
template <typename F>
NumpyArray2D MakeArray2D(const std::size_t n0, const std::size_t n1, F &&f) {
  auto *data = new Real[n0 * n1];
  for (std::size_t i0 = 0; i0 < n0; ++i0) {
    for (std::size_t i1 = 0; i1 < n1; ++i1) {
      data[i0 * n1 + i1] = f(static_cast<int>(i0), static_cast<int>(i1));
    }
  }
  nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<Real *>(p); });
  return NumpyArray2D(data, {n0, n1}, owner);
}

// Coordinates of a block, together with the number of cells (including ghosts) in each
// direction, so that coordinates can be returned as arrays covering the whole block.
struct BlockCoordinates {
  Coordinates_t coords;
  std::array<int, 3> ncells; // x1, x2, x3

  template <int dir>
  NumpyArray1D Xc() const {
    return MakeArray(ncells[dir - 1], [&](int i) { return coords.template Xc<dir>(i); });
  }
  template <int dir>
  NumpyArray1D Xf() const {
    return MakeArray(ncells[dir - 1] + 1,
                     [&](int i) { return coords.template Xf<dir>(i); });
  }
  template <int dir>
  Real Dx() const {
    return coords.template Dxc<dir>();
  }
};

template <typename F>
auto DispatchDir(const int dir, F &&f) {
  switch (dir) {
  case 1:
    return f(std::integral_constant<int, 1>{});
  case 2:
    return f(std::integral_constant<int, 2>{});
  case 3:
    return f(std::integral_constant<int, 3>{});
  default:
    throw nb::value_error("dir must be 1, 2, or 3");
  }
}

// Host copy of a variable on a block
class HostVariable {
 public:
  explicit HostVariable(const std::shared_ptr<Variable<Real>> &var)
      : device_(var->data.KokkosView()) {}

  // Copy device data to the host. On host-only builds the mirror aliases the device
  // data and no copy is made.
  void Pull() {
    host_ = Kokkos::create_mirror_view(Kokkos::HostSpace(), device_);
    Kokkos::deep_copy(host_, device_);
  }
  void Push() { Kokkos::deep_copy(device_, host_); }

  // numpy view of the host data indexed [component, k, j, i], with all tensor
  // dimensions flattened into the component dimension
  NumpyArray4D Array(nb::handle owner) const {
    // The view is LayoutRight with the three spatial dimensions last.
    constexpr int rank = DeviceView_t::rank;
    static_assert(rank >= 4);
    std::size_t ncomp = 1;
    for (int d = 0; d < rank - 3; ++d) {
      ncomp *= host_.extent(d);
    }
    const std::size_t shape[4] = {ncomp, host_.extent(rank - 3), host_.extent(rank - 2),
                                  host_.extent(rank - 1)};
    const int64_t strides[4] = {static_cast<int64_t>(host_.stride(rank - 4)),
                                static_cast<int64_t>(host_.stride(rank - 3)),
                                static_cast<int64_t>(host_.stride(rank - 2)),
                                static_cast<int64_t>(host_.stride(rank - 1))};
    return NumpyArray4D(host_.data(), 4, shape, owner, strides);
  }

 private:
  using DeviceView_t =
      std::remove_reference_t<decltype(std::declval<Variable<Real>>().data.KokkosView())>;
  DeviceView_t device_;
  typename DeviceView_t::host_mirror_type host_;
};

// Variables of a block selected by base name or metadata flags, grouped by base name
// and then sparse ID (InvalidSparseID for dense variables)
struct SelectedVariables {
  std::vector<std::string> base_names;
  std::unordered_map<std::string, std::map<int, std::shared_ptr<Variable<Real>>>> members;

  void Add(const std::string &base_name) {
    if (members.count(base_name) == 0) base_names.push_back(base_name);
    members[base_name];
  }
  void Add(const std::shared_ptr<Variable<Real>> &var) {
    Add(var->base_name());
    members[var->base_name()][var->GetSparseID()] = var;
  }
};

// Select the variables with the given base names (including all allocated members of
// sparse pools) or, if names is empty, all allocated variables with all of the given
// flags
SelectedVariables SelectVariables(const MeshBlockData<Real> &data,
                                  const std::optional<std::vector<std::string>> &names,
                                  const std::vector<MetadataFlag> &flags) {
  SelectedVariables selected;
  const auto &all_vars = data.GetVariableVector();
  if (names) {
    for (const auto &name : *names) {
      bool found = false;
      for (const auto &var : all_vars) {
        if (var->base_name() != name) continue;
        found = true;
        if (var->IsAllocated()) {
          selected.Add(var);
        } else if (!var->IsSparse()) {
          throw nb::value_error(
              ("Variable " + name + " is not allocated on this block").c_str());
        }
      }
      if (!found) {
        throw nb::key_error(("No variable " + name + " on this block").c_str());
      }
      // A sparse pool without allocated members is still in the pack
      selected.Add(name);
    }
  } else {
    for (const auto &var : all_vars) {
      bool match = var->IsAllocated();
      for (const auto &flag : flags) {
        match = match && var->IsSet(flag);
      }
      if (match) selected.Add(var);
    }
  }
  return selected;
}

// Common implementation of packs that give Python access to variables through host
// copies, playing the role of a SparsePack on the host. Variables are identified by
// their base names, so a sparse pool is packed with all of its allocated members.
// Entering the context copies the variables to the host, where they can be accessed as
// numpy arrays; exiting copies them back to the device. HostVar provides Pull, Push,
// and Array for one variable.
template <typename HostVar>
class HostPackBase {
 public:
  using Array_t = decltype(std::declval<const HostVar &>().Array(nb::handle()));

  void Enter() {
    for (auto &var : vars_) {
      var.Pull();
    }
    entered_ = true;
  }
  void Exit() {
    for (auto &var : vars_) {
      var.Push();
    }
    entered_ = false;
  }

  Array_t Get(nb::handle self, const std::string &name,
              std::optional<int> sparse_id) const {
    if (!entered_) {
      throw nb::value_error("Pack data can only be accessed inside a with block");
    }
    return Var(name, sparse_id).Array(self);
  }

  // Sparse IDs of the allocated members of a sparse pool
  std::vector<int> SparseIDs(const std::string &name) const {
    std::vector<int> ids;
    for (const auto &[sid, idx] : Members(name)) {
      if (sid != InvalidSparseID) ids.push_back(sid);
    }
    return ids;
  }

  const std::vector<std::string> &BaseNames() const { return base_names_; }
  bool Contains(const std::string &name) const { return index_.count(name) > 0; }

 protected:
  void AddBaseName(const std::string &base_name) {
    if (index_.count(base_name) == 0) base_names_.push_back(base_name);
    index_[base_name];
  }
  void AddVariable(const std::string &base_name, int sparse_id, HostVar &&var) {
    AddBaseName(base_name);
    index_[base_name][sparse_id] = vars_.size();
    vars_.push_back(std::move(var));
  }

  const HostVar &Var(const std::string &name, std::optional<int> sparse_id) const {
    const auto &members = Members(name);
    const int sid = sparse_id.value_or(InvalidSparseID);
    const auto it = members.find(sid);
    if (it == members.end()) {
      if (!sparse_id) {
        throw nb::key_error((name + " is a sparse pool, use pack[" + name +
                             ", sparse_id] to access its members")
                                .c_str());
      }
      throw nb::key_error(
          ("Sparse ID " + std::to_string(sid) + " of " + name + " is not allocated")
              .c_str());
    }
    return vars_[it->second];
  }

 private:
  const std::map<int, std::size_t> &Members(const std::string &name) const {
    const auto it = index_.find(name);
    if (it == index_.end()) throw nb::key_error(name.c_str());
    return it->second;
  }

  std::vector<HostVar> vars_;
  std::vector<std::string> base_names_;
  // base name -> sparse ID (InvalidSparseID for dense variables) -> index into vars_
  std::unordered_map<std::string, std::map<int, std::size_t>> index_;
  bool entered_ = false;
};

// Pack of variables on one block, accessed as arrays indexed [component, k, j, i]
class HostPack : public HostPackBase<HostVariable> {
 public:
  HostPack(MeshBlock &mb, const std::optional<std::vector<std::string>> &names,
           const std::vector<MetadataFlag> &flags) {
    const auto selected = SelectVariables(*mb.meshblock_data.Get(), names, flags);
    for (const auto &base_name : selected.base_names) {
      AddBaseName(base_name);
      for (const auto &[sid, var] : selected.members.at(base_name)) {
        AddVariable(base_name, sid, HostVariable(var));
      }
    }
  }
};

// Host copy of a variable on several blocks. The block data, which is only contiguous
// within each block, is gathered on the device into a single buffer indexed
// [block, component, k, j, i] that is contiguous over all blocks, and that buffer is
// copied to the host. Push scatters the data back to the blocks.
class HostMeshVariable {
  using BlockView_t =
      std::remove_reference_t<decltype(std::declval<Variable<Real>>().data.KokkosView())>;
  using MemSpace_t = typename BlockView_t::memory_space;
  using Buffer_t = Kokkos::View<Real *****, Kokkos::LayoutRight, MemSpace_t>;
  using Unmanaged4D_t = Kokkos::View<Real ****, Kokkos::LayoutRight, MemSpace_t,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // Row of the buffer holding one block, and the data of that block with the same shape
  auto Row(std::size_t row) const {
    return Kokkos::subview(buffer_, row, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }
  Unmanaged4D_t BlockData(std::size_t row) const {
    return Unmanaged4D_t(block_views_[row].data(), buffer_.extent(1), buffer_.extent(2),
                         buffer_.extent(3), buffer_.extent(4));
  }

 public:
  // rows holds the index of each block in the MeshData and the variable on that block
  explicit HostMeshVariable(
      const std::vector<std::pair<int, std::shared_ptr<Variable<Real>>>> &rows) {
    for (const auto &[block, var] : rows) {
      blocks_.push_back(block);
      block_views_.push_back(var->data.KokkosView());
    }
    // The block views are LayoutRight with the three spatial dimensions last
    const auto &v = block_views_.front();
    constexpr int rank = BlockView_t::rank;
    static_assert(rank >= 4);
    std::size_t ncomp = 1;
    for (int d = 0; d < rank - 3; ++d) {
      ncomp *= v.extent(d);
    }
    buffer_ =
        Buffer_t(Kokkos::view_alloc(Kokkos::WithoutInitializing, "HostMeshVariable"),
                 blocks_.size(), ncomp, v.extent(rank - 3), v.extent(rank - 2),
                 v.extent(rank - 1));
  }

  void Pull() {
    for (std::size_t row = 0; row < blocks_.size(); ++row) {
      Kokkos::deep_copy(Row(row), BlockData(row));
    }
    host_ = Kokkos::create_mirror_view(Kokkos::HostSpace(), buffer_);
    Kokkos::deep_copy(host_, buffer_);
  }
  void Push() {
    Kokkos::deep_copy(buffer_, host_);
    for (std::size_t row = 0; row < blocks_.size(); ++row) {
      Kokkos::deep_copy(BlockData(row), Row(row));
    }
  }

  NumpyArray5D Array(nb::handle owner) const {
    std::size_t shape[5];
    int64_t strides[5];
    for (int d = 0; d < 5; ++d) {
      shape[d] = host_.extent(d);
      strides[d] = static_cast<int64_t>(host_.stride(d));
    }
    return NumpyArray5D(host_.data(), 5, shape, owner, strides);
  }

  const std::vector<int> &Blocks() const { return blocks_; }

 private:
  std::vector<int> blocks_;
  std::vector<BlockView_t> block_views_;
  Buffer_t buffer_;
  typename Buffer_t::host_mirror_type host_;
};

// Pack of variables on all blocks of a MeshData, accessed as arrays indexed
// [block, component, k, j, i]. Sparse pool members only have rows for the blocks they
// are allocated on (see BlockIndices).
class HostMeshPack : public HostPackBase<HostMeshVariable> {
 public:
  HostMeshPack(MeshData<Real> &md, const std::optional<std::vector<std::string>> &names,
               const std::vector<MetadataFlag> &flags) {
    // base name -> sparse ID -> (block index, variable) for the blocks it is allocated on
    std::vector<std::string> base_names;
    std::unordered_map<
        std::string,
        std::map<int, std::vector<std::pair<int, std::shared_ptr<Variable<Real>>>>>>
        rows;
    for (int b = 0; b < md.NumBlocks(); ++b) {
      const auto selected = SelectVariables(*md.GetBlockData(b), names, flags);
      for (const auto &base_name : selected.base_names) {
        if (rows.count(base_name) == 0) base_names.push_back(base_name);
        auto &members = rows[base_name];
        for (const auto &[sid, var] : selected.members.at(base_name)) {
          members[sid].emplace_back(b, var);
        }
      }
    }
    for (const auto &base_name : base_names) {
      AddBaseName(base_name);
      for (const auto &[sid, var_rows] : rows.at(base_name)) {
        AddVariable(base_name, sid, HostMeshVariable(var_rows));
      }
    }
  }

  // Index in the MeshData of the block of each row of a variable's array
  const std::vector<int> &BlockIndices(const std::string &name,
                                       std::optional<int> sparse_id) const {
    return Var(name, sparse_id).Blocks();
  }
};

// Coordinates of all blocks of a MeshData
struct MeshCoordinates {
  std::vector<Coordinates_t> coords;
  std::array<int, 3> ncells; // x1, x2, x3

  template <int dir>
  NumpyArray2D Xc() const {
    return MakeArray2D(coords.size(), ncells[dir - 1],
                       [&](int b, int i) { return coords[b].template Xc<dir>(i); });
  }
  template <int dir>
  NumpyArray2D Xf() const {
    return MakeArray2D(coords.size(), ncells[dir - 1] + 1,
                       [&](int b, int i) { return coords[b].template Xf<dir>(i); });
  }
  template <int dir>
  NumpyArray1D Dx() const {
    return MakeArray(coords.size(), [&](int b) { return coords[b].template Dxc<dir>(); });
  }
};

std::tuple<int, int> Range(const IndexRange &r) { return {r.s, r.e}; }

auto Bounds(const IndexShape &cb, IndexDomain domain) {
  return std::make_tuple(Range(cb.GetBoundsK(domain)), Range(cb.GetBoundsJ(domain)),
                         Range(cb.GetBoundsI(domain)));
}

template <typename Pack>
void BindPackInterface(nb::class_<Pack> &cls) {
  cls.def(
         "__enter__",
         [](Pack &self) -> Pack & {
           self.Enter();
           return self;
         },
         nb::rv_policy::reference)
      .def(
          "__exit__", [](Pack &self, nb::args) { self.Exit(); },
          nb::sig("def __exit__(self, *args: object) -> None"))
      .def(
          "__getitem__",
          [](nb::handle self, const std::string &name) {
            return nb::cast<const Pack &>(self).Get(self, name, std::nullopt);
          },
          nb::arg("name"), "Array of the dense variable name.")
      .def(
          "__getitem__",
          [](nb::handle self, std::tuple<std::string, int> key) {
            const auto &[name, sparse_id] = key;
            return nb::cast<const Pack &>(self).Get(self, name, sparse_id);
          },
          nb::arg("key"), "Array of the member sparse_id of the sparse pool name.")
      .def("__contains__", &Pack::Contains, nb::arg("name"))
      .def("__len__", [](const Pack &self) { return self.BaseNames().size(); })
      .def("sparse_ids", &Pack::SparseIDs, nb::arg("name"),
           "Sparse IDs of the allocated members of the sparse pool name.")
      .def_prop_ro("names", &Pack::BaseNames, "Base names of the packed variables.");
}
} // namespace

void BindMesh(nb::module_ &m) {
  nb::enum_<IndexDomain>(m, "IndexDomain")
      .value("entire", IndexDomain::entire)
      .value("interior", IndexDomain::interior)
      .value("inner_x1", IndexDomain::inner_x1)
      .value("outer_x1", IndexDomain::outer_x1)
      .value("inner_x2", IndexDomain::inner_x2)
      .value("outer_x2", IndexDomain::outer_x2)
      .value("inner_x3", IndexDomain::inner_x3)
      .value("outer_x3", IndexDomain::outer_x3);

  nb::class_<SimTime>(m, "SimTime")
      .def_ro("start_time", &SimTime::start_time)
      .def_ro("time", &SimTime::time)
      .def_ro("tlim", &SimTime::tlim)
      .def_ro("dt", &SimTime::dt)
      .def_ro("ncycle", &SimTime::ncycle)
      .def_ro("nlim", &SimTime::nlim);

  nb::class_<BlockCoordinates>(m, "BlockCoordinates",
                               "Coordinates of a block. Arrays cover the entire block, "
                               "including ghost zones, and are indexed like the "
                               "corresponding dimension of a pack array.")
      .def(
          "Xc",
          [](const BlockCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Xc<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Cell center positions in direction dir (1, 2, or 3).")
      .def(
          "Xf",
          [](const BlockCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Xf<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Face positions in direction dir (1, 2, or 3).")
      .def(
          "Dx",
          [](const BlockCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Dx<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Cell width in direction dir (1, 2, or 3).");

  nb::class_<HostPack> host_pack(
      m, "HostPack",
      "Variables of a block copied to the host. Use as a context manager; inside the "
      "with block, pack[name] (or pack[name, sparse_id] for members of a sparse pool) is "
      "a numpy array indexed [component, k, j, i] whose changes are copied back to the "
      "device on exit.");
  BindPackInterface(host_pack);

  nb::class_<HostMeshPack> host_mesh_pack(
      m, "HostMeshPack",
      "Variables of all blocks of a MeshData copied to the host. Use as a context "
      "manager; inside the with block, pack[name] (or pack[name, sparse_id] for members "
      "of a sparse pool) is a numpy array indexed [block, component, k, j, i] that is "
      "contiguous over all blocks. Changes are copied back to the blocks on exit.");
  BindPackInterface(host_mesh_pack);
  host_mesh_pack.def(
      "block_indices",
      [](const HostMeshPack &self, const std::string &name,
         std::optional<int> sparse_id) { return self.BlockIndices(name, sparse_id); },
      nb::arg("name"), nb::arg("sparse_id") = nb::none(),
      "Index in the MeshData of the block of each row of the array of a variable. Rows "
      "are only present for the blocks a sparse pool member is allocated on.");

  nb::class_<MeshCoordinates>(
      m, "MeshCoordinates",
      "Coordinates of all blocks of a MeshData, as arrays indexed "
      "[block, index] that cover the entire blocks, including "
      "ghost zones.")
      .def(
          "Xc",
          [](const MeshCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Xc<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Cell center positions in direction dir (1, 2, or 3).")
      .def(
          "Xf",
          [](const MeshCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Xf<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Face positions in direction dir (1, 2, or 3).")
      .def(
          "Dx",
          [](const MeshCoordinates &c, int dir) {
            return DispatchDir(dir, [&](auto d) { return c.Dx<decltype(d)::value>(); });
          },
          nb::arg("dir"), "Cell width of each block in direction dir (1, 2, or 3).");

  nb::class_<MeshBlock>(m, "MeshBlock")
      .def_ro("gid", &MeshBlock::gid)
      .def_ro("lid", &MeshBlock::lid)
      .def_prop_ro(
          "packages", [](MeshBlock &self) -> Packages_t & { return self.packages; },
          nb::rv_policy::reference_internal)
      .def_prop_ro("coords",
                   [](MeshBlock &self) {
                     const auto &cb = self.cellbounds;
                     return BlockCoordinates{self.coords,
                                             {cb.ncellsi(IndexDomain::entire),
                                              cb.ncellsj(IndexDomain::entire),
                                              cb.ncellsk(IndexDomain::entire)}};
                   })
      .def(
          "bounds",
          [](MeshBlock &self, IndexDomain domain) {
            return Bounds(self.cellbounds, domain);
          },
          nb::arg("domain") = IndexDomain::interior,
          "Inclusive index ranges ((ks, ke), (js, je), (is, ie)) of a domain.")
      .def(
          "variable_names",
          [](MeshBlock &self, const std::vector<MetadataFlag> &flags) {
            std::vector<std::string> names;
            for (const auto &var : self.meshblock_data.Get()->GetVariableVector()) {
              bool match = true;
              for (const auto &flag : flags) {
                match = match && var->IsSet(flag);
              }
              if (match) names.push_back(var->label());
            }
            return names;
          },
          nb::arg("flags") = std::vector<MetadataFlag>{},
          "Names of the variables on this block that have all of the given flags.")
      .def(
          "pack",
          [](MeshBlock &self, std::optional<std::vector<std::string>> names,
             const std::vector<MetadataFlag> &flags) {
            return HostPack(self, names, flags);
          },
          nb::arg("names") = nb::none(), nb::arg("flags") = std::vector<MetadataFlag>{},
          "Pack variables of this block for access on the host. Packs the variables "
          "with the given base names (including all allocated members of sparse "
          "pools) or, if names is None, all allocated variables with all of the given "
          "flags.");

  using MD = MeshData<Real>;
  nb::class_<MD>(m, "MeshData", "The data of a set of blocks (a partition of the mesh).")
      .def_prop_ro("mesh", &MD::GetMeshPointer, nb::rv_policy::reference)
      .def("NumBlocks", &MD::NumBlocks)
      .def(
          "block",
          [](MD &self, int b) {
            if (b < 0 || b >= self.NumBlocks()) throw nb::index_error();
            return self.GetBlockData(b)->GetBlockPointer();
          },
          nb::arg("b"), nb::rv_policy::reference)
      .def_prop_ro("blocks",
                   [](MD &self) {
                     std::vector<std::shared_ptr<MeshBlock>> blocks;
                     for (int b = 0; b < self.NumBlocks(); ++b) {
                       blocks.push_back(self.GetBlockData(b)->GetBlockSharedPointer());
                     }
                     return blocks;
                   })
      .def(
          "bounds",
          [](MD &self, IndexDomain domain) {
            // All blocks have the same shape
            return Bounds(self.GetBlockData(0)->GetBlockPointer()->cellbounds, domain);
          },
          nb::arg("domain") = IndexDomain::interior,
          "Inclusive index ranges ((ks, ke), (js, je), (is, ie)) of a domain, which are "
          "the same on all blocks.")
      .def_prop_ro("coords",
                   [](MD &self) {
                     MeshCoordinates coords;
                     for (int b = 0; b < self.NumBlocks(); ++b) {
                       coords.coords.push_back(
                           self.GetBlockData(b)->GetBlockPointer()->coords);
                     }
                     const auto &cb = self.GetBlockData(0)->GetBlockPointer()->cellbounds;
                     coords.ncells = {cb.ncellsi(IndexDomain::entire),
                                      cb.ncellsj(IndexDomain::entire),
                                      cb.ncellsk(IndexDomain::entire)};
                     return coords;
                   })
      .def(
          "pack",
          [](MD &self, std::optional<std::vector<std::string>> names,
             const std::vector<MetadataFlag> &flags) {
            return HostMeshPack(self, names, flags);
          },
          nb::arg("names") = nb::none(), nb::arg("flags") = std::vector<MetadataFlag>{},
          "Pack variables of all blocks for access on the host as arrays indexed "
          "[block, component, k, j, i] that are contiguous over all blocks. Packs the "
          "variables with the given base names (including all allocated members of "
          "sparse pools) or, if names is None, all allocated variables with all of the "
          "given flags.");

  nb::class_<Mesh>(m, "Mesh")
      .def_ro("ndim", &Mesh::ndim)
      .def_ro("nbtotal", &Mesh::nbtotal)
      .def_prop_ro("block_list", [](Mesh &self) { return self.block_list; })
      .def_prop_ro(
          "packages", [](Mesh &self) -> Packages_t & { return self.packages; },
          nb::rv_policy::reference_internal);
}

} // namespace python
} // namespace parthenon

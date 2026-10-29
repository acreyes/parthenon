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

#include <memory>
#include <string>
#include <string_view>
#include <typeindex>
#include <vector>

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "interface/meshblock_data.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/params.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "parthenon_nanobind.hpp"

namespace parthenon {
namespace python {

namespace {
// Param types that can be read back into Python. Parameters of any other type can
// only be used from C++.
template <typename... Ts>
struct ParamReader {
  static nb::object Read(const StateDescriptor &pkg, const std::string &key) {
    const std::type_index type = pkg.ParamType(key);
    nb::object result;
    const bool found = ((type == std::type_index(typeid(Ts)) &&
                         (result = nb::cast(pkg.Param<Ts>(key)), true)) ||
                        ...);
    if (!found) {
      throw nb::type_error(("Param '" + key + "' of package '" + pkg.label() +
                            "' has type " + type.name() +
                            " which is not readable from Python")
                               .c_str());
    }
    return result;
  }
};
using PyParamReader = ParamReader<bool, int, Real, std::string, std::size_t,
                                  std::vector<bool>, std::vector<int>, std::vector<Real>,
                                  std::vector<std::string>, nb::object>;

// Store Python scalars with the matching C++ type so that C++ code can read them with
// Param<T>; any other Python object is stored as an nb::object.
void AddPyParam(StateDescriptor &pkg, const std::string &key, nb::handle value,
                Params::Mutability mutability) {
  if (nb::isinstance<nb::bool_>(value)) {
    pkg.AddParam(key, nb::cast<bool>(value), mutability);
  } else if (nb::isinstance<nb::int_>(value)) {
    pkg.AddParam(key, nb::cast<int>(value), mutability);
  } else if (nb::isinstance<nb::float_>(value)) {
    pkg.AddParam(key, nb::cast<Real>(value), mutability);
  } else if (nb::isinstance<nb::str>(value)) {
    pkg.AddParam(key, nb::cast<std::string>(value), mutability);
  } else {
    pkg.AddParam(key, nb::borrow<nb::object>(value), mutability);
  }
}
} // namespace

void BindState(nb::module_ &m) {
  nb::enum_<TaskStatus>(m, "TaskStatus")
      .value("complete", TaskStatus::complete)
      .value("incomplete", TaskStatus::incomplete)
      .value("iterate", TaskStatus::iterate)
      .value("fail", TaskStatus::fail);

  nb::enum_<AmrTag>(m, "AmrTag")
      .value("derefine", AmrTag::derefine)
      .value("same", AmrTag::same)
      .value("refine", AmrTag::refine);

  nb::class_<MetadataFlag>(m, "MetadataFlag")
      .def_prop_ro("name", &MetadataFlag::Name)
      .def("__eq__",
           [](const MetadataFlag &a, nb::handle b) {
             return nb::isinstance<MetadataFlag>(b) && a == nb::cast<MetadataFlag>(b);
           })
      .def("__repr__",
           [](const MetadataFlag &f) { return "<MetadataFlag " + f.Name() + ">"; });

  nb::class_<Metadata> metadata(m, "Metadata");
  metadata.def(nb::init<>())
      .def(nb::init<const std::vector<MetadataFlag> &, const std::vector<int> &,
                    const std::vector<std::string> &>(),
           nb::arg("flags"), nb::arg("shape") = std::vector<int>{},
           nb::arg("component_labels") = std::vector<std::string>{})
      .def("IsSet", &Metadata::IsSet, nb::arg("flag"))
      .def_prop_ro("shape", &Metadata::Shape);
  // The flags are class attributes. None (no topology) is not a valid Python attribute
  // name and is the default topology, so it is not bound.
#define PARTHENON_INTERNAL_FOR_FLAG(name)                                                \
  if (std::string_view(#name) != "None") metadata.attr(#name) = Metadata::name;
  PARTHENON_INTERNAL_FOREACH_BUILTIN_FLAG
#undef PARTHENON_INTERNAL_FOR_FLAG

  nb::enum_<Params::Mutability>(m, "Mutability")
      .value("Immutable", Params::Mutability::Immutable)
      .value("Mutable", Params::Mutability::Mutable)
      .value("Restart", Params::Mutability::Restart);

  // Passed to package callbacks; gives access to the block that the data belongs to.
  // MeshData is bound in mesh.cpp.
  nb::class_<MeshBlockData<Real>>(m, "MeshBlockData")
      .def_prop_ro("block", &MeshBlockData<Real>::GetBlockPointer,
                   nb::rv_policy::reference);

  nb::class_<StateDescriptor>(m, "StateDescriptor")
      .def(nb::init<const std::string &>(), nb::arg("label"))
      .def_prop_ro("label", &StateDescriptor::label)
      .def(
          "AddField",
          [](StateDescriptor &self, const std::string &name, const Metadata &md) {
            return self.AddField(name, md);
          },
          nb::arg("name"), nb::arg("metadata"))
      .def(
          "AddParam",
          [](StateDescriptor &self, const std::string &key, nb::handle value,
             Params::Mutability mutability) { AddPyParam(self, key, value, mutability); },
          nb::arg("key"), nb::arg("value"),
          nb::arg("mutability") = Params::Mutability::Immutable,
          "Add a parameter. bool, int, float and str values are stored as the "
          "corresponding C++ type; any other object is stored as a Python object.")
      .def("Param", &PyParamReader::Read, nb::arg("key"),
           nb::sig("def Param(self, key: str) -> typing.Any"))
      .def(
          "HasParam",
          [](StateDescriptor &self, const std::string &key) {
            return self.AllParams().hasKey(key);
          },
          nb::arg("key"))
      .def_rw("PreFillDerivedBlock", &StateDescriptor::PreFillDerivedBlock)
      .def_rw("FillDerivedBlock", &StateDescriptor::FillDerivedBlock)
      .def_rw("PostFillDerivedBlock", &StateDescriptor::PostFillDerivedBlock)
      .def_rw("PreFillDerivedMesh", &StateDescriptor::PreFillDerivedMesh)
      .def_rw("FillDerivedMesh", &StateDescriptor::FillDerivedMesh)
      .def_rw("PostFillDerivedMesh", &StateDescriptor::PostFillDerivedMesh)
      .def_rw("EstimateTimestepBlock", &StateDescriptor::EstimateTimestepBlock)
      .def_rw("EstimateTimestepMesh", &StateDescriptor::EstimateTimestepMesh)
      .def_rw("CheckRefinementBlock", &StateDescriptor::CheckRefinementBlock)
      .def_rw("UserWorkBeforeLoopMesh", &StateDescriptor::UserWorkBeforeLoopMesh);

  nb::class_<Packages_t>(m, "Packages")
      .def(nb::init<>())
      .def("Add", &Packages_t::Add, nb::arg("package"))
      .def(
          "Get",
          [](const Packages_t &self, const std::string &name) {
            const auto &all = self.AllPackages();
            const auto it = all.find(name);
            if (it == all.end()) throw nb::key_error(name.c_str());
            return it->second;
          },
          nb::arg("name"))
      .def("__contains__",
           [](const Packages_t &self, const std::string &name) {
             return self.AllPackages().count(name) > 0;
           })
      .def("names", [](const Packages_t &self) {
        std::vector<std::string> names;
        for (const auto &[name, pkg] : self.AllPackages()) {
          names.push_back(name);
        }
        return names;
      });
}

} // namespace python
} // namespace parthenon

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

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "parameter_input.hpp"
#include "parthenon_nanobind.hpp"

namespace parthenon {
namespace python {

namespace {
// Values that can be added from Python. Strings are added unresolved, exactly like
// values read from an input file, so they can later be read as any type.
using PyParamValue = std::variant<bool, int, Real, std::string>;

ParamValue ToParamValue(const PyParamValue &value) {
  if (const auto *str = std::get_if<std::string>(&value)) {
    return UnresolvedString(*str);
  }
  return std::visit([](const auto &v) -> ParamValue { return v; }, value);
}

std::pair<std::string, std::string> SplitPath(const std::string &path) {
  const auto pos = path.rfind('/');
  if (pos == std::string::npos) {
    throw nb::key_error(
        ("Expected a key of the form 'block/name', got '" + path + "'").c_str());
  }
  return {path.substr(0, pos), path.substr(pos + 1)};
}
} // namespace

void BindParameterInput(nb::module_ &m) {
  using PI = ParameterInput;
  nb::class_<PI>(m, "ParameterInput",
                 "Runtime input parameters, organized into blocks of named values.\n\n"
                 "Parameters may be added (add, load_string, load_file) until the first "
                 "time a value is read, after which parsing is finalized.")
      .def(nb::init<>())
      .def(
          "load_file",
          [](PI &self, const std::string &filename) {
            // Read with a plain stream rather than ParameterInput(filename), which
            // uses MPI and so cannot be used before ParthenonInitEnv.
            std::ifstream is(filename);
            if (!is)
              throw nb::value_error(("Cannot open input file " + filename).c_str());
            self.LoadFromStream(is);
          },
          nb::arg("filename"), "Add parameters from a Parthenon input file.")
      .def(
          "load_string",
          [](PI &self, const std::string &input) {
            std::istringstream is(input);
            self.LoadFromStream(is);
          },
          nb::arg("input"), "Add parameters from a string in input file format.")
      .def(
          "add",
          [](PI &self, const std::string &block, const std::string &name,
             const PyParamValue &value) {
            self.AddParsedParameter(block, name, ToParamValue(value), "# From Python");
          },
          nb::arg("block"), nb::arg("name"), nb::arg("value"),
          "Add (or override) a parameter during parsing.")
      .def("overlay", &PI::Overlay, nb::arg("other"),
           "Add all parameters from other, overriding existing ones.")
      .def("finalize_parsing", &PI::FinalizeParsing)
      .def("block_names", &PI::GetBlockNames)
      .def("parameter_names", &PI::GetParameterNames, nb::arg("block"))
      .def("has", &PI::DoesParameterExist, nb::arg("block"), nb::arg("name"))
      .def("has_block", &PI::DoesBlockExist, nb::arg("block"))
      .def("get_string_repr", &PI::GetAsUnresolvedString, nb::arg("block"),
           nb::arg("name"))
      .def(
          "GetInteger",
          [](PI &self, const std::string &b, const std::string &n) {
            return self.GetInteger(b, n);
          },
          nb::arg("block"), nb::arg("name"))
      .def(
          "GetReal",
          [](PI &self, const std::string &b, const std::string &n) {
            return self.GetReal(b, n);
          },
          nb::arg("block"), nb::arg("name"))
      .def(
          "GetBoolean",
          [](PI &self, const std::string &b, const std::string &n) {
            return self.GetBoolean(b, n);
          },
          nb::arg("block"), nb::arg("name"))
      .def(
          "GetString",
          [](PI &self, const std::string &b, const std::string &n) {
            return self.GetString(b, n);
          },
          nb::arg("block"), nb::arg("name"))
      .def(
          "GetOrAddInteger",
          [](PI &self, const std::string &b, const std::string &n, int v) {
            return self.GetOrAddInteger(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("default"))
      .def(
          "GetOrAddReal",
          [](PI &self, const std::string &b, const std::string &n, Real v) {
            return self.GetOrAddReal(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("default"))
      .def(
          "GetOrAddBoolean",
          [](PI &self, const std::string &b, const std::string &n, bool v) {
            return self.GetOrAddBoolean(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("default"))
      .def(
          "GetOrAddString",
          [](PI &self, const std::string &b, const std::string &n, const std::string &v) {
            return self.GetOrAddString(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("default"))
      .def(
          "SetInteger",
          [](PI &self, const std::string &b, const std::string &n, int v) {
            self.SetInteger(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("value"))
      .def(
          "SetReal",
          [](PI &self, const std::string &b, const std::string &n, Real v) {
            self.SetReal(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("value"))
      .def(
          "SetBoolean",
          [](PI &self, const std::string &b, const std::string &n, bool v) {
            self.SetBoolean(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("value"))
      .def(
          "SetString",
          [](PI &self, const std::string &b, const std::string &n, const std::string &v) {
            self.SetString(b, n, v);
          },
          nb::arg("block"), nb::arg("name"), nb::arg("value"))
      .def(
          "__contains__",
          [](PI &self, const std::string &path) {
            const auto [block, name] = SplitPath(path);
            return self.DoesParameterExist(block, name);
          },
          nb::arg("path"))
      .def(
          "__getitem__",
          [](PI &self, const std::string &path) {
            const auto [block, name] = SplitPath(path);
            if (!self.DoesParameterExist(block, name)) throw nb::key_error(path.c_str());
            return self.GetAsUnresolvedString(block, name);
          },
          nb::arg("path"),
          "Return the string representation of 'block/name'. Use the typed getters "
          "to read a value as a specific type.")
      .def(
          "__setitem__",
          [](PI &self, const std::string &path, const PyParamValue &value) {
            const auto [block, name] = SplitPath(path);
            self.AddParsedParameter(block, name, ToParamValue(value), "# From Python");
          },
          nb::arg("path"), nb::arg("value"))
      .def("dump", [](PI &self) {
        std::ostringstream os;
        self.ParameterDump(os);
        return os.str();
      });
}

} // namespace python
} // namespace parthenon

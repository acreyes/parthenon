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
// This file was created in part with generative AI.

#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include "argument_parser.hpp"

using parthenon::ArgParse;
using parthenon::ArgStatus;

namespace {
ArgStatus Parse(ArgParse &arg, std::vector<std::string> args) {
  std::vector<char *> argv;
  for (auto &a : args) {
    argv.push_back(a.data());
  }
  argv.push_back(nullptr);
  return arg.parse(static_cast<int>(args.size()), argv.data());
}
} // namespace

TEST_CASE("ArgParse fills RuntimeOptions", "[ArgParse]") {
  SECTION("Input file and run options") {
    ArgParse arg;
    REQUIRE(Parse(arg, {"exe", "-i", "input.in", "-d", "rundir", "-t", "01:02:03", "-w",
                        "30", "-m", "4", "block/par=1"}) == ArgStatus::ok);
    REQUIRE(arg.input_filename.value() == "input.in");
    REQUIRE(!arg.opts.IsRestart());
    REQUIRE(!arg.opts.analysis);
    REQUIRE(arg.opts.run_dir.value() == "rundir");
    REQUIRE(arg.opts.wall_time_limit == 3600 + 2 * 60 + 3);
    REQUIRE(arg.opts.watchdog_timeout.value() == 30);
    REQUIRE(arg.opts.mesh_flag == 4);
    REQUIRE(!arg.opts.params_regex);
  }

  SECTION("Restart and analysis") {
    ArgParse restart;
    REQUIRE(Parse(restart, {"exe", "-r", "out.rhdf"}) == ArgStatus::ok);
    REQUIRE(restart.opts.IsRestart());
    REQUIRE(restart.opts.restart_filename.value() == "out.rhdf");
    REQUIRE(!restart.opts.analysis);
    REQUIRE(!restart.input_filename);

    ArgParse analysis;
    REQUIRE(Parse(analysis, {"exe", "-a", "out.rhdf"}) == ArgStatus::ok);
    REQUIRE(analysis.opts.IsRestart());
    REQUIRE(analysis.opts.analysis);
  }

  SECTION("Parameter regex defaults to wildcard") {
    ArgParse arg;
    REQUIRE(Parse(arg, {"exe", "-i", "input.in", "-p"}) == ArgStatus::ok);
    REQUIRE(arg.opts.params_regex.value() == "(.*)");

    ArgParse arg_regex;
    REQUIRE(Parse(arg_regex, {"exe", "-i", "input.in", "-p", "parthenon/.*"}) ==
            ArgStatus::ok);
    REQUIRE(arg_regex.opts.params_regex.value() == "parthenon/.*");
  }

  SECTION("Missing input is an error") {
    ArgParse arg;
    REQUIRE(Parse(arg, {"exe", "-d", "rundir"}) == ArgStatus::error);
  }
}

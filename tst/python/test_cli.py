# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2026 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================
"""Tests of parthenon.cli, run against the parthenon package of the Python advection
example (built with parthenon_add_python_module), which must be on the Python path."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest

from advection import parthenon
from advection.parthenon import cli


def parse(*argv, parser=None):
    return cli.parse_args(list(argv), parser)


class TestParseArgs(unittest.TestCase):
    def assert_exits(self, *argv, parser=None):
        """Parsing argv exits with status 2 (argparse error)."""
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(
            io.StringIO()
        ):
            parse(*argv, parser=parser)
        self.assertEqual(cm.exception.code, 2)

    def test_defaults(self):
        args = parse("-i", "in.txt")
        self.assertEqual(args.input_file, "in.txt")
        opts = args.options
        self.assertIsNone(opts.restart_filename)
        self.assertFalse(opts.analysis)
        self.assertIsNone(opts.run_dir)
        self.assertIsNone(opts.params_regex)
        self.assertEqual(opts.mesh_flag, 0)
        self.assertEqual(opts.wall_time_limit, 0)
        self.assertIsNone(opts.watchdog_timeout)
        self.assertEqual(opts.kokkos_args, [])
        self.assertEqual(args.overrides, [])

    def test_options(self):
        opts = parse(
            "-r", "rst.rhdf", "-d", "run", "-m", "4", "-t", "01:02:03", "-w", "30"
        ).options
        self.assertEqual(opts.restart_filename, "rst.rhdf")
        self.assertTrue(opts.is_restart)
        self.assertFalse(opts.analysis)
        self.assertEqual(opts.run_dir, "run")
        self.assertEqual(opts.mesh_flag, 4)
        self.assertEqual(opts.wall_time_limit, 3723)
        self.assertEqual(opts.watchdog_timeout, 30)

    def test_analysis(self):
        opts = parse("-a", "rst.rhdf").options
        self.assertEqual(opts.restart_filename, "rst.rhdf")
        self.assertTrue(opts.analysis)

    def test_params_regex(self):
        self.assertEqual(parse("-i", "in", "-p").options.params_regex, "(.*)")
        self.assertEqual(parse("-i", "in", "-p", "mesh").options.params_regex, "mesh")

    def test_overrides_and_kokkos(self):
        args = parse(
            "-i",
            "in",
            "parthenon/mesh/nx1=8",
            "--kokkos-num-threads=2",
            "Advection/profile=a=b",
        )
        self.assertEqual(
            args.overrides,
            [("parthenon/mesh/nx1", "8"), ("Advection/profile", "a=b")],
        )
        self.assertEqual(args.options.kokkos_args, ["--kokkos-num-threads=2"])

    def test_parameter_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_file = os.path.join(tmp, "input")
            with open(input_file, "w") as f:
                f.write("<parthenon/mesh>\nnx1 = 4\nnx2 = 6\n")
            pin = parse(
                "-i", input_file, "parthenon/mesh/nx1=8", "new/block/x=1"
            ).parameter_input()
        self.assertEqual(pin.get_string_repr("parthenon/mesh", "nx1"), "8")
        self.assertEqual(pin.get_string_repr("parthenon/mesh", "nx2"), "6")
        self.assertEqual(pin.get_string_repr("new/block", "x"), "1")

    def test_app_option(self):
        parser = cli.make_parser()
        parser.add_argument("--pgen", choices=["block", "mesh"], default="block")
        self.assertEqual(parse("-i", "in", parser=parser).namespace.pgen, "block")
        args = parse("-i", "in", "--pgen", "mesh", "a/b=1", parser=parser)
        self.assertEqual(args.namespace.pgen, "mesh")
        self.assertEqual(args.overrides, [("a/b", "1")])
        self.assert_exits("-i", "in", "--pgen", "other", parser=parser)

    def test_errors(self):
        self.assert_exits()  # no input file
        self.assert_exits("-i", "in", "bogus")
        self.assert_exits("-i", "in", "novalue=1")
        self.assert_exits("-i", "in", "--unknown")
        self.assert_exits("-r", "a", "-a", "b")
        self.assert_exits("-i", "in", "-t", "10")

    def test_optional_input(self):
        args = cli.parse_args(["a/b=2"], require_input=False)
        self.assertIsNone(args.input_file)
        self.assertFalse(args.options.is_restart)

        base = parthenon.ParameterInput()
        base.add("a", "b", 1)
        base.add("a", "c", 3)
        pin = args.parameter_input(base)
        self.assertEqual(pin.get_string_repr("a", "b"), "2")
        self.assertEqual(pin.get_string_repr("a", "c"), "3")


class TestPackage(unittest.TestCase):
    def test_reexports_bindings(self):
        self.assertIs(parthenon.ParameterInput, parthenon._core.ParameterInput)
        self.assertIsInstance(
            cli.parse_args(["-i", "in"]).options, parthenon.RuntimeOptions
        )

    def test_import_order(self):
        # The parthenon subpackage imports the compiled module itself
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import advection.parthenon as p; p.ParameterInput()",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()

# This file was made in part with generative AI.

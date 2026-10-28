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
"""The Parthenon command line for Python main programs.

parse_args accepts the same command line as a C++ Parthenon executable (see ArgParse in
src/argument_parser.hpp):

    prog [-i FILE] [-r FILE | -a FILE] [-d DIR] [-p [REGEX]] [-m NPROC] [-t hh:mm:ss]
         [-w SECONDS] [-c] [--kokkos-...] [block/par=value ...]

and returns the input file, the RuntimeOptions to pass to
ParthenonManager.ParthenonInitEnv, and the parameter overrides. Applications add their
own options to a parser from make_parser:

    parser = parthenon.cli.make_parser(description="My application")
    parser.add_argument("--my-option")
    args = parthenon.cli.parse_args(argv, parser)
    status = pman.ParthenonInitEnv(args.parameter_input(), args.options)
    args.namespace.my_option

Like a C++ executable, a Python main program is run in parallel by launching it on every
rank with MPI, e.g.,

    mpirun -np 4 python my_app.py -i my_input parthenon/time/tlim=1.0

Each rank parses the command line; MPI is initialized by ParthenonInitEnv.
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence

from . import _core

# Regex used by -p without an argument (ArgParse::default_params_regex)
DEFAULT_PARAMS_REGEX = "(.*)"


def _wall_time(value: str) -> int:
    """Convert hh:mm:ss to seconds."""
    try:
        h, m, s = (int(x) for x in value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected hh:mm:ss, got {value!r}") from None
    return 3600 * h + 60 * m + s


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the Parthenon options to parser."""
    group = parser.add_argument_group("Parthenon options")
    group.add_argument("-i", dest="input_file", metavar="FILE", help="input file")
    restart = group.add_mutually_exclusive_group()
    restart.add_argument(
        "-r", dest="restart_file", metavar="FILE", help="restart with this file"
    )
    restart.add_argument(
        "-a", dest="analysis_file", metavar="FILE", help="analyze/postprocess this file"
    )
    group.add_argument(
        "-d", dest="run_dir", metavar="DIR", help="run directory [current directory]"
    )
    group.add_argument(
        "-p",
        dest="params_regex",
        metavar="REGEX",
        nargs="?",
        const=DEFAULT_PARAMS_REGEX,
        help="parse the input, report parameters for blocks matching REGEX (default: "
        "all blocks) and quit",
    )
    group.add_argument(
        "-c",
        dest="show_config",
        action="store_true",
        help="show the configuration and quit",
    )
    group.add_argument(
        "-m",
        dest="mesh_flag",
        metavar="NPROC",
        type=int,
        default=0,
        help="output the mesh structure for NPROC ranks and quit",
    )
    group.add_argument(
        "-t",
        dest="wall_time_limit",
        metavar="hh:mm:ss",
        type=_wall_time,
        default=0,
        help="wall time limit for the final output",
    )
    group.add_argument(
        "-w",
        dest="watchdog_timeout",
        metavar="SECONDS",
        type=int,
        help="watchdog timeout in seconds",
    )


def make_parser(
    description: str | None = None, prog: str | None = None
) -> argparse.ArgumentParser:
    """Create a parser with the Parthenon options, to which an application can add its
    own options."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        allow_abbrev=False,
        epilog="Input parameters can be overridden with block/par=value arguments, and "
        "--kokkos-* arguments are passed to Kokkos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments(parser)
    return parser


def parse_override(arg: str) -> tuple[str, str] | None:
    """Split a block/par=value argument into ("block/par", "value"), or return None if
    arg is not one. As in ParameterInput::ModifyFromCmdline, the parameter name follows
    the last "/" before the first "="."""
    equal = arg.find("=")
    if equal < 0 or arg.rfind("/", 0, equal) < 0:
        return None
    return arg[:equal], arg[equal + 1 :]


@dataclasses.dataclass
class Arguments:
    """The parsed command line."""

    # Input file (-i)
    input_file: str | None
    # Options to pass to ParthenonManager.ParthenonInitEnv
    options: _core.RuntimeOptions
    # Parameter overrides ("block/par", "value"), in command line order
    overrides: list[tuple[str, str]]
    # All parsed options, including those added by the application
    namespace: argparse.Namespace

    def parameter_input(
        self, base: _core.ParameterInput | None = None
    ) -> _core.ParameterInput:
        """Start from base (e.g., parameters set in Python, default: no parameters), read
        the input file, if any, and apply the overrides."""
        pin = _core.ParameterInput() if base is None else base
        if self.input_file is not None:
            pin.load_file(self.input_file)
        for path, value in self.overrides:
            pin[path] = value
        return pin


def parse_args(
    argv: Sequence[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
    require_input: bool = True,
) -> Arguments:
    """Parse a Parthenon command line (sys.argv[1:] if argv is None) with parser, which
    must have the Parthenon options (see make_parser). Exits with an error message if the
    command line is invalid, and after printing the configuration for -c. Like the C++
    executables, an input file (-i) or restart file (-r or -a) is required unless
    require_input is False, e.g., for an application that sets its parameters in
    Python."""
    if parser is None:
        parser = make_parser()
    namespace, rest = parser.parse_known_args(argv)

    if namespace.show_config:
        _core.ShowConfig()
        parser.exit()

    kokkos_args: list[str] = []
    overrides: list[tuple[str, str]] = []
    for arg in rest:
        override = parse_override(arg)
        if arg.startswith("--kokkos"):
            kokkos_args.append(arg)
        elif override is not None:
            overrides.append(override)
        else:
            parser.error(f"unrecognized argument: {arg}")

    if (
        require_input
        and namespace.input_file is None
        and namespace.restart_file is None
        and namespace.analysis_file is None
    ):
        parser.error("no input file or restart file is specified")

    options = _core.RuntimeOptions(
        restart_filename=namespace.restart_file or namespace.analysis_file,
        analysis=namespace.analysis_file is not None,
        run_dir=namespace.run_dir,
        params_regex=namespace.params_regex,
        mesh_flag=namespace.mesh_flag,
        wall_time_limit=namespace.wall_time_limit,
        watchdog_timeout=namespace.watchdog_timeout,
        kokkos_args=kokkos_args,
    )
    return Arguments(namespace.input_file, options, overrides, namespace)


# This file was made in part with generative AI.

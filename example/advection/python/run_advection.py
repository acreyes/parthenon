#!/usr/bin/env python3
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
"""Python version of the advection example.

The advection package and driver are the C++ ones of the advection-example executable,
which the advection package (built by parthenon_add_python_module) provides. This script
is the application: it sets up the problem and the main program, and accepts the same
command line as the C++ executable, e.g.,

    python run_advection.py -i parthinput.advection parthenon/time/tlim=0.5

and --pgen, which chooses between a problem generator that works on one block at a time
and one that works on all blocks of a MeshData at once. Without an input file (or restart
file), the parameters are set by advection_input(), which is the Python version of
parthinput.advection:

    python run_advection.py parthenon/time/tlim=0.5

To run in parallel, launch the script on every rank with MPI, as for the C++ executable.
MPI is initialized by ParthenonInitEnv, and each rank only works on its own blocks:

    mpirun -np 4 python run_advection.py -i parthinput.advection --pgen mesh

The build puts this script next to the advection package and creates a launcher,
<build>/example/advection/advection-example-py, that runs it with the build's Python:

    mpirun -np 4 <build>/example/advection/advection-example-py -i parthinput.advection
"""

from __future__ import annotations

import dataclasses
import sys
from typing import NamedTuple, Union

import numpy as np
import numpy.typing as npt

import advection
from advection import parthenon

ParameterValue = Union[bool, int, float, str]

# The parameters of parthinput.advection, as {block: {name: value}}. Values are converted
# to strings and parsed like values in an input file, so they are read as whatever type
# the C++ code asks for (e.g., 1 can be read as an integer or a Real).
ADVECTION_INPUT: dict[str, dict[str, ParameterValue]] = {
    "parthenon/job": {"problem_id": "advection"},
    "parthenon/mesh": {
        "refinement": "adaptive",
        "numlevel": 3,
        "nx1": 64,
        "x1min": -0.5,
        "x1max": 0.5,
        "ix1_bc": "periodic",
        "ox1_bc": "periodic",
        "nx2": 64,
        "x2min": -0.5,
        "x2max": 0.5,
        "ix2_bc": "periodic",
        "ox2_bc": "periodic",
        "nx3": 1,
        "x3min": -0.5,
        "x3max": 0.5,
        "ix3_bc": "periodic",
        "ox3_bc": "periodic",
    },
    "parthenon/meshblock": {"nx1": 16, "nx2": 16, "nx3": 1},
    "parthenon/time": {
        "nlim": -1,
        "tlim": 1.0,
        "integrator": "rk2",
        "ncycle_out_mesh": -10000,
    },
    "Advection": {
        "cfl": 0.45,
        "vx": 1.0,
        "vy": 1.0,
        "vz": 1.0,
        "profile": "hard_sphere",
        "refine_tol": 0.3,
        "derefine_tol": 0.03,
        "compute_error": False,
        "num_vars": 1,
        "vec_size": 1,
        "fill_derived": False,
    },
    "parthenon/output1": {"file_type": "rst", "dt": 0.05},
    "parthenon/output0": {
        "file_type": "hdf5",
        "dt": 0.05,
        "variables": "advected, advected_1, one_minus_advected, one_minus_advected_sq, "
        "one_minus_sqrt_one_minus_advected_sq, my_derived_var",
    },
    "parthenon/output3": {"file_type": "hst", "dt": 0.05},
    "parthenon/output4": {"file_type": "hst", "dt": 0.1, "packages": "advection_app"},
    # Soft disabled (dt < 0), as Ascent is an optional dependency
    "parthenon/output5": {
        "file_type": "ascent",
        "dt": -0.05,
        "actions_file": "custom_ascent_actions.yaml",
    },
}


def advection_input() -> parthenon.ParameterInput:
    """The parameters of parthinput.advection, set in Python."""
    pin = parthenon.ParameterInput()
    for block, parameters in ADVECTION_INPUT.items():
        for name, value in parameters.items():
            pin.add(block, name, value)
    return pin


def process_packages(pin: parthenon.ParameterInput) -> parthenon.Packages:
    packages = parthenon.Packages()
    pkg = advection.Initialize(pin)
    pkg.AddParam("initial_profile", Profile.from_package(pkg))
    packages.Add(pkg)

    app = parthenon.StateDescriptor("advection_app")
    app.PreFillDerivedBlock = advection.PreFill
    app.PostFillDerivedBlock = advection.PostFill
    packages.Add(app)
    return packages


class BlockCells(NamedTuple):
    """Cells set by the "block" profile, as index arrays: cell n is (k[n], j[n], i[n]),
    and moves with speed[n] along velocity component comp[n]."""

    k: npt.NDArray[np.intp]
    j: npt.NDArray[np.intp]
    i: npt.NDArray[np.intp]
    comp: npt.NDArray[np.intp]
    speed: npt.NDArray[np.float64]


@dataclasses.dataclass(frozen=True)
class Profile:
    """The initial profile, read from the parameters that the C++ advection package
    computes from the input. Each field is read from the parameter of the same name,
    except kind, which is the "profile" parameter."""

    kind: str = dataclasses.field(metadata={"param": "profile"})
    amp: float
    vel: float
    k_par: float
    cos_a2: float
    cos_a3: float
    sin_a2: float
    sin_a3: float
    vx: float
    vy: float
    vz: float

    @classmethod
    def from_package(cls, pkg: parthenon.StateDescriptor) -> Profile:
        return cls(
            **{
                f.name: pkg.Param(f.metadata.get("param", f.name))
                for f in dataclasses.fields(cls)
            }
        )

    def value(
        self,
        x1: npt.NDArray[np.float64],
        x2: npt.NDArray[np.float64],
        x3: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Initial value of the advected variables at positions given by
        broadcastable coordinate arrays."""
        rsq = x1**2 + x2**2 + x3**2
        if self.kind == "wave":
            x = self.cos_a2 * (x1 * self.cos_a3 + x2 * self.sin_a3) + x3 * self.sin_a2
            return 1.0 + self.amp * np.sin(self.k_par * x) * self.vel
        if self.kind == "smooth_gaussian":
            return 1.0 + self.amp * np.exp(-100.0 * rsq)
        if self.kind == "hard_sphere":
            return (rsq < 0.15**2).astype(np.float64)
        return np.zeros_like(rsq)

    def block_cells(self) -> BlockCells:
        """For the "block" profile, the cells of the first block that are set to move in
        all 6 directions."""
        return BlockCells(
            k=np.array([4, 4, 6, 6, 4, 4]),
            j=np.array([4, 6, 4, 6, 8, 10]),
            i=np.array([4, 4, 4, 6, 8, 8]),
            comp=np.array([0, 0, 1, 1, 2, 2]),
            speed=np.array(
                [self.vx, -self.vx, self.vy, -self.vy, self.vz, -self.vz],
                dtype=np.float64,
            ),
        )


def block_problem_generator(
    mb: parthenon.MeshBlock, pin: parthenon.ParameterInput
) -> None:
    """Problem generator working on one block at a time."""
    profile: Profile = mb.packages.Get("advection_package").Param("initial_profile")
    # Initialize the entire block, including ghost cells, which are overwritten by
    # boundary communication and boundary conditions before the first step. Arrays are
    # indexed [k, j, i].
    value = profile.value(
        mb.coords.Xc(1)[None, None, :],
        mb.coords.Xc(2)[None, :, None],
        mb.coords.Xc(3)[:, None, None],
    )

    # Like the C++ problem generator, initialize all independent variables
    with mb.pack(flags=[parthenon.Metadata.Independent]) as pack:
        for name in pack.names:
            pack[name][:] = value

        if profile.kind == "block" and mb.gid == 0 and "v" in pack:
            c = profile.block_cells()
            pack["advected"][0, c.k, c.j, c.i] = 10.0
            pack["v"][c.comp, c.k, c.j, c.i] = c.speed


def mesh_problem_generator(
    mesh: parthenon.Mesh, pin: parthenon.ParameterInput, md: parthenon.MeshData
) -> None:
    """Problem generator working on all blocks of a MeshData at once."""
    pkg = md.block(0).packages.Get("advection_package")
    profile: Profile = pkg.Param("initial_profile")
    # As in block_problem_generator, initialize the entire blocks. Coordinates are
    # indexed [block, index] and the value [block, k, j, i].
    value = profile.value(
        md.coords.Xc(1)[:, None, None, :],
        md.coords.Xc(2)[:, None, :, None],
        md.coords.Xc(3)[:, :, None, None],
    )

    # Like the C++ problem generator, initialize all independent variables. Entering
    # the with block copies each variable from all blocks, whose data is only contiguous
    # within each block, into one array that is contiguous over all blocks, which is
    # indexed [block, component, k, j, i]. Exiting copies the arrays back to the blocks.
    with md.pack(flags=[parthenon.Metadata.Independent]) as pack:
        for name in pack.names:
            pack[name][:] = value[:, None]

        if profile.kind == "block" and "v" in pack:
            # The rows of the block with gid 0, if it is in this MeshData, as a column
            # so that it broadcasts against the cell index arrays
            rows = np.flatnonzero(np.array([b.gid for b in md.blocks]) == 0)[:, None]
            c = profile.block_cells()
            pack["advected"][rows, 0, c.k, c.j, c.i] = 10.0
            pack["v"][rows, c.comp, c.k, c.j, c.i] = c.speed


def main(argv: list[str]) -> int:
    parser = parthenon.cli.make_parser(description=__doc__)
    parser.add_argument(
        "--pgen",
        choices=["block", "mesh"],
        default="block",
        help="use the problem generator that works on one block (block) or on all "
        "blocks of a MeshData (mesh) at a time",
    )
    args = parthenon.cli.parse_args(argv, parser, require_input=False)

    # Without an input or restart file, start from the parameters set in Python. The
    # command line overrides are applied in either case.
    use_python_input = args.input_file is None and not args.options.is_restart
    pin = args.parameter_input(advection_input() if use_python_input else None)

    pman = parthenon.ParthenonManager()
    app_input = pman.app_input
    app_input.ProcessPackages = process_packages
    if args.namespace.pgen == "mesh":
        app_input.MeshProblemGenerator = mesh_problem_generator
    else:
        app_input.ProblemGenerator = block_problem_generator
    app_input.UserWorkAfterLoop = advection.UserWorkAfterLoop
    app_input.UserMeshWorkBeforeOutput = advection.UserMeshWorkBeforeOutput
    app_input.RegisterDefaultReflectingBoundaryConditions()

    status = pman.ParthenonInitEnv(pin, args.options)
    if status != parthenon.ParthenonStatus.ok:
        pman.ParthenonFinalize()
        return 0 if status == parthenon.ParthenonStatus.complete else 1

    pman.ParthenonInitPackagesAndMesh()
    driver = advection.AdvectionDriver(pman.pinput, pman.app_input, pman.pmesh)
    driver.Execute()
    # The driver holds simulation data, so it must be destroyed before finalizing
    del driver
    pman.ParthenonFinalize()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


# This file was made in part with generative AI.

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
"""C++ components of the Parthenon advection example, as a Python library.

The advection package and the driver that evolves it are compiled into the _advection
module and re-exported here. The package also contains the parthenon subpackage, with
the Parthenon bindings and Python helpers. Both are put in this package by
parthenon_add_python_module. An application (e.g., run_advection.py) imports this
package to set up and run a problem, and is run in parallel by launching it on every rank
with MPI, e.g.,

    mpirun -np 4 python run_advection.py -i parthinput.advection
"""

from . import parthenon
from ._advection import (
    AdvectionDriver,
    Initialize,
    PostFill,
    PreFill,
    UserMeshWorkBeforeOutput,
    UserWorkAfterLoop,
)

__all__ = [
    "AdvectionDriver",
    "Initialize",
    "PostFill",
    "PreFill",
    "UserMeshWorkBeforeOutput",
    "UserWorkAfterLoop",
    "parthenon",
]

# This file was made in part with generative AI.

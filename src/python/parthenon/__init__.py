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
"""Python interface to Parthenon.

This package is not used on its own: parthenon_add_python_module (see
cmake/ParthenonPython.cmake) copies it into each application package, next to the
application's compiled module, which contains the Parthenon bindings. The generated
_core module re-exports those bindings, which this package re-exports in turn, together
with pure-Python helpers such as cli.
"""

from ._core import *  # noqa: F401,F403
from . import cli  # noqa: E402

# This file was made in part with generative AI.

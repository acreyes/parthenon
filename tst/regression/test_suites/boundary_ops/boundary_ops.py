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
# This file was made in part with generative AI.

# Modules
import re
import sys
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    """Registered boundary and flux-correction ops on a 2D forest with irregular nodes.

    Runs the boundary_ops example on 1, 2 and the test's number of ranks (all serial
    without MPI). Each run checks its ghost zones against the neighbor averages and its
    shared fluxes against the means over the blocks sharing them itself; the test then
    requires the ghost zones and the fluxes (by hash) of all runs to be bitwise
    identical.
    """

    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        if not hasattr(self, "max_ranks"):
            self.max_ranks = parameters.num_ranks
        if parameters.mpi_cmd != "":
            parameters.num_ranks = [1, 2, self.max_ranks][step - 1]
        parameters.driver_cmd_line_args = [
            "parthenon/job/problem_id=boundary_ops_%d" % step
        ]
        return parameters

    def Analyse(self, parameters):
        try:
            import h5py
            import numpy as np
        except ModuleNotFoundError:
            print("Couldn't find h5py and numpy to read the outputs.")
            return False

        hashes = []
        for stdout in parameters.stdouts:
            if b" 0 wrong ghost cells" not in stdout:
                print("A run reported wrong ghost cells.")
                return False
            if b" 0 wrong flux elements" not in stdout:
                print("A run reported wrong flux elements.")
                return False
            # The central nodes of the valence 3, 5 and 6 fans are shared by that many
            # blocks, and no other node is
            if b" 3:3 " not in stdout or b" 5:5 6:6," not in stdout:
                print("A run did not combine the irregular nodes.")
                return False
            hashes.append(re.search(rb"hash ([0-9a-f]+)", stdout).group(1))
        if len(set(hashes)) != 1:
            print("Fluxes differ between runs: %s" % hashes)
            return False

        data = []
        for step in range(1, 4):
            with h5py.File("boundary_ops_%d.out0.00000.phdf" % step, "r") as f:
                data.append(f["ghost_avg"][()])
        for step in range(1, 3):
            if not np.array_equal(data[0], data[step]):
                print("Ghost zones differ between run 1 and run %d" % (step + 1))
                return False
        return True

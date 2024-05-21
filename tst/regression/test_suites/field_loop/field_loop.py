# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

# Modules
import math
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

import sys
import os
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        assert ( 
            parameters.num_ranks <= 8
            ), "Use <= 8 ranks for field_loop test."

        # TEST: 2D field loop advection
        if step == 1:
            parameters.driver_cmd_line_args = [
                    "parthenon/job/problem_id=fieldloop2D",
                    "parthenon/mesh/nx3=1",
                    "parthenon/meshblock/nx3=1",
                    "fieldloop/tilt=false"
                    ]
        if step == 2:
            parameters.driver_cmd_line_args = [
                    "parthenon/job/problem_id=fieldloop3D",
                    "parthenon/mesh/nx1=16",
                    "parthenon/mesh/nx2=16",
                    "parthenon/mesh/nx3=16",
                    "parthenon/meshblock/nx1=8",
                    "parthenon/meshblock/nx3=8",
                    "fieldloop/tilt=true"
                    ]

        return parameters

    def Analyse(self, parameters):
        data = np.loadtxt("fieldloop2D.out2.hst", comments="#")
        max_divTot = np.max(data[:,2])
        max_divMax = np.max(data[:,3])
        print(f"Max average divergence 2D: {max_divTot}\nMax divergence 2D: {max_divMax}")
        if max(max_divTot, max_divMax) > 1.e-14:
            print(f"TEST FAIL: Divergence 2D exceeds 1.e-14")
            return False

        data = np.loadtxt("fieldloop3D.out2.hst", comments="#")
        max_divTot = np.max(data[:,2])
        max_divMax = np.max(data[:,3])
        print(f"Max average divergence 3D: {max_divTot}\nMax divergence 3D: {max_divMax}")
        if max(max_divTot, max_divMax) > 1.e-14:
            print(f"TEST FAIL: Divergence 3D exceeds 1.e-14")
            return False
        return True


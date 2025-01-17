# ========================================================================================
# Parthenon performance portable AMR framework
# Copyright(C) 2021 The Parthenon collaboration
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
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
import sys
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        parameters.coverage_status = "both"

        if parameters.sparse_disabled:
            parameters.driver_cmd_line_args = [
                "parthenon/sparse/enable_sparse=false",
            ]

        # Run a test with two trees
        if step == 2:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx2=32",
                "parthenon/job/problem_id=sparse_twotree",
            ]

        # Run a test with two trees and a statically refined region
        if step == 3:
            parameters.driver_cmd_line_args = [
                "parthenon/mesh/nx2=32",
                "parthenon/time/nlim=50",
                "parthenon/job/problem_id=sparse_twotree_static",
                "parthenon/mesh/refinement=static",
                "parthenon/static_refinement0/x1min=-0.75",
                "parthenon/static_refinement0/x1max=-0.5",
                "parthenon/static_refinement0/x2min=-0.75",
                "parthenon/static_refinement0/x2max=-0.5",
                "parthenon/static_refinement0/level=3",
            ]

        return parameters

    def Analyse(self, parameters):

        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            from phdf_diff import compare
        except ModuleNotFoundError:
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        # compare against fake sparse version, needs to match up to tolerance used for sparse allocation
        delta = compare(
            [
                "sparse.out0.final.phdf",
                parameters.parthenon_path
                + "/tst/regression/gold_standard/sparse_fake.out0.final.phdf",
            ],
            one=True,
            tol=2e-6,
            # don't check metadata, because SparseInfo will differ
            check_metadata=False,
        )

        if delta != 0:
            return False

        if not parameters.sparse_disabled:
            # compare against true sparse, needs to match to machine precision
            delta = compare(
                [
                    "sparse.out0.final.phdf",
                    parameters.parthenon_path
                    + "/tst/regression/gold_standard/sparse_true.out0.final.phdf",
                ],
                one=True,
                tol=1e-12,
                check_metadata=False,
            )
            if delta != 0:
                print("Sparse advection failed for standard AMR grid setup.")
                return False

            delta = compare(
                [
                    "sparse_twotree.out0.final.phdf",
                    parameters.parthenon_path
                    + "/tst/regression/gold_standard/sparse_twotree.out0.final.phdf",
                ],
                one=True,
                tol=1e-12,
                check_metadata=False,
            )
            if delta != 0:
                print("Sparse advection failed for two-tree AMR grid setup.")
                return False

            delta = compare(
                [
                    "sparse_twotree_static.out0.final.phdf",
                    parameters.parthenon_path
                    + "/tst/regression/gold_standard/sparse_twotree_static.out0.final.phdf",
                ],
                one=True,
                tol=1e-12,
                check_metadata=False,
            )
            if delta != 0:
                print("Sparse advection failed for two-tree SMR grid setup.")

        return delta == 0

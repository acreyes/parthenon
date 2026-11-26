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

sys.dont_write_bytecode = True

# Two sets of runs; within a set every cycle must give the same hash
SET_A = ["provider_lanes/pair_change=true", "provider_lanes/ncycles=6"]
SET_B = [
    "provider_lanes/rows_min=0",
    "provider_lanes/rows_vary=true",
    "provider_lanes/rebuilds=2",
    "provider_lanes/ncycles=5",
]
SET_C = [
    "provider_lanes/pair_change=true",
    "provider_lanes/rebuild_cycle=-1",
    "provider_lanes/rows_fixed=true",
    "provider_lanes/rows_min=3",
    "provider_lanes/ncycles=6",
]
SET_D = [
    "provider_lanes/idle_rank=1",
    "provider_lanes/rebuild_cycle=-1",
    "provider_lanes/ncycles=4",
]
SETS = {"A": SET_A, "B": SET_B, "C": SET_C, "D": SET_D}
# (set, ranks (0 for the test's number), coalesced comms, extra arguments)
RUNS = [
    ("A", 2, False, []),
    ("A", 0, False, []),
    ("A", 0, True, []),
    ("A", 2, True, []),
    ("A", 3, True, ["provider_lanes/full_rebuild=true"]),
    ("B", 1, False, []),
    ("B", 3, False, []),
    ("B", 3, True, []),
    ("B", 2, True, []),
    ("C", 2, False, []),
    ("C", 2, True, []),
    ("C", 0, True, []),
    ("D", 3, False, []),
    ("D", 3, True, []),
]


class TestCase(utils.test_case.TestCaseAbs):
    """Provider lanes with and without coalesced comms.

    Runs the provider_lanes example (record, variable-length rows and flux combine
    exchanges on two lanes, the groups lane rebuilt mid-run) on 1 to 4 ranks with
    coalesced comms off and on. Set A also changes the rows capacity of one pair of
    blocks on two ranks, so with coalesced comms exactly that pair's two lane infos are
    sent; set B has empty payloads, per-pair capacities and two rebuilds before one
    exchange. Set C grows that pair's capacity three times with the same payload, so
    its buffers move to new pools, whose keys repeat the old pool's. In set D rank 1
    sits out one exchange of the groups lane without rebuilding it while the others
    drop it and rebuild, then they take it back and rebuild while rank 1 still does
    not. Each run checks its exchanged values, its boundary cache initializations
    and, with coalesced comms, its lane info messages itself; the test then requires the
    per-cycle hashes of all runs of a set to be identical.
    """

    def Prepare(self, parameters, step):
        parameters.coverage_status = "both"
        if not hasattr(self, "max_ranks"):
            self.max_ranks = parameters.num_ranks
        lane_set, ranks, coalesced, extra = RUNS[step - 1]
        if parameters.mpi_cmd != "":
            parameters.num_ranks = ranks if ranks > 0 else self.max_ranks
        parameters.driver_cmd_line_args = (
            ["parthenon/job/problem_id=provider_lanes_%d" % step]
            + SETS[lane_set]
            + [
                "parthenon/mesh/do_coalesced_comms=%s"
                % ("true" if coalesced else "false")
            ]
            + extra
        )
        return parameters

    def Analyse(self, parameters):
        hashes = {k: set() for k in SETS}
        for (lane_set, ranks, coalesced, extra), stdout in zip(
            RUNS, parameters.stdouts
        ):
            lines = [
                l for l in stdout.split(b"\n") if l.startswith(b"provider_lanes: cycle")
            ]
            if len(lines) == 0:
                print("A run printed no cycles.")
                return False
            for line in lines:
                if b" 0 wrong" not in line or b"WRONG" in line:
                    print("A run failed its checks: %s" % line.decode())
                    return False
                if coalesced != (b"lane infos" in line):
                    print("Lane infos missing or unexpected: %s" % line.decode())
                    return False
            hashes[lane_set].add(
                tuple(re.search(rb"hash ([0-9a-f]+)", l).group(1) for l in lines)
            )
        for lane_set, h in hashes.items():
            if len(h) != 1:
                print("Hashes differ between the runs of set %s: %s" % (lane_set, h))
                return False
        return True

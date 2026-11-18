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
// This file was made in part with generative AI.
#ifndef EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_
#define EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_

#include <cstdint>
#include <memory>

#include <parthenon/package.hpp>

namespace boundary_ops {
using namespace parthenon::package::prelude;

// Every block fills its interior with this value of its gid. The values are not exactly
// representable, so a sum over several neighbors depends on the summation order.
KOKKOS_INLINE_FUNCTION Real BlockValue(int gid) { return 1.0 / (gid + 3.0); }

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
TaskStatus SetBlockValues(MeshData<Real> *md);
// Allocates the sparse variable on every block (a no-op without sparse support).
void AllocateSparse(Mesh *pmesh);
// Checks every ghost region of every local block, for the dense and the sparse variable,
// against the mean of its neighbors' values. Returns the global number of wrong cells;
// reports the global number of regions that averaged two or more neighbors in
// naveraged.
int CheckGhosts(Mesh *pmesh, int *naveraged);

// Variable-length payloads: the pack op of "varlen" fills and sends a cycle- and
// neighbor-dependent number of leading buffer elements, the unpack op checks the
// received length and values.
KOKKOS_INLINE_FUNCTION int VarlenLength(int send_gid, int recv_gid, int cycle,
                                        int capacity) {
  return 1 + (7 * send_gid + 3 * recv_gid + 5 * cycle) % capacity;
}
KOKKOS_INLINE_FUNCTION Real VarlenValue(int send_gid, int recv_gid, int m, int cycle) {
  return send_gid + 1.0e3 * recv_gid + 1.0e6 * m + 0.5 * cycle;
}
struct VarlenStats {
  std::int64_t messages = 0, sent = 0, capacity = 0, wrong = 0;
};
void SetCycle(Mesh *pmesh, int cycle);
VarlenStats GetVarlenStats();

} // namespace boundary_ops

#endif // EXAMPLE_BOUNDARY_OPS_BOUNDARY_OPS_HPP_

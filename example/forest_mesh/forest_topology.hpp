//========================================================================================
// (C) (or copyright) 2023. Triad National Security, LLC. All rights reserved.
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
#ifndef EXAMPLE_FOREST_MESH_FOREST_TOPOLOGY_HPP_
#define EXAMPLE_FOREST_MESH_FOREST_TOPOLOGY_HPP_

#include <array>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "basic_types.hpp"
#include "defs.hpp"
#include "mesh/forest/forest.hpp"
#include "mesh/forest/logical_location.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

constexpr int NDIM = 2;
template <class T, int SIZE>
using sptr_vec_t = std::array<std::shared_ptr<T>, SIZE>;

struct EdgeLoc {
  Direction dir;
  bool lower;

  // In 2D we can ignore connectivity of K-direction faces,
  CellCentOffsets GetFaceIdx2D() const {
    if (dir == Direction::I && lower) return CellCentOffsets(0, -1, 0);
    if (dir == Direction::I && !lower) return CellCentOffsets(0, 1, 0);
    if (dir == Direction::J && lower) return CellCentOffsets(-1, 0, 0);
    if (dir == Direction::J && !lower) return CellCentOffsets(1, 0, 0);
    return CellCentOffsets(0, 0, 0);
    // return (1 - 2 * lower) * std::pow(3, (static_cast<uint>(dir) + 1) % 2) + 1 + 3 + 9;
  }

  static const EdgeLoc South;
  static const EdgeLoc North;
  static const EdgeLoc West;
  static const EdgeLoc East;
};
inline const EdgeLoc EdgeLoc::South{Direction::I, true};
inline const EdgeLoc EdgeLoc::North{Direction::I, false};
inline const EdgeLoc EdgeLoc::West{Direction::J, true};
inline const EdgeLoc EdgeLoc::East{Direction::J, false};
inline bool operator==(const EdgeLoc &lhs, const EdgeLoc &rhs) {
  return (lhs.dir == rhs.dir) && (lhs.lower == rhs.lower);
}

LogicalCoordinateTransformation LogicalCoordinateTransformationFromSharedEdge2D(EdgeLoc origin, EdgeLoc neighbor,
                                                        int orientation) {
  if (origin.dir == Direction::K || neighbor.dir == Direction::K) {
    PARTHENON_FAIL("In 2D we shouldn't have explicit edges in the Z direction.");
  }

  LogicalCoordinateTransformation out;
  out.dir_connection[static_cast<uint>(origin.dir)] = static_cast<uint>(neighbor.dir);
  out.dir_flip[static_cast<uint>(origin.dir)] = orientation == -1;
  out.dir_connection[(static_cast<uint>(origin.dir) + 1) % 2] =
      (static_cast<uint>(neighbor.dir) + 1) % 2;
  out.dir_flip[(static_cast<uint>(origin.dir) + 1) % 2] =
      (neighbor.lower == origin.lower);
  return out;
}

} // namespace forest
} // namespace parthenon

template <>
class std::hash<parthenon::forest::EdgeLoc> {
 public:
  std::size_t operator()(const parthenon::forest::EdgeLoc &key) const noexcept {
    return 2 * static_cast<uint>(key.dir) + key.lower;
  }
};

template <>
class std::hash<parthenon::CellCentOffsets> {
 public:
  std::size_t operator()(const parthenon::CellCentOffsets &key) const noexcept {
    return key.GetIdx();
  }
};

namespace parthenon {
namespace forest {

class Face;
class Node {
 public:
  Node(int id_in, std::array<Real, NDIM> pos) : id(id_in), x(pos) {}

  static std::shared_ptr<Node> create(int id, std::array<Real, NDIM> pos) {
    return std::make_shared<Node>(id, pos);
  }

  std::uint32_t id;
  std::array<Real, NDIM> x;
  std::unordered_set<std::shared_ptr<Face>> associated_faces;
};

class Edge {
 public:
  Edge() = default;
  explicit Edge(sptr_vec_t<Node, 2> nodes_in) : nodes(nodes_in) {}

  Edge(sptr_vec_t<Node, 2> nodes_in, const CellCentOffsets &ploc)
      : nodes{nodes_in}, loc{ploc} {
    PARTHENON_REQUIRE(loc->IsEdge(), "Trying to pass a non-edge location to an edge.");
    auto dirs = loc->GetTangentDirections();
    dir = dirs[0];
    auto ndirs = loc->GetNormals();
    normals = {ndirs[0], ndirs[1]};
  }

  sptr_vec_t<Node, 2> nodes;
  Direction dir;
  std::array<std::pair<Direction, Offset>, 2> normals;
  std::optional<CellCentOffsets> loc{};

  int LogicalCoordinateTransformation(const Edge &e2) const {
    if (nodes[0] == e2.nodes[0] && nodes[1] == e2.nodes[1]) {
      return 1;
    } else if (nodes[0] == e2.nodes[1] && nodes[1] == e2.nodes[0]) {
      return -1;
    } else {
      return 0;
    }
  }
};

class Face : public std::enable_shared_from_this<Face> {
 private:
  struct Private_t {};

 public:
  Face() : tree() {}

  // Constructor that can only be called internally
  Face(std::int64_t id, sptr_vec_t<Node, 4> nodes_in, Private_t)
      : nodes(nodes_in), tree(Tree::create(id, NDIM, 0)), dir{Direction::I, Direction::J},
        normal{Direction::K}, normal_rhanded(true) {
    edges[EdgeLoc::South] = Edge({nodes[0], nodes[1]});
    edges[EdgeLoc::West] = Edge({nodes[0], nodes[2]});
    edges[EdgeLoc::East] = Edge({nodes[1], nodes[3]});
    edges[EdgeLoc::North] = Edge({nodes[2], nodes[3]});
  }

  static std::shared_ptr<Face> create(std::int64_t id, sptr_vec_t<Node, 4> nodes_in) {
    auto result = std::make_shared<Face>(id, nodes_in, Private_t());
    // Associate the new face with the nodes
    for (auto &node : result->nodes)
      node->associated_faces.insert(result);
    return result;
  }

  std::shared_ptr<Face> getptr() { return shared_from_this(); }

  Direction dir[2];
  Direction normal;
  bool normal_rhanded;

  sptr_vec_t<Node, 4> nodes;
  std::unordered_map<EdgeLoc, Edge> edges;
  std::shared_ptr<Tree> tree;
};

// We choose face nodes to be ordered as:
//
//   2---3
//   |   |
//   0---1
//
// with the X0 direction pointing from 0->1 and the X1 direction pointing from 0->2
// the permutations of nodes below correspond to the same topological face but different
// choices for the X0 and X1 directions. Even though there are 24 possible permutations
// of 4 nodes, only 8 of those permutations give the same shape (i.e. same set of edges).
// Two separate macrocells that share a face can give different coordinate orientations
// to the face. The easiest way to do this is just write this as a lookup table.
inline const std::array<std::array<int, 7>, 8> allowed_face_node_permutations{
    // First four elements define permutation,
    // fifth and sixth define axis flippety-floppety with X0 = 1 and X2 = 2
    // seventh denotes if parity transformation occurred
    // clockwise 90 deg rotations
    std::array<int, 7>{0, 1, 2, 3, 1, 2, 0},   // X0 ->  X0, X1 ->  X1
    std::array<int, 7>{1, 3, 0, 2, 2, -1, 0},  // X0 ->  X1, X1 -> -X0
    std::array<int, 7>{3, 2, 1, 0, -1, -2, 0}, // X0 -> -X0, X1 -> -X1
    std::array<int, 7>{2, 0, 3, 1, -2, 1, 0},  // X0 -> -X1, X1 ->  X0
    // Parity about X0 and clockwise 90 deg rotations
    std::array<int, 7>{1, 0, 3, 2, -1, 2, 1}, // X0 -> -X0, X1 ->  X1
    std::array<int, 7>{0, 2, 1, 3, 2, 1, 1},  // X0 ->  X1, X1 ->  X0
    std::array<int, 7>{2, 3, 0, 1, 1, -2, 1}, // X0 ->  X0, X1 -> -X1
    std::array<int, 7>{3, 1, 2, 0, -2, -1, 1} // X0 -> -X1, X1 -> -X0
};

inline std::optional<LogicalCoordinateTransformation> CompareFaces(const Face *f1, const Face *f2) {
  for (auto &perm : allowed_face_node_permutations) {
    if (f1->nodes[0] == f2->nodes[perm[0]] && f1->nodes[1] == f2->nodes[perm[1]] &&
        f1->nodes[2] == f2->nodes[perm[2]] && f1->nodes[3] == f2->nodes[perm[3]]) {
      LogicalCoordinateTransformation orient;
      orient.SetDirection(f1->dir[0], f2->dir[abs(perm[4]) - 1], perm[4] < 0);
      orient.SetDirection(f1->dir[1], f2->dir[abs(perm[5]) - 1], perm[5] < 0);

      // Set the out of plane coordinate
      bool flip = f1->normal_rhanded == f2->normal_rhanded;
      if (perm[6]) flip = !flip; // Parity in the face is flipped
      orient.SetDirection(f1->normal, f2->normal, flip);
      return orient;
    }
  }
  return {}; // These are not the same face
}

inline std::optional<LogicalCoordinateTransformation> CompareEdges(const Edge *e1, const Edge *e2) {
  if (e1->nodes[0] == e2->nodes[0] && e1->nodes[1] == e2->nodes[1]) {
    LogicalCoordinateTransformation orient;
    orient.SetDirection(e1->dir, e2->dir, false);

    // Set the out of edge coordinates, these are paired arbitrarily
    orient.SetDirection(e1->normals[0].first, e2->normals[0].first,
                        static_cast<int>(e1->normals[0].second) +
                            static_cast<int>(e2->normals[0].second));
    orient.SetDirection(e1->normals[1].first, e2->normals[1].first,
                        static_cast<int>(e1->normals[1].second) +
                            static_cast<int>(e2->normals[1].second));
  } else if (e1->nodes[0] == e2->nodes[1] && e1->nodes[1] == e2->nodes[0]) {
    LogicalCoordinateTransformation orient;
    orient.SetDirection(e1->dir, e2->dir, true);

    // Set the out of edge coordinates, these are paired arbitrarily
    orient.SetDirection(e1->normals[0].first, e2->normals[1].first,
                        static_cast<int>(e1->normals[0].second) +
                            static_cast<int>(e2->normals[1].second));
    orient.SetDirection(e1->normals[1].first, e2->normals[0].first,
                        static_cast<int>(e1->normals[1].second) +
                            static_cast<int>(e2->normals[0].second));
  }
  return {}; // These are not the same edge
}

inline void ListFaces(const std::shared_ptr<Node> &node) {
  for (auto &face : node->associated_faces) {
    printf("{%i, %i, %i, %i}\n", face->nodes[0]->id, face->nodes[1]->id,
           face->nodes[2]->id, face->nodes[3]->id);
  }
}

using NeighborDesc = std::tuple<std::shared_ptr<Face>, EdgeLoc, int>;
inline std::vector<NeighborDesc> FindEdgeNeighbors(const std::shared_ptr<Face> &face_in,
                                                   EdgeLoc loc) {
  std::vector<NeighborDesc> neighbors;
  auto edge = face_in->edges[loc];

  std::unordered_set<std::shared_ptr<Face>> possible_neighbors;
  for (auto &node : edge.nodes)
    possible_neighbors.insert(node->associated_faces.begin(),
                              node->associated_faces.end());

  // Check each neighbor to see if it shares an edge
  for (auto &neigh : possible_neighbors) {
    if (neigh != face_in) {
      for (auto &[neigh_loc, neigh_edge] : neigh->edges) {
        int orientation = edge.LogicalCoordinateTransformation(neigh_edge);
        if (orientation)
          neighbors.push_back(std::make_tuple(neigh, neigh_loc, orientation));
      }
    }
  }
  return neighbors;
}
} // namespace forest
} // namespace parthenon

#endif // EXAMPLE_FOREST_MESH_FOREST_TOPOLOGY_HPP_

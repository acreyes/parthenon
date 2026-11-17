//========================================================================================
// (C) (or copyright) 2024. Triad National Security, LLC. All rights reserved.
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

#include <algorithm>
#include <array>
#include <iterator>
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
#include "mesh/forest/forest_node.hpp"
#include "mesh/forest/forest_topology.hpp"
#include "mesh/forest/logical_location.hpp"
#include "mesh/forest/tree.hpp"
#include "utils/bit_hacks.hpp"
#include "utils/indexer.hpp"

namespace parthenon {
namespace forest {

template <class T1, class T2>
std::vector<std::shared_ptr<Node>> NodeListOverlap(T1 nodes_1, T2 nodes_2) {
  std::sort(std::begin(nodes_1), std::end(nodes_1));
  std::sort(std::begin(nodes_2), std::end(nodes_2));
  std::vector<std::shared_ptr<Node>> node_intersection;
  std::set_intersection(std::begin(nodes_1), std::end(nodes_1), std::begin(nodes_2),
                        std::end(nodes_2), std::back_inserter(node_intersection));
  return node_intersection;
}

std::optional<CellCentOffsets> Face::IsEdge(const Edge &edge) {
  auto node_overlap = NodeListOverlap(nodes, edge.nodes);
  if (node_overlap.size() != 2) return {};
  auto offsets = AverageOffsets(node_to_offset[face_index[node_overlap[0]]],
                                node_to_offset[face_index[node_overlap[1]]]);
  if (!offsets.IsEdge()) return {};
  return offsets;
}

void Face::SetNeighbors() {
  std::unordered_set<std::weak_ptr<Face>, WeakPtrHash<Face>, WeakPtrEqual<Face>>
      neighbors_local;
  for (auto &node : nodes)
    neighbors_local.insert(node->associated_faces.begin(), node->associated_faces.end());
  for (std::weak_ptr<Face> w_neighbor : neighbors_local) {
    PARTHENON_REQUIRE(!w_neighbor.expired(), "Invalid weak pointer to face.");
    auto neighbor = w_neighbor.lock();
    auto node_overlap = NodeListOverlap(nodes, neighbor->nodes);
    if (node_overlap.size() > 0 && node_overlap.size() < 4) {
      std::array<int, 2> offset{0, 0};
      for (int i = 0; i < 4; ++i) {
        if (std::find(node_overlap.begin(), node_overlap.end(), nodes[i]) !=
            node_overlap.end()) {
          for (int o = 0; o < 2; ++o)
            offset[o] += static_cast<int>(node_to_offset[i][o]);
        }
      }
      for (auto &o : offset)
        o /= node_overlap.size();
      neighbors(offset[0], offset[1])
          .push_back(std::make_pair(neighbor, LogicalCoordinateTransformation()));
      neighbors_to_offsets[neighbor] = CellCentOffsets(offset[0], offset[1], -1);
    }
  }
}

std::tuple<int, int, Offset>
Face::GetEdgeDirections(const std::vector<std::shared_ptr<Node>> &nodes) {
  PARTHENON_REQUIRE(nodes.size() == 2, "The argument can't be an edge.");
  int I0 = face_index[nodes[0]];
  int I1 = face_index[nodes[1]];
  int dir_tang = (I1 - I0 > 0 ? 1 : -1) * (IntegerLog2Floor(std::abs(I1 - I0)) + 1);
  auto offsets = AverageOffsets(node_to_offset[I0], node_to_offset[I1]);
  PARTHENON_REQUIRE(offsets.IsEdge(), "Something is wrong.");

  CoordinateDirection dir_norm;
  Offset offset;
  if (std::abs(static_cast<int>(offsets[0])) == 1) {
    dir_norm = X1DIR;
    offset = offsets[0];
  } else if (std::abs(static_cast<int>(offsets[1])) == 1) {
    dir_norm = X2DIR;
    offset = offsets[1];
  } else {
    PARTHENON_FAIL("Shouldn't get here.");
  }
  return std::make_tuple(dir_tang, static_cast<int>(dir_norm), offset);
}

void Face::SetEdgeCoordinateTransforms() {
  for (int ox = -1; ox <= 1; ++ox) {
    for (int oy = -1; oy <= 1; ++oy) {
      if (std::abs(ox) + std::abs(oy) == 1) {
        for (auto &[w_neighbor, coord_trans] : neighbors(ox, oy)) {
          PARTHENON_REQUIRE(!w_neighbor.expired(), "Invalid weak pointer to face.");
          auto neighbor = w_neighbor.lock();
          auto node_overlap = NodeListOverlap(nodes, neighbor->nodes);
          PARTHENON_REQUIRE(node_overlap.size() == 2, "This is clearly not an edge.");
          std::sort(node_overlap.begin(), node_overlap.end(), [this](auto &n1, auto &n2) {
            return this->face_index[n1] < this->face_index[n2];
          });

          auto [dir1, dir2, offset] = GetEdgeDirections(node_overlap);
          auto [dir1_neigh, dir2_neigh, offset_neigh] =
              neighbor->GetEdgeDirections(node_overlap);

          LogicalCoordinateTransformation ct;
          // Set the direction mapping for direction along the edge
          ct.SetDirection(static_cast<CoordinateDirection>(std::abs(dir1)),
                          static_cast<CoordinateDirection>(std::abs(dir1_neigh)),
                          dir1_neigh < 0);
          // Set the direction mapping for direction tangent to the edge
          ct.SetDirection(static_cast<CoordinateDirection>(std::abs(dir2)),
                          static_cast<CoordinateDirection>(std::abs(dir2_neigh)),
                          offset == offset_neigh);
          ct.offset = AverageOffsets(node_to_offset[face_index[node_overlap[0]]],
                                     node_to_offset[face_index[node_overlap[1]]]);
          // Remove x3 offset since this is a 2D mesh
          ct.offset[2] = 0;
          ct.use_offset = true;
          coord_trans = ct;
        }
      }
    }
  }
}

void Face::SetNodeCoordinateTransforms() {
  // A node neighbor shares only the node P with this face. Walk the faces around P
  // through shared edges, this = F_0, F_1, ..., F_k = neighbor. In each face, call s+ the
  // direction along its edge with the next face of the walk and s- the other direction.
  // The transform maps s+ to s+ and s- to s-, both reversed through P. For k == 2 this
  // equals composing the two edge transforms; composing k > 2 edge transforms around an
  // irregular node does not map this face's corner at P onto the neighbor's corner at P.
  auto along_edge_dir = [](const CellCentOffsets &edge_offset) {
    return static_cast<int>(edge_offset[0]) == 0 ? 0 : 1;
  };
  Indexer2D offset_idxer({-1, 1}, {-1, 1});
  for (int n = 0; n < offset_idxer.size(); ++n) {
    auto [ox1, ox2] = offset_idxer(n);
    if (std::abs(ox1) + std::abs(ox2) == 2) {
      for (auto &[w_neighbor, coord_trans] : neighbors(ox1, ox2)) {
        PARTHENON_REQUIRE(!w_neighbor.expired(), "Invalid weak pointer to face.");
        auto neighbor = w_neighbor.lock();
        auto node_overlap = NodeListOverlap(nodes, neighbor->nodes);
        PARTHENON_REQUIRE(node_overlap.size() == 1,
                          "Must only have a single node overlap for node neighbors.");
        auto node = node_overlap[0];

        // Breadth first search from this face to the neighbor through edges at node
        std::unordered_map<Face *, Face *> previous{{this, nullptr}};
        std::vector<Face *> queue{this};
        for (std::size_t q = 0; q < queue.size() && !previous.count(neighbor.get());
             ++q) {
          for (auto &[w_face, off] : queue[q]->neighbors_to_offsets) {
            auto face = w_face.lock();
            if (face && off.IsEdge() && face->face_index.count(node) &&
                !previous.count(face.get())) {
              previous[face.get()] = queue[q];
              queue.push_back(face.get());
            }
          }
        }
        PARTHENON_REQUIRE(previous.count(neighbor.get()),
                          "Node neighbor is not connected through edges at the node.");
        Face *last = previous[neighbor.get()];
        Face *first = neighbor.get();
        while (previous[first] != this)
          first = previous[first];

        // s+ in this face and s- in the neighbor, the other direction in each is the
        // remaining one
        const int splus = along_edge_dir(neighbors_to_offsets[first->getptr()]);
        const int sminus_neigh =
            along_edge_dir(neighbor->neighbors_to_offsets[last->getptr()]);
        const auto &corner = node_to_offset[face_index[node]];
        const auto &corner_neigh = neighbor->node_to_offset[neighbor->face_index[node]];

        LogicalCoordinateTransformation ct;
        for (const auto &[d, dn] : {std::make_pair(splus, 1 - sminus_neigh),
                                    std::make_pair(1 - splus, sminus_neigh)}) {
          ct.SetDirection(static_cast<CoordinateDirection>(d + 1),
                          static_cast<CoordinateDirection>(dn + 1),
                          corner[d] == corner_neigh[dn]);
        }
        ct.offset = {ox1, ox2, 0};
        ct.use_offset = true;
        coord_trans = ct;
      }
    }
  }
}

} // namespace forest
} // namespace parthenon

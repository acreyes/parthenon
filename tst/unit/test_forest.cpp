//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2024-2026 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021-2024. Triad National Security, LLC. All rights reserved.
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
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch.hpp>

#include "kokkos_abstraction.hpp"
#include "mesh/forest/forest.hpp"

using namespace parthenon::forest;
using parthenon::Real;
namespace {
void DerefineAllPossibleLocations(Forest &forest) {
  auto locs = forest.GetMeshBlockListAndResolveGids();
  std::vector<parthenon::LogicalLocation> deref_locs;
  for (const auto &l : locs) {
    auto parent = l.GetParent();
    const int ndim = 2;
    int ndaught{0};
    for (const auto &d : parent.GetDaughters(ndim)) {
      ndaught += forest.count(d);
    }
    if (ndaught == std::pow(2, ndim)) deref_locs.push_back(parent);
  }
  std::sort(deref_locs.begin(), deref_locs.end(),
            [](auto &l, auto &r) { return l.level() > r.level(); });
  for (const auto &d : deref_locs) {
    forest.Derefine(d);
  }
}

// Create an n-tree forest with one internal point of valence n and refine one tree
// twice at a location next to the n-valence point. This should result in a forest with
// 10 + 7 * (n - 1) blocks after properly nested refinement. The logical coordinate
// transformations between the produced trees are nontrivial. If nblocks_min !=
// nblocks_max, this creates a forest that is the combination of disconnected n-tree
// forests with n between nblocks_min and nblocks_max
Forest n_blocks(int nblocks_min, int nblocks_max) {
  std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
  ForestDefinition forest_def;
  int nc = 0;
  int fc = 0;
  Real xoffset = 0.0;
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    for (int point = 0; point < 2 * nblocks; ++point) {
      nodes[nc + point] = Node::create(
          nc + point, {static_cast<Real>(std::sin(point * M_PI / nblocks) + xoffset),
                       static_cast<Real>(std::cos(point * M_PI / nblocks))});
    }
    nodes[nc + 2 * nblocks] =
        Node::create(nc + 2 * nblocks, {static_cast<Real>(0.0 + xoffset), 0.0});
    auto &n = nodes;
    for (int t = 0; t < nblocks; ++t)
      forest_def.AddFace(fc + t,
                         {n[nc + 2 * t + 1], n[nc + 2 * t],
                          n[nc + (2 * t + 2) % (2 * nblocks)], n[nc + 2 * nblocks]});
    nc += 2 * nblocks + 1;
    fc += nblocks;
    xoffset += 2.2;
  }
  auto forest = Forest::Make2D(forest_def);

  // Do some refinements that should propagate into all trees
  fc = 0;
  for (int nblocks = nblocks_min; nblocks <= nblocks_max; ++nblocks) {
    forest.Refine(parthenon::LogicalLocation(fc + 2, 0, 0, 0, 0));
    forest.Refine(parthenon::LogicalLocation(fc + 2, 1, 1, 1, 0));
    forest.Refine(parthenon::LogicalLocation(fc + 2, 2, 3, 3, 0));
    fc += nblocks;
  }

  return forest;
}
} // namespace

TEST_CASE("Simple forest construction", "[forest]") {
  // Create two trees in two dimensions that both have a single block
  auto tree1 = Tree::create(1, 2, 0);
  auto tree2 = Tree::create(2, 2, 0);

  // Periodic connectivity to self
  for (int offy : {-1, 1}) {
    tree1->AddNeighborTree(parthenon::CellCentOffsets(0, offy, 0), tree1,
                           LogicalCoordinateTransformation(), true);
    tree2->AddNeighborTree(parthenon::CellCentOffsets(0, offy, 0), tree2,
                           LogicalCoordinateTransformation(), true);
  }
  // Connectivity to the other tree (both periodic and internal)
  for (int offy : {-1, 0, 1}) {
    for (int offx : {-1, 1}) {
      bool tree1_p;
      bool tree2_p;
      if (offx == 1) {
        tree1_p = false;
        tree2_p = true;
      } else {
        tree1_p = true;
        tree2_p = false;
      }
      tree1->AddNeighborTree(parthenon::CellCentOffsets(offx, offy, 0), tree2,
                             LogicalCoordinateTransformation(), tree1_p);
      tree2->AddNeighborTree(parthenon::CellCentOffsets(offx, offy, 0), tree1,
                             LogicalCoordinateTransformation(), tree2_p);
    }
  }

  // Create a forest from the two trees
  Forest forest;
  forest.AddTree(tree1);
  forest.AddTree(tree2);

  GIVEN("A periodic forest with two trees") {
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 2);

    // Refine the lower left block repeatedly and check that there are the correct
    // number of blocks after each refinement when things are properly nested. Numbers
    // were determined by hand.
    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 5);

    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 11);

    forest.Refine(locs[0]);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 23);

    // Now flag all blocks for refinement and derefine employing
    // proper nesting. Should just be reverse of previous refinement
    // operations.
    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 11);

    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 5);

    DerefineAllPossibleLocations(forest);
    locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 2);
  }
}

TEST_CASE("Singular forest construction", "[forest]") {
  GIVEN("A forest with three trees and three-valent point") {
    auto forest = n_blocks(3, 3);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 24);
  }
  GIVEN("A forest with four trees and four-valent point") {
    auto forest = n_blocks(4, 4);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 31);
  }
  GIVEN("A forest with five trees and five-valent point") {
    auto forest = n_blocks(5, 5);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 38);
  }
  GIVEN("A forest with three-, four-, and five-valent points") {
    auto forest = n_blocks(3, 5);
    auto locs = forest.GetMeshBlockListAndResolveGids();
    REQUIRE(locs.size() == 93);
  }
}

namespace {
// Build the faces of an n-valent fan around a single interior node, with each face's
// nodes listed in one of the eight orderings that describe the same quadrilateral, and
// set their neighbors and coordinate transformations as Forest::Make2D does
std::vector<std::shared_ptr<Face>> FanFaces(int valence, int orientation_shift) {
  const std::array<std::array<int, 4>, 8> perms{std::array<int, 4>{0, 1, 2, 3},
                                                {1, 3, 0, 2},
                                                {3, 2, 1, 0},
                                                {2, 0, 3, 1},
                                                {1, 0, 3, 2},
                                                {0, 2, 1, 3},
                                                {2, 3, 0, 1},
                                                {3, 1, 2, 0}};
  std::vector<std::shared_ptr<Node>> n;
  for (int p = 0; p < 2 * valence; ++p)
    n.push_back(
        Node::create(p, {std::sin(p * M_PI / valence), std::cos(p * M_PI / valence)}));
  n.push_back(Node::create(2 * valence, {0.0, 0.0}));
  std::vector<std::shared_ptr<Face>> faces;
  for (int t = 0; t < valence; ++t) {
    const std::array<std::shared_ptr<Node>, 4> canonical{
        n[2 * t + 1], n[2 * t], n[(2 * t + 2) % (2 * valence)], n[2 * valence]};
    const auto &perm = perms[(t * 3 + orientation_shift) % 8];
    faces.push_back(Face::create(t, {canonical[perm[0]], canonical[perm[1]],
                                     canonical[perm[2]], canonical[perm[3]]}));
  }
  for (auto &f : faces) {
    f->SetNeighbors();
    f->SetEdgeCoordinateTransforms();
  }
  for (auto &f : faces)
    f->SetNodeCoordinateTransforms();
  return faces;
}

std::array<int, 3> Arr(const parthenon::CellCentOffsets &o) {
  return static_cast<std::array<int, 3>>(o);
}

LogicalCoordinateTransformation GetTransform(const std::shared_ptr<Face> &from,
                                             const std::shared_ptr<Face> &to) {
  const auto off = from->neighbors_to_offsets.at(to);
  for (auto &[w_neighbor, ct] :
       from->neighbors(static_cast<int>(off[0]), static_cast<int>(off[1])))
    if (w_neighbor.lock() == to) return ct;
  PARTHENON_FAIL("Neighbor not found.");
  return LogicalCoordinateTransformation();
}

// The level 2 block of a face that touches the node with the given corner offsets
parthenon::LogicalLocation CornerBlock(std::int64_t tree,
                                       const parthenon::CellCentOffsets &corner) {
  const int lx1 = static_cast<int>(corner[0]) > 0 ? 3 : 0;
  const int lx2 = static_cast<int>(corner[1]) > 0 ? 3 : 0;
  return parthenon::LogicalLocation(tree, 2, lx1, lx2, 0);
}
} // namespace

TEST_CASE("Node neighbor transformations around irregular nodes", "[forest]") {
  for (int valence = 3; valence <= 8; ++valence) {
    for (int shift = 0; shift < 8; ++shift) {
      auto faces = FanFaces(valence, shift);
      int nnode_pairs = 0;
      for (auto &a : faces) {
        for (auto &[w_e, off] : a->neighbors_to_offsets) {
          if (!off.IsNode()) continue;
          ++nnode_pairs;
          auto e = w_e.lock();
          std::shared_ptr<Node> node;
          for (auto &na : a->nodes)
            if (e->face_index.count(na)) node = na;
          const auto corner_a = Face::node_to_offset[a->face_index.at(node)];
          const auto corner_e = Face::node_to_offset[e->face_index.at(node)];
          const auto t_ae = GetTransform(a, e);
          const auto t_ea = GetTransform(e, a);
          INFO("valence " << valence << " orientation shift " << shift << " face "
                          << a->GetId() << " -> node neighbor " << e->GetId());

          // The neighbor sits at this face's corner of the shared node
          CHECK(t_ae.use_offset);
          CHECK(t_ae.offset == std::array<int, 3>{static_cast<int>(corner_a[0]),
                                                  static_cast<int>(corner_a[1]), 0});

          // The offset to the neighbor seen from the neighbor points away from the node,
          // so the receiver's reverse index (tag maps, buffer keys) is its own offset
          // to this face
          const auto seen = t_ae.Transform(parthenon::CellCentOffsets(
              static_cast<int>(corner_a[0]), static_cast<int>(corner_a[1]), 0));
          CHECK(Arr(seen) == std::array<int, 3>{-static_cast<int>(corner_e[0]),
                                                -static_cast<int>(corner_e[1]), 0});

          // Corner blocks at the node map onto each other
          auto loc_a = CornerBlock(a->GetId(), corner_a);
          auto neigh = loc_a.GetSameLevelNeighbor(static_cast<int>(corner_a[0]),
                                                  static_cast<int>(corner_a[1]), 0);
          CHECK(t_ae.Transform(neigh, e->GetId()) == CornerBlock(e->GetId(), corner_e));

          // The two directions are inverses of each other
          const auto round_trip = ComposeTransformations(t_ae, t_ea);
          CHECK(round_trip.dir_connection == std::array<int, 3>{0, 1, 2});
          CHECK(round_trip.dir_flip == std::array<bool, 3>{false, false, false});
          CHECK(round_trip.offset == std::array<int, 3>{0, 0, 0});

          // Where both faces share an edge neighbor at the node, the transformation is
          // the composition of the two edge transformations
          for (auto &[w_b, off_b] : a->neighbors_to_offsets) {
            auto b = w_b.lock();
            if (!off_b.IsEdge() || !b->face_index.count(node)) continue;
            if (!e->neighbors_to_offsets.count(b) ||
                !e->neighbors_to_offsets.at(b).IsEdge())
              continue;
            const auto composed =
                ComposeTransformations(GetTransform(a, b), GetTransform(b, e));
            REQUIRE(composed.dir_connection == t_ae.dir_connection);
            REQUIRE(composed.dir_flip == t_ae.dir_flip);
            REQUIRE(composed.offset == t_ae.offset);
          }
        }
      }
      REQUIRE(nnode_pairs == valence * (valence - 3));
    }
  }
}

TEST_CASE("Singular forest neighbors are reciprocal", "[forest]") {
  for (int valence = 3; valence <= 8; ++valence) {
    GIVEN("A refined forest with an interior point of valence " +
          std::to_string(valence)) {
      auto forest = n_blocks(valence, valence);
      auto locs = forest.GetMeshBlockListAndResolveGids();
      REQUIRE(locs.size() == 10 + 7 * (valence - 1));
      int nmissing = 0;
      for (const auto &loc : locs) {
        for (const auto &nb : forest.FindNeighbors(loc)) {
          bool found = false;
          for (const auto &nbnb : forest.FindNeighbors(nb.global_loc))
            found = found || nbnb.global_loc == loc;
          if (!found) {
            ++nmissing;
            UNSCOPED_INFO(loc.label() << " lists " << nb.global_loc.label()
                                      << " which does not list it back");
          }
        }
      }
      REQUIRE(nmissing == 0);
    }
  }
}

//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
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
#ifndef MESH_LOGICAL_LOCATION_HPP_
#define MESH_LOGICAL_LOCATION_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#include "utils/morton_number.hpp"

namespace parthenon {

//--------------------------------------------------------------------------------------
//! \struct LogicalLocation
//  \brief stores logical location and level of MeshBlock

class LogicalLocation { // aggregate and POD type
  // These values can exceed the range of std::int32_t even if the root grid has only a
  // single MeshBlock if >30 levels of AMR are used, since the corresponding max index =
  // 1*2^31 > INT_MAX = 2^31 -1 for most 32-bit signed integer type impelementations
  std::int64_t lx1_, lx2_, lx3_;
  MortonNumber morton_;
  int level_;

 public:
  LogicalLocation(int lev, std::int64_t l1, std::int64_t l2, std::int64_t l3)
      : lx1_(l1), lx2_(l2), lx3_(l3), level_(lev), morton_(lev, l1, l2, l3) {}
  LogicalLocation() : LogicalLocation(0, 0, 0, 0) {}

  const auto &lx1() const { return lx1_; }
  const auto &lx2() const { return lx2_; }
  const auto &lx3() const { return lx3_; }
  const auto &level() const { return level_; }
  const auto &morton() const { return morton_; }

  // operators useful for sorting
  bool operator==(LogicalLocation &ll) {
    return ((ll.level() == level_) && (ll.lx1() == lx1_) && (ll.lx2() == lx2_) &&
            (ll.lx3() == lx3_));
  }
  // LFR: These are old comparison operators. Greater gets used for sorting
  //      the derefinenment list, but we may want to remove them at some
  //      point to avoid confusion with the comparators below.
  static bool Lesser(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level() < right.level();
  }
  static bool Greater(const LogicalLocation &left, const LogicalLocation &right) {
    return left.level() > right.level();
  }

  bool IsContainedIn(const LogicalLocation &container) const {
    if (container.level() > level_) return false;
    const std::int64_t shifted_lx1 = lx1_ >> (level_ - container.level());
    const std::int64_t shifted_lx2 = lx2_ >> (level_ - container.level());
    const std::int64_t shifted_lx3 = lx3_ >> (level_ - container.level());
    return (shifted_lx1 == container.lx1()) && (shifted_lx2 == container.lx2()) &&
           (shifted_lx3 == container.lx3());
  }

  bool Contains(const LogicalLocation &containee) const {
    if (containee.level() < level_) return false;
    const std::int64_t shifted_lx1 = containee.lx1() >> (containee.level() - level_);
    const std::int64_t shifted_lx2 = containee.lx2() >> (containee.level() - level_);
    const std::int64_t shifted_lx3 = containee.lx3() >> (containee.level() - level_);
    return (shifted_lx1 == lx1_) && (shifted_lx2 == lx2_) && (shifted_lx3 == lx3_);
  }

  // Being a neighbor implies that you share a face, edge, or node and don't share a
  // volume
  bool IsNeighbor(const LogicalLocation &in) const {
    if (in.level() < level()) return in.IsNeighbor(*this);
    if (Contains(in)) return false; // You share a volume
    // Only need to consider case where other block is equally or more refined than you
    auto offset = 1 << (in.level() - level());
    const auto shifted_lx1 = lx1_ << (in.level() - level());
    const auto shifted_lx2 = lx2_ << (in.level() - level());
    const auto shifted_lx3 = lx3_ << (in.level() - level());
    const bool bx1 =
        (in.lx1() >= (shifted_lx1 - 1)) && (in.lx1() <= (shifted_lx1 + offset));
    const bool bx2 =
        (in.lx2() >= (shifted_lx2 - 1)) && (in.lx2() <= (shifted_lx2 + offset));
    const bool bx3 =
        (in.lx3() >= (shifted_lx3 - 1)) && (in.lx3() <= (shifted_lx3 + offset));
    return bx1 && bx2 && bx3;
  }

  LogicalLocation GetSameLevelNeighbor(int ox1, int ox2, int ox3) const {
    return LogicalLocation(level_, lx1_ + ox1, lx2_ + ox2, lx3_ + ox3);
  }

  LogicalLocation GetParent() const {
    if (level_ == 0) return *this;
    return LogicalLocation(level_ - 1, lx1_ >> 1, lx2_ >> 1, lx3_ >> 1);
  }

  std::vector<LogicalLocation> GetDaughters() const {
    std::vector<LogicalLocation> daughters;
    for (int i : {0, 1}) {
      for (int j : {0, 1}) {
        for (int k : {0, 1}) {
          daughters.push_back(GetDaughter(i, j, k));
        }
      }
    }
    return daughters;
  }

  LogicalLocation GetDaughter(int ox1, int ox2, int ox3) const {
    return LogicalLocation(level_ + 1, (lx1_ << 1) + ox1, (lx2_ << 1) + ox2,
                           (lx3_ << 1) + ox3);
  }

  std::set<LogicalLocation>
  GetPossibleBlocksSurroundingTopologicalElement(int ox1, int ox2, int ox3) const {
    std::vector<LogicalLocation> locs;

    const auto irange =
        (std::abs(ox1) == 1) ? std::vector<int>{0, ox1} : std::vector<int>{0};
    const auto jrange =
        (std::abs(ox2) == 1) ? std::vector<int>{0, ox2} : std::vector<int>{0};
    const auto krange =
        (std::abs(ox3) == 1) ? std::vector<int>{0, ox3} : std::vector<int>{0};

    auto AddNeighbors = [&](const LogicalLocation &loc) {
      int n_cells_level = std::pow(2, loc.level());
      for (int i : irange) {
        for (int j : jrange) {
          for (int k : krange) {
            const auto lx1 = loc.lx1() + i;
            const auto lx2 = loc.lx2() + j;
            const auto lx3 = loc.lx3() + k;
            // TODO(LFR): Deal with periodic boundaries, maybe a little complicated
            // because of root grid stuff
            if (0 <= lx1 && lx1 < n_cells_level && 0 <= lx2 && lx2 < n_cells_level &&
                0 <= lx3 && lx3 < n_cells_level) {
              locs.emplace_back(loc.level(), lx1, lx2, lx3);
              auto parent = locs.back().GetParent();
              if (IsNeighbor(parent)) locs.push_back(parent);
            }
          }
        }
      }
    };

    AddNeighbors(*this);

    // Iterate over daughters of this block that share the same topological element
    for (int l :
         (std::abs(ox1) == 1) ? std::vector<int>{ox1 > 0} : std::vector<int>{0, 1}) {
      for (int m :
           (std::abs(ox2) == 1) ? std::vector<int>{ox2 > 0} : std::vector<int>{0, 1}) {
        for (int n :
             (std::abs(ox3) == 1) ? std::vector<int>{ox3 > 0} : std::vector<int>{0, 1}) {
          AddNeighbors(GetDaughter(l, m, n));
        }
      }
    }
    // The above procedure likely duplicated some blocks, so put them in a set
    return std::set<LogicalLocation>(std::begin(locs), std::end(locs));
  }
};

inline bool operator<(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() < rhs.level();
  return lhs.morton() < rhs.morton();
}

inline bool operator>(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  if (lhs.morton() == rhs.morton()) return lhs.level() > rhs.level();
  return lhs.morton() > rhs.morton();
}

inline bool operator==(const LogicalLocation &lhs, const LogicalLocation &rhs) {
  return ((lhs.level() == rhs.level()) && (lhs.lx1() == rhs.lx1()) &&
          (lhs.lx2() == rhs.lx2()) && (lhs.lx3() == rhs.lx3()));
}

struct block_ownership_t {
 public:
  const bool &operator()(int ox1, int ox2, int ox3) const {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }
  bool &operator()(int ox1, int ox2, int ox3) {
    return ownership[ox1 + 1][ox2 + 1][ox3 + 1];
  }

 private:
  bool ownership[3][3][3];
};

inline block_ownership_t
DetermineOwnership(const LogicalLocation &main_block,
                   const std::set<LogicalLocation> &allowed_neighbors) {
  block_ownership_t main_owns;

  auto ownership_less_than = [](const LogicalLocation &a, const LogicalLocation &b) {
    // Ownership is first determined by block with the highest level, then by maximum
    // Morton number this is reversed in precedence from the normal comparators where
    // Morton number takes precedence
    if (a.level() == b.level()) return a.morton() < b.morton();
    return a.level() < b.level();
  };

  for (int ox1 : {-1, 0, 1}) {
    for (int ox2 : {-1, 0, 1}) {
      for (int ox3 : {-1, 0, 1}) {
        auto possible_neighbors =
            main_block.GetPossibleBlocksSurroundingTopologicalElement(ox1, ox2, ox3);

        std::vector<LogicalLocation> actual_neighbors;
        std::set_intersection(std::begin(allowed_neighbors), std::end(allowed_neighbors),
                              std::begin(possible_neighbors),
                              std::end(possible_neighbors),
                              std::back_inserter(actual_neighbors));

        auto max = std::max_element(std::begin(actual_neighbors),
                                    std::end(actual_neighbors), ownership_less_than);
        main_owns(ox1, ox2, ox3) =
            (*max == main_block || ownership_less_than(*max, main_block) ||
             actual_neighbors.size() == 0);
      }
    }
  }
  return main_owns;
}

} // namespace parthenon

template <>
struct std::hash<parthenon::LogicalLocation> {
  std::size_t operator()(const parthenon::LogicalLocation &key) const noexcept {
    // TODO(LFR): Think more carefully about what the best choice for this key is,
    // probably the least significant sizeof(size_t) * 8 bits of the morton number
    // with 3 * (level - 21) trailing bits removed.
    return key.morton().bits[0];
  }
};

#endif // MESH_LOGICAL_LOCATION_HPP_

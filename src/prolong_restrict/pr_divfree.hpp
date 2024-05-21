#ifndef PROLONG_RESTRICT_PR_DIVFREE_HPP
#define PROLONG_RESTRICT_PR_DIVFREE_HPP
#include <algorithm>
#include <cstring>

#include "coordinates/coordinates.hpp"  // for coordinates
#include "interface/variable_state.hpp" // For variable state in ParArray
#include "kokkos_abstraction.hpp"       // ParArray
#include "mesh/domain.hpp"              // for IndesShape

namespace parthenon {
namespace refinement_ops {

namespace {
template <int pm>
KOKKOS_FORCEINLINE_FUNCTION Kokkos::Array<Real, 3>
GetDeltas(const Coordinates_t &coords) {
  return {{coords.Dxf<1>(), coords.Dxf<2>(), coords.Dxf<3>()}};
}

KOKKOS_FORCEINLINE_FUNCTION
constexpr Kokkos::Array<int, 3> Offsets(const int normDir, const int dim,
                                        const int mult = 1) {
  int i = 0, j = 0, k = 0;
  switch (normDir) {
  case (0):
    i = mult;
    break;
  case (1):
    j = mult;
    break;
  case (2):
    k = mult;
    break;
  }
  if (dim < 3) {
    k = 0;
  }
  return {{i, j, k}};
}

template <int I1, int I2, int I3, int DIM>
KOKKOS_FORCEINLINE_FUNCTION Real getCoarseFaceVal(
    const ParArrayND<Real, VariableState> *pfine, const Coordinates_t &coords,
    const int l, const int m, const int n, const int k, const int j, const int i) {
  auto &fine = *pfine;
  Real b1 = 0.;
  Real area = 0.;
  if constexpr (DIM == 3) {
    constexpr auto N2 = Offsets(I2, DIM);
    constexpr auto N3 = Offsets(I3, DIM);
    b1 += fine(I1, l, m, n, k, j, i) * coords.FaceArea<I1 + 1>(k, j, i);
    b1 += fine(I1, l, m, n, k + N2[2], j + N2[1], i + N2[0]) *
          coords.FaceArea<I1 + 1>(k + N2[2], j + N2[1], i + N2[0]);
    b1 += fine(I1, l, m, n, k + N3[2], j + N3[1], i + N3[0]) *
          coords.FaceArea<I1 + 1>(k + N3[2], j + N3[1], i + N3[0]);
    b1 +=
        fine(I1, l, m, n, k + N2[2] + N3[2], j + N2[1] + N3[1], i + N2[0] + N3[0]) *
        coords.FaceArea<I1 + 1>(k + N2[2] + N3[2], j + N2[1] + N3[1], i + N2[0] + N3[0]);
    area =
        coords.FaceArea<I1 + 1>(k, j, i) +
        coords.FaceArea<I1 + 1>(k + N2[2] + N3[2], j + N2[1] + N3[1], i + N2[0] + N3[0]) +
        coords.FaceArea<I1 + 1>(k + N2[2], j + N2[1], i + N2[0]) +
        coords.FaceArea<I1 + 1>(k + N3[2], j + N3[1], i + N3[0]);
  } else {
    constexpr auto N2 = (I2 < 2) ? Offsets(I2, DIM) : Offsets(I3, DIM);
    b1 += fine(I1, l, m, n, k, j, i) * coords.FaceArea<I1 + 1>(k, j, i);
    b1 += fine(I1, l, m, n, k + N2[2], j + N2[1], i + N2[0]) *
          coords.FaceArea<I1 + 1>(k + N2[2], j + N2[1], i + N2[0]);
    area = coords.FaceArea<I1 + 1>(k, j, i) +
           coords.FaceArea<I1 + 1>(k + N2[2], j + N2[1], i + N2[0]);
  }
  return b1 / area;
}

template <int I1, int I2, int I3, int DIM>
KOKKOS_FORCEINLINE_FUNCTION Real getFaceNormDif(
    const ParArrayND<Real, VariableState> *pfine, const Coordinates_t &coords,
    const int l, const int m, const int n, const int k, const int j, const int i) {

  auto &fine = *pfine;
  constexpr auto N2 = Offsets(I2, DIM);
  constexpr auto N3 = Offsets(I3, DIM);
  return (fine(I1, l, m, n, k + N2[2] + N3[2], j + N2[1] + N3[1], i + N2[0] + N3[0]) -
          fine(I1, l, m, n, k, j, i) +
          fine(I1, l, m, n, k + N2[2], j + N2[1], i + N2[0]) -
          fine(I1, l, m, n, k + N3[2], j + N3[1], i + N3[0]));
}

template <int I1, int I2, int I3>
KOKKOS_FORCEINLINE_FUNCTION Real getFaceCrossDif(
    const ParArrayND<Real, VariableState> *pfine, const Coordinates_t &coords,
    const int l, const int m, const int n, const int k, const int j, const int i) {

  auto &fine = *pfine;
  // d23b1 = 4(b1mm + b1pp - b1pm - b1mp)
  constexpr auto N2 = Offsets(I2, 3);
  constexpr auto N3 = Offsets(I3, 3);
  // this is missing any geometric factors for non-cartesian geometries
  return 4. *
         (fine(I1, l, m, n, k, j, i) +
          fine(I1, l, m, n, k + N2[2] + N3[2], j + N2[1] + N3[1], i + N2[0] + N3[0]) -
          fine(I1, l, m, n, k + N2[2], j + N2[1], i + N2[0]) -
          fine(I1, l, m, n, k + N3[2], j + N3[1], i + N3[0]));
}

} // namespace

struct ProlongateInternalBalsara {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return IsSubmanifold(fel, cel);
  }
  // Implements the divergence reconstruction from
  // [1] D. S. Balsara, “Divergence-Free Adaptive Mesh Refinement for
  // Magnetohydrodynamics,
  //     ” Journal of Computational Physics, vol. 174, no. 2, pp. 614–648, Dec. 2001,
  //     doi: 10.1006/jcph.2001.6917.
  // The following is Eqs~(3.32)-(3.38) together with 4.4-4.5
  template <int DIM, TopologicalElement fel = TopologicalElement::CC,
            TopologicalElement cel = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, VariableState> *,
     const ParArrayND<Real, VariableState> *pfine) {

    constexpr int I1 = static_cast<int>(fel) % 3;
    if constexpr (I1 + 1 > DIM) {
      return;
    } else {

      auto &fine = *pfine;
      // this should be the index for the lower left fine face on the coarse cell
      const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
      const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
      const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

      constexpr int I2 = (I1 + 1) % 3;
      constexpr int I3 = (I1 + 2) % 3;
      const Real d1 = coarse_coords.Dxf<I1 + 1>();
      const Real d2 = coarse_coords.Dxf<I2 + 1>();
      const Real d3 = coarse_coords.Dxf<I3 + 1>();
      const Real id1 = 1. / d1;
      const Real id2 = 1. / d2;
      const Real id3 = 1. / d3;

      // bip/m is face-field at plus/minuse relative to cell center in i-direction
      constexpr auto N1 = Offsets(I1, DIM, 2);
      const Real b1m =
          getCoarseFaceVal<I1, I2, I3, DIM>(pfine, coords, l, m, n, fk, fj, fi);
      const Real b1p = getCoarseFaceVal<I1, I2, I3, DIM>(
          pfine, coords, l, m, n, fk + N1[2], fj + N1[1], fi + N1[0]);

      constexpr auto N2 = Offsets(I2, DIM, 2);
      const Real b2m =
          getCoarseFaceVal<I2, I3, I1, DIM>(pfine, coords, l, m, n, fk, fj, fi);
      const Real b2p = getCoarseFaceVal<I2, I3, I1, DIM>(
          pfine, coords, l, m, n, fk + N2[2], fj + N2[1], fi + N2[0]);

      constexpr auto N3 = Offsets(I3, DIM, 2);
      const Real b3m =
          getCoarseFaceVal<I3, I1, I2, DIM>(pfine, coords, l, m, n, fk, fj, fi);
      const Real b3p = getCoarseFaceVal<I3, I1, I2, DIM>(
          pfine, coords, l, m, n, fk + N3[2], fj + N3[1], fi + N3[0]);
      Real divbC = (b1p - b1m) * id1 + (b2p - b2m) * id2 + (b3p - b3m) * id3;

      // dibjp/m is the transverse derivative of bj in the i-th direction at the p/m face
      const Real d1b2p = getFaceNormDif<I2, I1, I3, DIM>(
                             pfine, coords, l, m, n, fk + N2[2], fj + N2[1], fi + N2[0]) *
                         id1;
      const Real d1b2m =
          getFaceNormDif<I2, I1, I3, DIM>(pfine, coords, l, m, n, fk, fj, fi) * id1;

      const Real d1b3p = getFaceNormDif<I3, I1, I2, DIM>(
                             pfine, coords, l, m, n, fk + N3[2], fj + N3[1], fi + N3[0]) *
                         id1;
      const Real d1b3m =
          getFaceNormDif<I3, I1, I2, DIM>(pfine, coords, l, m, n, fk, fj, fi) * id1;

      const Real d2b1p = getFaceNormDif<I1, I2, I3, DIM>(
                             pfine, coords, l, m, n, fk + N1[2], fj + N1[1], fi + N1[0]) *
                         id2;
      const Real d2b1m =
          getFaceNormDif<I1, I2, I3, DIM>(pfine, coords, l, m, n, fk, fj, fi) * id2;

      const Real d3b1p = getFaceNormDif<I1, I3, I2, DIM>(
                             pfine, coords, l, m, n, fk + N1[2], fj + N1[1], fi + N1[0]) *
                         id3;
      const Real d3b1m =
          getFaceNormDif<I1, I3, I2, DIM>(pfine, coords, l, m, n, fk, fj, fi) * id3;

      Real a123 = 0., b123 = 0., c123 = 0., a23 = 0.;
      if constexpr (DIM == 3) {
        const Real d12b1p = getFaceCrossDif<I1, I2, I3>(
            pfine, coords, l, m, n, fk + N1[2], fj + N1[1], fi + N1[0]);
        const Real d12b1m =
            getFaceCrossDif<I1, I2, I3>(pfine, coords, l, m, n, fk, fj, fi);
        a23 = 0.5 * id2 * id3 * (d12b1p + d12b1m);
        /* a123 = id1*id2*id3*(d12b1p - d12b1m); */
        b123 = id1 * id2 * id3 *
               (getFaceCrossDif<I2, I3, I1>(pfine, coords, l, m, n, fk + N2[2],
                                            fj + N2[1], fi + N2[0]) -
                getFaceCrossDif<I2, I3, I1>(pfine, coords, l, m, n, fk, fj, fi));
        c123 = id1 * id2 * id3 *
               (getFaceCrossDif<I3, I1, I2>(pfine, coords, l, m, n, fk + N3[2],
                                            fj + N3[1], fi + N3[0]) -
                getFaceCrossDif<I3, I1, I2>(pfine, coords, l, m, n, fk, fj, fi));
      }
      // b1(x1,x2,x3) = a0 + a1*x1 + a11*x1^2 + a2*x2 + a3*x3 + a12*x1*x2 + a13*x1*x3 +
      //                a23*x2*x3 + a123*x1*x2*x3 + a112*x1^2*x2 + a113*x1^2*x3
      // origin is at the center of the coarse cell
      const Real a1 = id1 * (b1p - b1m);
      const Real a2 = 0.5 * (d2b1p + d2b1m) + 1. / 16. * c123 * d1 * d1;
      const Real a3 = 0.5 * (d3b1p + d3b1m) + 1. / 16. * b123 * d1 * d1;
      const Real a12 = id1 * (d2b1p - d2b1m);
      const Real a13 = id1 * (d3b1p - d3b1m);
      const Real b12 = id2 * (d1b2p - d1b2m);
      const Real c13 = id3 * (d1b3p - d1b3m);

      const Real a11 = -0.5 * (b12 + c13);
      const Real a0 = 0.5 * (b1p + b1m) - 0.25 * a11 * d1 * d1;
      constexpr auto n1 = Offsets(I1, DIM);
      constexpr auto n2 = (I2 < DIM) ? Offsets(I2, DIM) : Offsets(I3, DIM);
      if constexpr (I2 < DIM) {
        fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) = a0 - 0.25 * d2 * a2;
        fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) =
            a0 + 0.25 * d2 * a2;
      } else {
        fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) = a0 - 0.25 * d3 * a3;
        fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) =
            a0 + 0.25 * d3 * a3;
      }
      if constexpr (DIM == 3) {
        fine(I1, l, m, n, fk + n1[2], fj + n1[1], fi + n1[0]) +=
            -0.25 * d3 * a3 + 1. / 16. * d2 * d3 * a23;
        fine(I1, l, m, n, fk + n1[2] + n2[2], fj + n1[1] + n2[1], fi + n1[0] + n2[0]) +=
            -0.25 * d3 * a3 - 1. / 16. * d2 * d3 * a23;

        constexpr auto n3 = Offsets(I3, DIM);
        fine(I1, l, m, n, fk + n1[2] + n3[2], fj + n1[1] + n3[1], fi + n1[0] + n3[0]) =
            a0 - 0.25 * d2 * a2 + 0.25 * d3 * a3 - 1. / 16. * d2 * d3 * a23;
        fine(I1, l, m, n, fk + n1[2] + n2[2] + n3[2], fj + n1[1] + n2[1] + n3[1],
             fi + n1[0] + n2[0] + n3[0]) =
            a0 + 0.25 * d2 * a2 + 0.25 * d3 * a3 + 1. / 16. * d2 * d3 * a23;
      }
    }
  }
};

} // namespace refinement_ops
} // namespace parthenon

#endif

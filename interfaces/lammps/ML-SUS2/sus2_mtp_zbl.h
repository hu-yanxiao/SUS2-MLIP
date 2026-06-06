/* -*- c++ -*- ----------------------------------------------------------
   ZBL helper for the ML-SUS2 LAMMPS interface.
------------------------------------------------------------------------- */

#ifndef LMP_SUS2_MTP_ZBL_H
#define LMP_SUS2_MTP_ZBL_H

#include <cmath>

struct SUS2MTPZBLPairValue {
  double energy;
  double dEdr;
};

struct SUS2MTPZBLPairConstants {
  double screening_inv;
  double prefactor;
};

namespace sus2_mtp_zbl {

inline double DefaultInnerCutoff() { return 0.7; }
inline double DefaultOuterCutoff() { return 1.4; }
inline double DefaultTypewiseCutoffFactor() { return 0.7; }

inline double CovalentRadius(int atomic_number)
{
  static const double radii[94] = {
      0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333,
      0.893333, 1.86667,  1.66667, 1.50667, 1.38667, 1.46667, 1.36,     1.32,    1.28,
      2.34667,  2.05333,  1.77333, 1.62667, 1.61333, 1.46667, 1.42667,  1.38667, 1.33333,
      1.32,     1.34667,  1.45333, 1.49333, 1.45333, 1.53333, 1.46667,  1.52,    1.56,
      2.52,     2.22667,  1.96,    1.85333, 1.76,    1.65333, 1.53333,  1.50667, 1.50667,
      1.44,     1.53333,  1.64,    1.70667, 1.68,    1.68,    1.64,     1.76,    1.74667,
      2.78667,  2.34667,  2.16,    1.96,    2.10667, 2.09333, 2.08,     2.06667, 2.01333,
      2.02667,  2.01333,  2.0,     1.98667, 1.98667, 1.97333, 2.04,     1.94667, 1.82667,
      1.74667,  1.64,     1.57333, 1.54667, 1.48,    1.49333, 1.50667,  1.76,    1.73333,
      1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,  2.22667, 2.10667,
      2.02667,  2.04,     2.05333, 2.06667};
  if (atomic_number < 1 || atomic_number > 94) return 0.0;
  return radii[atomic_number - 1];
}

inline double PairOuterCutoff(int atomic_number_i, int atomic_number_j,
                              double global_outer_cutoff,
                              double typewise_cutoff_factor)
{
  const double pair_outer =
      typewise_cutoff_factor *
      (CovalentRadius(atomic_number_i) + CovalentRadius(atomic_number_j));
  return pair_outer < global_outer_cutoff ? pair_outer : global_outer_cutoff;
}

inline void FillPairCutoffTables(int species_count, const int *atomic_numbers,
                                 double global_inner_cutoff,
                                 double global_outer_cutoff,
                                 bool typewise_cutoff_enabled,
                                 double typewise_cutoff_factor,
                                 double *pair_inner_cutoffs,
                                 double *pair_outer_cutoffs,
                                 double *pair_outer_sq)
{
  for (int i = 0; i < species_count; ++i) {
    for (int j = 0; j < species_count; ++j) {
      const int pair_index = i * species_count + j;
      const double pair_inner = typewise_cutoff_enabled ? 0.0 : global_inner_cutoff;
      const double pair_outer = typewise_cutoff_enabled ?
          PairOuterCutoff(atomic_numbers[i], atomic_numbers[j], global_outer_cutoff,
                          typewise_cutoff_factor) :
          global_outer_cutoff;
      pair_inner_cutoffs[pair_index] = pair_inner;
      pair_outer_cutoffs[pair_index] = pair_outer;
      pair_outer_sq[pair_index] = pair_outer * pair_outer;
    }
  }
}

inline double MaxPairOuterCutoff(int species_count, const int *atomic_numbers,
                                 double global_outer_cutoff,
                                 bool typewise_cutoff_enabled,
                                 double typewise_cutoff_factor)
{
  if (!typewise_cutoff_enabled) return global_outer_cutoff;
  double max_outer = 0.0;
  for (int i = 0; i < species_count; ++i) {
    for (int j = 0; j < species_count; ++j) {
      const double pair_outer =
          PairOuterCutoff(atomic_numbers[i], atomic_numbers[j],
                          global_outer_cutoff, typewise_cutoff_factor);
      if (pair_outer > max_outer) max_outer = pair_outer;
    }
  }
  return max_outer;
}

inline SUS2MTPZBLPairConstants MakePairConstants(int atomic_number_i,
                                                 int atomic_number_j)
{
  const double ev_angstrom_per_e2 = 14.3996454784255;
  const double screening_inv =
      2.134563 * (std::pow(static_cast<double>(atomic_number_i), 0.23) +
                  std::pow(static_cast<double>(atomic_number_j), 0.23));
  const double prefactor = ev_angstrom_per_e2 *
      static_cast<double>(atomic_number_i) *
      static_cast<double>(atomic_number_j);
  return SUS2MTPZBLPairConstants{screening_inv, prefactor};
}

inline void FillPairConstants(int species_count, const int *atomic_numbers,
                              SUS2MTPZBLPairConstants *pair_constants)
{
  for (int i = 0; i < species_count; ++i)
    for (int j = 0; j < species_count; ++j)
      pair_constants[i * species_count + j] =
          MakePairConstants(atomic_numbers[i], atomic_numbers[j]);
}

inline SUS2MTPZBLPairValue ComputePairHostCached(
    const SUS2MTPZBLPairConstants& constants, double distance,
    double inner_cutoff, double outer_cutoff);

inline SUS2MTPZBLPairValue ComputePairHost(int atomic_number_i, int atomic_number_j,
                                           double distance, double inner_cutoff,
                                           double outer_cutoff,
                                           double typewise_cutoff_factor);

inline SUS2MTPZBLPairValue ComputePairHost(int atomic_number_i, int atomic_number_j,
                                           double distance, double inner_cutoff,
                                           double outer_cutoff)
{
  return ComputePairHost(atomic_number_i, atomic_number_j, distance, inner_cutoff,
                         outer_cutoff, 0.0);
}

inline SUS2MTPZBLPairValue ComputePairHost(int atomic_number_i, int atomic_number_j,
                                           double distance, double inner_cutoff,
                                           double outer_cutoff,
                                           double typewise_cutoff_factor)
{
  if (typewise_cutoff_factor > 0.0) {
    inner_cutoff = 0.0;
    outer_cutoff = PairOuterCutoff(atomic_number_i, atomic_number_j,
                                   outer_cutoff, typewise_cutoff_factor);
  }
  return ComputePairHostCached(MakePairConstants(atomic_number_i, atomic_number_j),
                               distance, inner_cutoff, outer_cutoff);
}

inline SUS2MTPZBLPairValue ComputePairHostCached(
    const SUS2MTPZBLPairConstants& constants, double distance,
    double inner_cutoff, double outer_cutoff)
{
  if (distance >= outer_cutoff) return SUS2MTPZBLPairValue{0.0, 0.0};

  const double coefficients[4] = {0.18175, 0.50986, 0.28022, 0.02817};
  const double exponents[4] = {3.1998, 0.94229, 0.4029, 0.20162};
  const double x = constants.screening_inv * distance;

  double phi = 0.0;
  double dphi_dr = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double exp_value = std::exp(-exponents[i] * x);
    phi += coefficients[i] * exp_value;
    dphi_dr -= coefficients[i] * exponents[i] *
        constants.screening_inv * exp_value;
  }

  const double base_energy = constants.prefactor * phi / distance;
  const double base_dEdr =
      constants.prefactor * (dphi_dr / distance - phi / (distance * distance));

  double switch_value = 1.0;
  double switch_derivative = 0.0;
  if (distance > inner_cutoff) {
    const double pi_factor = std::acos(-1.0) / (outer_cutoff - inner_cutoff);
    switch_value = 0.5 * std::cos(pi_factor * (distance - inner_cutoff)) + 0.5;
    switch_derivative =
        -0.5 * pi_factor * std::sin(pi_factor * (distance - inner_cutoff));
  }

  return SUS2MTPZBLPairValue{switch_value * base_energy,
                             switch_value * base_dEdr +
                                 switch_derivative * base_energy};
}

#ifdef KOKKOS_INLINE_FUNCTION
template <typename Float> struct SUS2MTPZBLPairValueKokkos {
  Float energy;
  Float dEdr;
};

template <typename Float>
KOKKOS_INLINE_FUNCTION Float CovalentRadiusKokkos(int atomic_number)
{
  const Float radii[94] = {
      static_cast<Float>(0.426667), static_cast<Float>(0.613333), static_cast<Float>(1.6),
      static_cast<Float>(1.25333), static_cast<Float>(1.02667), static_cast<Float>(1.0),
      static_cast<Float>(0.946667), static_cast<Float>(0.84), static_cast<Float>(0.853333),
      static_cast<Float>(0.893333), static_cast<Float>(1.86667), static_cast<Float>(1.66667),
      static_cast<Float>(1.50667), static_cast<Float>(1.38667), static_cast<Float>(1.46667),
      static_cast<Float>(1.36), static_cast<Float>(1.32), static_cast<Float>(1.28),
      static_cast<Float>(2.34667), static_cast<Float>(2.05333), static_cast<Float>(1.77333),
      static_cast<Float>(1.62667), static_cast<Float>(1.61333), static_cast<Float>(1.46667),
      static_cast<Float>(1.42667), static_cast<Float>(1.38667), static_cast<Float>(1.33333),
      static_cast<Float>(1.32), static_cast<Float>(1.34667), static_cast<Float>(1.45333),
      static_cast<Float>(1.49333), static_cast<Float>(1.45333), static_cast<Float>(1.53333),
      static_cast<Float>(1.46667), static_cast<Float>(1.52), static_cast<Float>(1.56),
      static_cast<Float>(2.52), static_cast<Float>(2.22667), static_cast<Float>(1.96),
      static_cast<Float>(1.85333), static_cast<Float>(1.76), static_cast<Float>(1.65333),
      static_cast<Float>(1.53333), static_cast<Float>(1.50667), static_cast<Float>(1.50667),
      static_cast<Float>(1.44), static_cast<Float>(1.53333), static_cast<Float>(1.64),
      static_cast<Float>(1.70667), static_cast<Float>(1.68), static_cast<Float>(1.68),
      static_cast<Float>(1.64), static_cast<Float>(1.76), static_cast<Float>(1.74667),
      static_cast<Float>(2.78667), static_cast<Float>(2.34667), static_cast<Float>(2.16),
      static_cast<Float>(1.96), static_cast<Float>(2.10667), static_cast<Float>(2.09333),
      static_cast<Float>(2.08), static_cast<Float>(2.06667), static_cast<Float>(2.01333),
      static_cast<Float>(2.02667), static_cast<Float>(2.01333), static_cast<Float>(2.0),
      static_cast<Float>(1.98667), static_cast<Float>(1.98667), static_cast<Float>(1.97333),
      static_cast<Float>(2.04), static_cast<Float>(1.94667), static_cast<Float>(1.82667),
      static_cast<Float>(1.74667), static_cast<Float>(1.64), static_cast<Float>(1.57333),
      static_cast<Float>(1.54667), static_cast<Float>(1.48), static_cast<Float>(1.49333),
      static_cast<Float>(1.50667), static_cast<Float>(1.76), static_cast<Float>(1.73333),
      static_cast<Float>(1.73333), static_cast<Float>(1.81333), static_cast<Float>(1.74667),
      static_cast<Float>(1.84), static_cast<Float>(1.89333), static_cast<Float>(2.68),
      static_cast<Float>(2.41333), static_cast<Float>(2.22667), static_cast<Float>(2.10667),
      static_cast<Float>(2.02667), static_cast<Float>(2.04), static_cast<Float>(2.05333),
      static_cast<Float>(2.06667)};
  if (atomic_number < 1 || atomic_number > 94) return static_cast<Float>(0.0);
  return radii[atomic_number - 1];
}

template <typename Float>
KOKKOS_INLINE_FUNCTION Float PairOuterCutoffKokkos(int atomic_number_i,
                                                   int atomic_number_j,
                                                   Float global_outer_cutoff,
                                                   Float typewise_cutoff_factor)
{
  const Float pair_outer = typewise_cutoff_factor *
      (CovalentRadiusKokkos<Float>(atomic_number_i) +
       CovalentRadiusKokkos<Float>(atomic_number_j));
  return pair_outer < global_outer_cutoff ? pair_outer : global_outer_cutoff;
}

template <typename Float>
KOKKOS_INLINE_FUNCTION SUS2MTPZBLPairValueKokkos<Float>
ComputePairKokkos(int atomic_number_i, int atomic_number_j, Float distance,
                  Float inner_cutoff, Float outer_cutoff,
                  Float typewise_cutoff_factor);

template <typename Float>
KOKKOS_INLINE_FUNCTION SUS2MTPZBLPairValueKokkos<Float>
ComputePairKokkos(int atomic_number_i, int atomic_number_j, Float distance,
                  Float inner_cutoff, Float outer_cutoff)
{
  return ComputePairKokkos<Float>(atomic_number_i, atomic_number_j, distance,
                                  inner_cutoff, outer_cutoff,
                                  static_cast<Float>(0.0));
}

template <typename Float>
KOKKOS_INLINE_FUNCTION SUS2MTPZBLPairValueKokkos<Float>
ComputePairKokkos(int atomic_number_i, int atomic_number_j, Float distance,
                  Float inner_cutoff, Float outer_cutoff,
                  Float typewise_cutoff_factor)
{
  if (typewise_cutoff_factor > static_cast<Float>(0.0)) {
    inner_cutoff = static_cast<Float>(0.0);
    outer_cutoff = PairOuterCutoffKokkos<Float>(atomic_number_i, atomic_number_j,
                                                outer_cutoff,
                                                typewise_cutoff_factor);
  }
  if (distance >= outer_cutoff)
    return SUS2MTPZBLPairValueKokkos<Float>{static_cast<Float>(0.0),
                                            static_cast<Float>(0.0)};

  const Float coefficients[4] = {static_cast<Float>(0.18175),
                                 static_cast<Float>(0.50986),
                                 static_cast<Float>(0.28022),
                                 static_cast<Float>(0.02817)};
  const Float exponents[4] = {static_cast<Float>(3.1998),
                              static_cast<Float>(0.94229),
                              static_cast<Float>(0.4029),
                              static_cast<Float>(0.20162)};
  const Float screening_inv = static_cast<Float>(2.134563) *
      (Kokkos::pow(static_cast<Float>(atomic_number_i), static_cast<Float>(0.23)) +
       Kokkos::pow(static_cast<Float>(atomic_number_j), static_cast<Float>(0.23)));
  const Float x = screening_inv * distance;

  Float phi = static_cast<Float>(0.0);
  Float dphi_dr = static_cast<Float>(0.0);
  for (int i = 0; i < 4; ++i) {
    const Float exp_value = Kokkos::exp(-exponents[i] * x);
    phi += coefficients[i] * exp_value;
    dphi_dr -= coefficients[i] * exponents[i] * screening_inv * exp_value;
  }

  const Float prefactor = static_cast<Float>(14.3996454784255) *
      static_cast<Float>(atomic_number_i) * static_cast<Float>(atomic_number_j);
  const Float base_energy = prefactor * phi / distance;
  const Float base_dEdr =
      prefactor * (dphi_dr / distance - phi / (distance * distance));

  Float switch_value = static_cast<Float>(1.0);
  Float switch_derivative = static_cast<Float>(0.0);
  if (distance > inner_cutoff) {
    const Float pi_factor = static_cast<Float>(3.141592653589793238462643383279502884) /
        (outer_cutoff - inner_cutoff);
    switch_value = static_cast<Float>(0.5) *
            Kokkos::cos(pi_factor * (distance - inner_cutoff)) +
        static_cast<Float>(0.5);
    switch_derivative =
        -static_cast<Float>(0.5) * pi_factor *
        Kokkos::sin(pi_factor * (distance - inner_cutoff));
  }

  return SUS2MTPZBLPairValueKokkos<Float>{
      switch_value * base_energy,
      switch_value * base_dEdr + switch_derivative * base_energy};
}
#endif

} // namespace sus2_mtp_zbl

#endif

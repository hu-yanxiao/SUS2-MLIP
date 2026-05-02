/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Contributing author, Richard Meng, Queen's University at Kingston, 21.01.24, contact@richardzjm.com
//

#include "pair_sus2_mtp_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"

#include <cmath>

using namespace LAMMPS_NS;

namespace {

constexpr int kEnvGateChannels = 6;
constexpr KK_FLOAT kEnvGateMaxLogDensityCoeff = 6.0;

KOKKOS_INLINE_FUNCTION KK_FLOAT laguerre_positive_param_kk(KK_FLOAT raw)
{
  constexpr KK_FLOAT floor = 1.0e-6;
  if (raw > static_cast<KK_FLOAT>(40.0)) return floor + raw;
  if (raw < static_cast<KK_FLOAT>(-40.0)) return floor + Kokkos::exp(raw);
  return floor + Kokkos::log(static_cast<KK_FLOAT>(1.0) + Kokkos::exp(raw));
}

KOKKOS_INLINE_FUNCTION KK_FLOAT stable_sigmoid_kk(KK_FLOAT value)
{
  if (value >= static_cast<KK_FLOAT>(0.0)) {
    const KK_FLOAT z = Kokkos::exp(-value);
    return static_cast<KK_FLOAT>(1.0) / (static_cast<KK_FLOAT>(1.0) + z);
  }
  const KK_FLOAT z = Kokkos::exp(value);
  return z / (static_cast<KK_FLOAT>(1.0) + z);
}

KOKKOS_INLINE_FUNCTION KK_FLOAT env_gate_density_coeff_kk(KK_FLOAT raw_log_density)
{
  const KK_FLOAT capped =
      raw_log_density < kEnvGateMaxLogDensityCoeff ? raw_log_density : kEnvGateMaxLogDensityCoeff;
  return Kokkos::exp(capped);
}

KOKKOS_INLINE_FUNCTION KK_FLOAT clamp01_kk(KK_FLOAT value)
{
  if (value < static_cast<KK_FLOAT>(0.0)) return static_cast<KK_FLOAT>(0.0);
  if (value > static_cast<KK_FLOAT>(1.0)) return static_cast<KK_FLOAT>(1.0);
  return value;
}

KOKKOS_INLINE_FUNCTION void bernstein_degree5_kk(KK_FLOAT y, KK_FLOAT *basis,
                                                 KK_FLOAT *basis_der)
{
  y = clamp01_kk(y);
  const KK_FLOAT u = static_cast<KK_FLOAT>(1.0) - y;
  const KK_FLOAT y2 = y * y;
  const KK_FLOAT y3 = y2 * y;
  const KK_FLOAT y4 = y3 * y;
  const KK_FLOAT y5 = y4 * y;
  const KK_FLOAT u2 = u * u;
  const KK_FLOAT u3 = u2 * u;
  const KK_FLOAT u4 = u3 * u;
  const KK_FLOAT u5 = u4 * u;

  basis[0] = u5;
  basis[1] = static_cast<KK_FLOAT>(5.0) * y * u4;
  basis[2] = static_cast<KK_FLOAT>(10.0) * y2 * u3;
  basis[3] = static_cast<KK_FLOAT>(10.0) * y3 * u2;
  basis[4] = static_cast<KK_FLOAT>(5.0) * y4 * u;
  basis[5] = y5;

  KK_FLOAT degree4[5] = {u4, static_cast<KK_FLOAT>(4.0) * y * u3,
                         static_cast<KK_FLOAT>(6.0) * y2 * u2,
                         static_cast<KK_FLOAT>(4.0) * y3 * u, y4};
  basis_der[0] = static_cast<KK_FLOAT>(-5.0) * degree4[0];
  for (int q = 1; q < 5; ++q)
    basis_der[q] = static_cast<KK_FLOAT>(5.0) * (degree4[q - 1] - degree4[q]);
  basis_der[5] = static_cast<KK_FLOAT>(5.0) * degree4[4];
}

KOKKOS_INLINE_FUNCTION void env_gate_activation_kk(KK_FLOAT r, KK_FLOAT r_env,
                                                   KK_FLOAT activation_on_ratio,
                                                   KK_FLOAT &activation,
                                                   KK_FLOAT &activation_der)
{
  const KK_FLOAT r_on = activation_on_ratio * r_env;
  if (r <= r_on) {
    activation = static_cast<KK_FLOAT>(0.0);
    activation_der = static_cast<KK_FLOAT>(0.0);
    return;
  }
  if (r >= r_env) {
    activation = static_cast<KK_FLOAT>(1.0);
    activation_der = static_cast<KK_FLOAT>(0.0);
    return;
  }

  const KK_FLOAT width = r_env - r_on;
  const KK_FLOAT t = (r - r_on) / width;
  const KK_FLOAT t2 = t * t;
  const KK_FLOAT t3 = t2 * t;
  activation = static_cast<KK_FLOAT>(10.0) * t3 -
               static_cast<KK_FLOAT>(15.0) * t3 * t +
               static_cast<KK_FLOAT>(6.0) * t3 * t2;
  activation_der = static_cast<KK_FLOAT>(30.0) * t2 *
                   (static_cast<KK_FLOAT>(1.0) - t) *
                   (static_cast<KK_FLOAT>(1.0) - t) / width;
}

}    // namespace

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairSUS2MTPKokkos<DeviceType>::PairSUS2MTPKokkos(LAMMPS(*lmp)) : PairSUS2MTP(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;

  host_flag = (execution_space == Host);
  centroidstressflag = CENTROID_AVAIL;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> PairSUS2MTPKokkos<DeviceType>::~PairSUS2MTPKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  memoryKK->destroy_kokkos(k_cvatom, cvatom);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template <class DeviceType> void PairSUS2MTPKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR, "Pair style sus2mtp/kk can currently only run on a single CPU thread.");

    PairSUS2MTP::init_style();
    return;
  }

  if (force->newton_pair == 0) error->all(FLERR, "Pair style SUS2MTP requires newton pair on.");

  // neighbor list request for KOKKOS
  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType, LMPHostType> &&
                           !std::is_same_v<DeviceType, LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType, LMPDeviceType>);
  if (neighflag == FULL) error->all(FLERR, "Must use half neighbor list style with pair sus2mtp/kk.");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template <class DeviceType> double PairSUS2MTPKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairSUS2MTP::init_one(i, j);
  //Don't need to do anything with the cutoff because the MTP (and original MLIP package) only uses one cutoff for all species combos.
  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template <class DeviceType> void PairSUS2MTPKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairSUS2MTP::coeff(narg, arg);
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template <class DeviceType> void PairSUS2MTPKokkos<DeviceType>::settings(int narg, char **arg)
{
  // We may need to process in chunks to deal with memory limitations
  // For now we expect the user to specify the chunk size, with an optional tabstep.
  if (narg < 3 || (narg - 1) % 2 != 0)
    error->all(
        FLERR,
        "Pair sus2mtp/kk requires a potential file plus keyword/value pairs, including \"chunksize\".");

  int tabstep_arg_idx = -1;
  bool found_chunksize = false;
  for (int i = 1; i < narg; i += 2) {
    const std::string keyword = LAMMPS_NS::utils::lowercase(arg[i]);
    if (keyword == "chunksize") {
      input_chunk_size = utils::inumeric(FLERR, arg[i + 1], true, lmp);
      found_chunksize = true;
      continue;
    }
    if (keyword == "tabstep") {
      tabstep_arg_idx = i;
      continue;
    }

    error->all(FLERR,
               "Pair sus2mtp/kk only supports the keywords \"chunksize\" and \"tabstep\".");
  }

  if (!found_chunksize)
    error->all(FLERR, "Pair sus2mtp/kk requires the keyword \"chunksize\".");

  char *base_args[3] = {arg[0], nullptr, nullptr};
  int base_narg = 1;
  if (tabstep_arg_idx >= 0) {
    base_args[base_narg++] = arg[tabstep_arg_idx];
    base_args[base_narg++] = arg[tabstep_arg_idx + 1];
  }

  PairSUS2MTP::settings(
      base_narg, base_args);    // This also calls read_file which parses and loads the arrays in host

  // Store SUS2-MLIP parameters from base class (CPU version already calculated these)
  L_max = PairSUS2MTP::L_max;
  K_scaling = PairSUS2MTP::K_scaling;
  regression_coeffs_count = PairSUS2MTP::regression_coeffs_count;

  // ---------- Now we move arrays to device ----------
  // First we set up the index lists
  MemKK::realloc_kokkos(d_alpha_index_basic, "sus2mtp/kk:alpha_index_basic", alpha_index_basic_count,
                        4);
  MemKK::realloc_kokkos(d_alpha_index_times, "sus2mtp/kk:alpha_index_times", alpha_index_times_count,
                        4);
  MemKK::realloc_kokkos(d_alpha_moment_mapping, "sus2mtp/kk:moment_mapping", alpha_scalar_count);
  
  // SUS2-MLIP: Setup mapping arrays
  MemKK::realloc_kokkos(d_mu_to_K, "sus2mtp/kk:mu_to_K", radial_func_count);
  MemKK::realloc_kokkos(d_mu_to_sigma, "sus2mtp/kk:mu_to_sigma", radial_func_count);
  MemKK::realloc_kokkos(d_k_to_mu_offsets, "sus2mtp/kk:k_to_mu_offsets", K_scaling + 1);
  MemKK::realloc_kokkos(d_grouped_mu, "sus2mtp/kk:grouped_mu", radial_func_count);
  MemKK::realloc_kokkos(d_mu_radial_offset, "sus2mtp/kk:mu_radial_offset", radial_func_count);
  MemKK::realloc_kokkos(d_mu_to_basic_group, "sus2mtp/kk:mu_to_basic_group", radial_func_count);
  MemKK::realloc_kokkos(d_grouped_radial_coeffs, "sus2mtp/kk:grouped_radial_coeffs", radial_func_count,
                        radial_basis_size);
  MemKK::realloc_kokkos(d_pair_to_table_index, "sus2mtp/kk:pair_to_table_index",
                        species_count * species_count);
  
  // Allocate separate arrays for shift_coeffs
  // CRITICAL FIX: These must be allocated to avoid NaN in energy calculation
  MemKK::realloc_kokkos(d_shift_coeffs, "sus2mtp/kk:shift_coeffs", species_count);
  
  // Allocate unified regression_coeffs array using size and offsets from CPU version
  // All offsets are directly taken from base class to ensure exact consistency
  MemKK::realloc_kokkos(d_regression_coeffs, "sus2mtp/kk:regression_coeffs", regression_coeffs_count);
  
  // Setup separate arrays for species_coeffs and linear_coeffs
  MemKK::realloc_kokkos(d_species_coeffs, "sus2mtp/kk:species_coeffs", species_count);
  MemKK::realloc_kokkos(d_linear_coeffs, "sus2mtp/kk:linear_coeffs", alpha_scalar_count);

  basic_mu_group_count = 0;
  for (int mu = 0; mu < radial_func_count; mu++) {
    bool found = false;
    for (int i = 0; i < alpha_index_basic_count; i++) {
      if (alpha_index_basic[i][0] == mu) {
        found = true;
        break;
      }
    }
    if (found) basic_mu_group_count++;
  }
  MemKK::realloc_kokkos(d_basic_mu_offsets, "sus2mtp/kk:basic_mu_offsets", basic_mu_group_count + 1);
  MemKK::realloc_kokkos(d_basic_mu_values, "sus2mtp/kk:basic_mu_values", basic_mu_group_count);
  MemKK::realloc_kokkos(d_basic_grouped_indices, "sus2mtp/kk:basic_grouped_indices",
                        alpha_index_basic_count);
  MemKK::realloc_kokkos(d_alpha_basic_mu_group, "sus2mtp/kk:alpha_basic_mu_group",
                        alpha_index_basic_count);
  
  // Set offset pointers to match CPU version (direct assignment, no calculation)
  shift_coeffs_offset = PairSUS2MTP::shift_coeffs_offset;
  scal_coeffs_offset = PairSUS2MTP::scal_coeffs_offset;
  radial_coeffs_offset = PairSUS2MTP::radial_coeffs_offset;
  // We need to init these as very small views to begin with because the user might specify a very large chunk_size which is much more than inum. We will resize these as needed in compute.
  MemKK::realloc_kokkos(d_moment_tensor_vals, "sus2mtp/kk:moment_tensor_vals", 1, alpha_moment_count);
  MemKK::realloc_kokkos(d_nbh_energy_ders_wrt_moments, "sus2mtp/kk:nbh_energy_ders_wrt_moments", 1,
                        alpha_moment_count);
  MemKK::realloc_kokkos(d_env_gate_values, "sus2mtp/kk:env_gate_values", 1);
  MemKK::realloc_kokkos(d_env_gate_der_prefactors, "sus2mtp/kk:env_gate_der_prefactors", 1);
  MemKK::realloc_kokkos(d_env_gate_rho_chain_values, "sus2mtp/kk:env_gate_rho_chain_values", 1);
  MemKK::realloc_kokkos(d_env_gate_activation_basic_vals,
                        "sus2mtp/kk:env_gate_activation_basic_vals", 1,
                        alpha_index_basic_count);

  //Declare host arrays
  auto h_alpha_index_basic = Kokkos::create_mirror_view(d_alpha_index_basic);
  auto h_alpha_index_times = Kokkos::create_mirror_view(d_alpha_index_times);
  auto h_alpha_moment_mapping = Kokkos::create_mirror_view(d_alpha_moment_mapping);
  auto h_species_coeffs = Kokkos::create_mirror_view(d_species_coeffs);
  auto h_linear_coeffs = Kokkos::create_mirror_view(d_linear_coeffs);
  
  // SUS2-MLIP host arrays
  auto h_mu_to_K = Kokkos::create_mirror_view(d_mu_to_K);
  auto h_mu_to_sigma = Kokkos::create_mirror_view(d_mu_to_sigma);
  auto h_k_to_mu_offsets = Kokkos::create_mirror_view(d_k_to_mu_offsets);
  auto h_grouped_mu = Kokkos::create_mirror_view(d_grouped_mu);
  auto h_mu_radial_offset = Kokkos::create_mirror_view(d_mu_radial_offset);
  auto h_mu_to_basic_group = Kokkos::create_mirror_view(d_mu_to_basic_group);
  auto h_grouped_radial_coeffs = Kokkos::create_mirror_view(d_grouped_radial_coeffs);
  auto h_basic_mu_offsets = Kokkos::create_mirror_view(d_basic_mu_offsets);
  auto h_basic_mu_values = Kokkos::create_mirror_view(d_basic_mu_values);
  auto h_basic_grouped_indices = Kokkos::create_mirror_view(d_basic_grouped_indices);
  auto h_alpha_basic_mu_group = Kokkos::create_mirror_view(d_alpha_basic_mu_group);
  auto h_pair_to_table_index = Kokkos::create_mirror_view(d_pair_to_table_index);
  auto h_regression_coeffs = Kokkos::create_mirror_view(d_regression_coeffs);
  auto h_shift_coeffs = Kokkos::create_mirror_view(d_shift_coeffs);

  //Populate the host arrays
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < alpha_index_basic_count; i++)
      h_alpha_index_basic(i, j) = alpha_index_basic[i][j];
    for (int i = 0; i < alpha_index_times_count; i++)
      h_alpha_index_times(i, j) = alpha_index_times[i][j];
  }
  for (int i = 0; i < alpha_scalar_count; i++) {
    h_alpha_moment_mapping(i) = alpha_moment_mapping[i];
    h_linear_coeffs(i) = linear_coeffs[i];
  }
  for (int i = 0; i < species_count; i++) h_species_coeffs(i) = species_coeffs[i];
  
  // SUS2-MLIP: Copy from base class
  for (int i = 0; i < radial_func_count; i++) {
    h_mu_to_K(i) = PairSUS2MTP::mu_to_K[i];
    h_mu_to_sigma(i) = PairSUS2MTP::mu_to_sigma[i];
    h_mu_radial_offset(i) = i * (radial_basis_size + species_count);
    h_mu_to_basic_group(i) = -1;
  }
  for (int k = 0; k <= K_scaling; k++) h_k_to_mu_offsets(k) = 0;
  for (int mu = 0; mu < radial_func_count; mu++) h_k_to_mu_offsets(h_mu_to_K(mu) + 1)++;
  for (int k = 0; k < K_scaling; k++) h_k_to_mu_offsets(k + 1) += h_k_to_mu_offsets(k);
  auto h_k_cursor = Kokkos::create_mirror_view(d_k_to_mu_offsets);
  for (int k = 0; k <= K_scaling; k++) h_k_cursor(k) = h_k_to_mu_offsets(k);
  for (int mu = 0; mu < radial_func_count; mu++) {
    int k = h_mu_to_K(mu);
    h_grouped_mu(h_k_cursor(k)++) = mu;
  }
  for (int grouped_idx = 0; grouped_idx < radial_func_count; grouped_idx++) {
    int mu = h_grouped_mu(grouped_idx);
    int offset_mu = h_mu_radial_offset(mu);
    for (int ri = 0; ri < radial_basis_size; ri++)
      h_grouped_radial_coeffs(grouped_idx, ri) =
          PairSUS2MTP::regression_coeffs[radial_coeffs_offset + offset_mu + ri];
  }
  int grouped_basic_cursor = 0;
  int basic_mu_cursor = 0;
  h_basic_mu_offsets(0) = 0;
  for (int mu = 0; mu < radial_func_count; mu++) {
    bool found = false;
    for (int i = 0; i < alpha_index_basic_count; i++) {
      if (alpha_index_basic[i][0] == mu) {
        if (!found) {
          h_basic_mu_values(basic_mu_cursor) = mu;
          h_mu_to_basic_group(mu) = basic_mu_cursor;
          found = true;
        }
        h_basic_grouped_indices(grouped_basic_cursor++) = i;
        h_alpha_basic_mu_group(i) = basic_mu_cursor;
      }
    }
    if (found) {
      basic_mu_cursor++;
      h_basic_mu_offsets(basic_mu_cursor) = grouped_basic_cursor;
    }
  }
  for (int i = 0; i < regression_coeffs_count; i++) {
    h_regression_coeffs(i) = PairSUS2MTP::regression_coeffs[i];
  }
  for (int i = 0; i < species_count * species_count; i++) {
    h_pair_to_table_index(i) = PairSUS2MTP::pair_to_table_index[i];
  }
  
  // CRITICAL FIX: Copy shift_coeffs from CPU base class
  // shift_coeffs are at the beginning of regression_coeffs array
  for (int i = 0; i < species_count; i++) {
    h_shift_coeffs(i) = PairSUS2MTP::regression_coeffs[shift_coeffs_offset + i];
  }

  // Peform the copy from host to device
  Kokkos::deep_copy(d_alpha_index_basic, h_alpha_index_basic);
  Kokkos::deep_copy(d_alpha_index_times, h_alpha_index_times);
  Kokkos::deep_copy(d_alpha_moment_mapping, h_alpha_moment_mapping);
  Kokkos::deep_copy(d_species_coeffs, h_species_coeffs);
  Kokkos::deep_copy(d_linear_coeffs, h_linear_coeffs);
  
  // SUS2-MLIP: Copy to device
  Kokkos::deep_copy(d_mu_to_K, h_mu_to_K);
  Kokkos::deep_copy(d_mu_to_sigma, h_mu_to_sigma);
  Kokkos::deep_copy(d_k_to_mu_offsets, h_k_to_mu_offsets);
  Kokkos::deep_copy(d_grouped_mu, h_grouped_mu);
  Kokkos::deep_copy(d_mu_radial_offset, h_mu_radial_offset);
  Kokkos::deep_copy(d_mu_to_basic_group, h_mu_to_basic_group);
  Kokkos::deep_copy(d_grouped_radial_coeffs, h_grouped_radial_coeffs);
  Kokkos::deep_copy(d_basic_mu_offsets, h_basic_mu_offsets);
  Kokkos::deep_copy(d_basic_mu_values, h_basic_mu_values);
  Kokkos::deep_copy(d_basic_grouped_indices, h_basic_grouped_indices);
  Kokkos::deep_copy(d_alpha_basic_mu_group, h_alpha_basic_mu_group);
  Kokkos::deep_copy(d_pair_to_table_index, h_pair_to_table_index);
  Kokkos::deep_copy(d_regression_coeffs, h_regression_coeffs);
  Kokkos::deep_copy(d_shift_coeffs, h_shift_coeffs);  // CRITICAL FIX: Copy shift_coeffs
  
  // No need to deep copy the working buffers.
  build_preinterpolation_table();
}

template <class DeviceType>
void PairSUS2MTPKokkos<DeviceType>::build_preinterpolation_table()
{
  // Check if preinterpolation table should be used
  if (radial_basis_type_index != SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP &&
      radial_basis_type_index != SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP &&
      radial_basis_type_index != SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP &&
      radial_basis_type_index != SUS2RadialMTPBasis::JACOBI_SSS_LMP) {
    do_list = false;
    return;
  }

  do_list = true;
  inv_dr = PairSUS2MTP::inv_dr;
  double dr = actual_dr;

  int C = species_count;
  int R = radial_basis_size;
  int K_ = K_scaling;
  int pairs_count = C * C;
  int species_pairs = used_pair_count;

  // Allocate device-side memory first
  d_radial_list = decltype(d_radial_list)("device:radial_list", species_pairs, list_grid_size,
                                          basic_mu_group_count);
  d_radial_der_list = decltype(d_radial_der_list)("device:radial_der_list", species_pairs,
                                                  list_grid_size, basic_mu_group_count);

  // Create host-side mirrors that are compatible with device layout
  auto h_radial_list = Kokkos::create_mirror_view(d_radial_list);
  auto h_radial_der_list = Kokkos::create_mirror_view(d_radial_der_list);
  auto h_basic_mu_values = Kokkos::create_mirror_view(d_basic_mu_values);
  Kokkos::deep_copy(h_basic_mu_values, d_basic_mu_values);

  // Initialize to zero
  Kokkos::deep_copy(h_radial_list, 0.0);
  Kokkos::deep_copy(h_radial_der_list, 0.0);

  // Build preinterpolation table
  for (int i = 0; i < C; i++) {                              // Central atom type
    for (int j = 0; j < C; j++) {                          // Neighbor atom type
      int shift = i * C + j;  // Species pair index
      int table_index = pair_to_table_index[shift];
      if (table_index < 0) continue;

      for (int n = 0; n < list_grid_size; n++) {           // Distance points
        double dist = dr * n;

        for (int mu_group = 0; mu_group < basic_mu_group_count; mu_group++) {
          const int mu = h_basic_mu_values(mu_group);

          // 1. Get k_ value
          int k_ = mu_to_K[mu];
          int sigma = mu_to_sigma[mu];

          // 2. Calculate radial basis function values and derivatives
          // Use CPU version's radial_basis object
          double scal = regression_coeffs[C + 2*k_*pairs_count + C*i + j];
          double s = regression_coeffs[C + 2*k_*pairs_count + pairs_count + C*i + j];

          radial_basis->calc_radial_basis_ders(dist, scal, s, sigma);

          // 3. Apply coefficients and accumulate to preinterpolation table
          for (int xi = 0; xi < R; xi++) {
            double factor = regression_coeffs[C + 2*pairs_count*K_ + mu*(R+C) + xi] *
                           regression_coeffs[C + 2*pairs_count*K_ + R + i] *
                           regression_coeffs[C + 2*pairs_count*K_ + R + j];

            h_radial_list(table_index, n, mu_group) += radial_basis->radial_basis_vals[xi] * factor;
            h_radial_der_list(table_index, n, mu_group) += radial_basis->radial_basis_ders[xi] * factor;
          }
        }
      }
    }
  }

  // Copy to device (now layouts are compatible)
  Kokkos::deep_copy(d_radial_list, h_radial_list);
  Kokkos::deep_copy(d_radial_der_list, h_radial_der_list);

  if (env_gate_enabled) {
    d_env_gate_rho_list =
        decltype(d_env_gate_rho_list)("device:env_gate_rho_list", C, list_grid_size);
    d_env_gate_rho_der_list =
        decltype(d_env_gate_rho_der_list)("device:env_gate_rho_der_list", C, list_grid_size);
    auto h_env_gate_rho_list = Kokkos::create_mirror_view(d_env_gate_rho_list);
    auto h_env_gate_rho_der_list = Kokkos::create_mirror_view(d_env_gate_rho_der_list);
    Kokkos::deep_copy(h_env_gate_rho_list, 0.0);
    Kokkos::deep_copy(h_env_gate_rho_der_list, 0.0);

    const double r_env = env_gate_cutoff_ratio * max_cutoff;
    for (int i = 0; i < C; i++) {
      double density_coeffs[6];
      for (int q = 0; q < env_gate_channel_count; ++q) {
        const double raw = regression_coeffs[env_gate_log_density_coeffs_offset + i * env_gate_channel_count + q];
        const double capped = raw < static_cast<double>(kEnvGateMaxLogDensityCoeff)
                                  ? raw
                                  : static_cast<double>(kEnvGateMaxLogDensityCoeff);
        density_coeffs[q] = std::exp(capped);
      }
      for (int n = 0; n < list_grid_size; n++) {
        const double dist = dr * n;
        if (dist <= 0.0 || dist >= r_env) continue;
        const double y = dist / r_env;
        const double dy_dr = 1.0 / r_env;
        const double cutoff = 1.0 - dist / r_env;
        const double f_env = cutoff * cutoff;
        const double df_dr = -2.0 * cutoff / r_env;
        KK_FLOAT basis[6];
        KK_FLOAT basis_der[6];
        bernstein_degree5_kk(static_cast<KK_FLOAT>(y), basis, basis_der);
        for (int q = 0; q < env_gate_channel_count; ++q) {
          const double weighted_basis = f_env * basis[q];
          const double weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
          h_env_gate_rho_list(i, n) += density_coeffs[q] * weighted_basis;
          h_env_gate_rho_der_list(i, n) += density_coeffs[q] * weighted_basis_der;
        }
      }
    }
    Kokkos::deep_copy(d_env_gate_rho_list, h_env_gate_rho_list);
    Kokkos::deep_copy(d_env_gate_rho_der_list, h_env_gate_rho_der_list);
  }

  // Log information
  if (comm->me == 0) {
    utils::logmesg(lmp, "Preinterpolation table built successfully:\n");
    utils::logmesg(
        lmp,
        "  - requested_tabstep={} A, actual_dr={} A, grid_points={}, max_cutoff={} A\n",
        requested_tabstep, actual_dr, list_grid_size, max_cutoff);
    utils::logmesg(lmp, "  - Table size: {} species pairs x {} grid points x {} radial functions\n",
                   species_pairs, list_grid_size, basic_mu_group_count);
    utils::logmesg(lmp, "  - Memory used: {:.2f} MB\n",
                   (double)(species_pairs * list_grid_size * basic_mu_group_count * 2 *
                            sizeof(typename PairSUS2MTPKokkos<DeviceType>::radial_table_value_type)) /
                       (1024 * 1024));
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION KK_FLOAT PairSUS2MTPKokkos<DeviceType>::int_pow(KK_FLOAT base,
                                                                       int exp) const
{
  KK_FLOAT result = 1.0;
  for (int i = 0; i < exp; i++) result *= base;
  return result;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::eval_radial_basic_mu_group(
    const int itype, const int jtype, const KK_FLOAT dist, const int mu_group, KK_FLOAT &val,
    KK_FLOAT &der) const
{
  const int mu = d_basic_mu_values(mu_group);
  if (do_list) {
    int r_list = (int) Kokkos::floor(dist * inv_dr);
    const int last_interval = list_grid_size - 2;
    if (r_list < 0) r_list = 0;
    if (r_list > last_interval) r_list = last_interval;
    const int r_next = r_list + 1;
    const int shift = itype * species_count + jtype;
    const int table_index = d_pair_to_table_index(shift);
    if (table_index >= 0) {
      F_FLOAT ddr = dist * inv_dr - r_list;
      if (ddr < 0.0) ddr = 0.0;
      if (ddr > 1.0) ddr = 1.0;

      const F_FLOAT v1 = d_radial_list(table_index, r_list, mu_group);
      const F_FLOAT v2 = d_radial_list(table_index, r_next, mu_group);
      const F_FLOAT d1 = d_radial_der_list(table_index, r_list, mu_group);
      const F_FLOAT d2 = d_radial_der_list(table_index, r_next, mu_group);
      val = v1 + ddr * (v2 - v1);
      der = d1 + ddr * (d2 - d1);
      return;
    }
  }

  const int pairs_count = species_count * species_count;
  const int k_scaling = d_mu_to_K(mu);
  const int scal_idx = species_count + k_scaling * 2 * pairs_count + itype * species_count + jtype;
  const int s_idx =
      species_count + k_scaling * 2 * pairs_count + pairs_count + itype * species_count + jtype;
  const int coeff_offset = radial_coeffs_offset + d_mu_radial_offset(mu);

  const F_FLOAT scal = d_regression_coeffs[scal_idx];
  const F_FLOAT s_param = d_regression_coeffs[s_idx];
  const F_FLOAT species_factor =
      d_regression_coeffs[radial_coeffs_offset + radial_basis_size + itype] *
      d_regression_coeffs[radial_coeffs_offset + radial_basis_size + jtype];
  const bool use_laguerre =
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P ||
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP ||
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS ||
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP;
  const bool use_laguerre_pos =
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS ||
      radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP;

  if (use_laguerre) {
    const F_FLOAT scal_eff = use_laguerre_pos ? laguerre_positive_param_kk(scal) : scal;
    const F_FLOAT s_eff = use_laguerre_pos ? laguerre_positive_param_kk(s_param) : s_param;
    const F_FLOAT rho = (s_eff > 1.0e-8) ? s_eff : static_cast<F_FLOAT>(1.0e-8);
    const F_FLOAT u = scal_eff * Kokkos::log(1.0 + dist / rho);
    const F_FLOAT u_r = scal_eff / (rho + dist);
    const F_FLOAT Dr = dist - max_cutoff;
    const F_FLOAT cutoff = Dr * Dr;
    const F_FLOAT cutoff_der = 2.0 * Dr;
    const F_FLOAT exp_factor = Kokkos::exp(-0.5 * u);

    F_FLOAT phi_prev = 0.0;
    F_FLOAT dphi_prev = 0.0;
    F_FLOAT phi_curr = scaling * cutoff * exp_factor;
    F_FLOAT dphi_curr = scaling * cutoff_der * exp_factor - 0.5 * u_r * phi_curr;

    val = d_regression_coeffs[coeff_offset] * species_factor * phi_curr;
    der = d_regression_coeffs[coeff_offset] * species_factor * dphi_curr;

    for (int ri = 1; ri < radial_basis_size; ri++) {
      const F_FLOAT inv_ri = 1.0 / static_cast<F_FLOAT>(ri);
      const F_FLOAT coeff = (2.0 * static_cast<F_FLOAT>(ri) - 1.0 - u) * inv_ri;
      const F_FLOAT prev_coeff = static_cast<F_FLOAT>(ri - 1) * inv_ri;
      const F_FLOAT phi_next = coeff * phi_curr - prev_coeff * phi_prev;
      const F_FLOAT dphi_next = -u_r * inv_ri * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;
      const F_FLOAT basis_coeff = d_regression_coeffs[coeff_offset + ri] * species_factor;

      val = Kokkos::fma(basis_coeff, phi_next, val);
      der = Kokkos::fma(basis_coeff, dphi_next, der);
      phi_prev = phi_curr;
      dphi_prev = dphi_curr;
      phi_curr = phi_next;
      dphi_curr = dphi_next;
    }
  } else {
    const F_FLOAT transformed = scal * (dist - s_param) / 2.0;
    const F_FLOAT ksi = Kokkos::tanh(transformed);
    const F_FLOAT Dr = dist - max_cutoff;
    const F_FLOAT cutoff = Dr * Dr;
    const F_FLOAT der_ksi = 1.0 - ksi * ksi;
    const F_FLOAT mult = der_ksi * scal / 2.0;

    F_FLOAT basis_prev2 = scaling * cutoff;
    F_FLOAT der_prev2 = 2.0 * Dr * scaling;

    val = d_regression_coeffs[coeff_offset] * species_factor * basis_prev2;
    der = d_regression_coeffs[coeff_offset] * species_factor * der_prev2;

    if (radial_basis_size == 1) return;

    F_FLOAT basis_prev1 = scaling * ksi * cutoff;
    F_FLOAT der_prev1 = scaling * (mult * cutoff + 2.0 * ksi * Dr);

    val += d_regression_coeffs[coeff_offset + 1] * species_factor * basis_prev1;
    der += d_regression_coeffs[coeff_offset + 1] * species_factor * der_prev1;

    for (int ri = 2; ri < radial_basis_size; ri++) {
      const F_FLOAT basis = Kokkos::fma(2.0f * ksi, basis_prev1, -basis_prev2);
      const F_FLOAT der_basis =
          Kokkos::fma(2.0f, Kokkos::fma(mult, basis_prev1, ksi * der_prev1), -der_prev2);
      const F_FLOAT coeff = d_regression_coeffs[coeff_offset + ri] * species_factor;
      val = Kokkos::fma(coeff, basis, val);
      der = Kokkos::fma(coeff, der_basis, der);
      basis_prev2 = basis_prev1;
      basis_prev1 = basis;
      der_prev2 = der_prev1;
      der_prev1 = der_basis;
    }
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::eval_env_gate_weighted_basis(
    const int itype, const int jtype, const KK_FLOAT dist, KK_FLOAT *weighted_basis,
    KK_FLOAT *weighted_basis_der) const
{
  if (do_list) {
    int r_list = (int) Kokkos::floor(dist * inv_dr);
    const int last_interval = list_grid_size - 2;
    if (r_list < 0) r_list = 0;
    if (r_list > last_interval) r_list = last_interval;
    const int r_next = r_list + 1;
    const int shift = itype * species_count + jtype;
    const int table_index = d_pair_to_table_index(shift);
    if (table_index >= 0) {
      KK_FLOAT ddr = dist * inv_dr - r_list;
      if (ddr < static_cast<KK_FLOAT>(0.0)) ddr = static_cast<KK_FLOAT>(0.0);
      if (ddr > static_cast<KK_FLOAT>(1.0)) ddr = static_cast<KK_FLOAT>(1.0);
      for (int q = 0; q < kEnvGateChannels; ++q) {
        const KK_FLOAT v1 = d_env_gate_radial_list(table_index, r_list, q);
        const KK_FLOAT v2 = d_env_gate_radial_list(table_index, r_next, q);
        const KK_FLOAT d1 = d_env_gate_radial_der_list(table_index, r_list, q);
        const KK_FLOAT d2 = d_env_gate_radial_der_list(table_index, r_next, q);
        weighted_basis[q] = v1 + ddr * (v2 - v1);
        weighted_basis_der[q] = d1 + ddr * (d2 - d1);
      }
      return;
    }
  }

  const KK_FLOAT r_env = env_gate_cutoff_ratio * max_cutoff;
  const KK_FLOAT y = dist / r_env;
  const KK_FLOAT dy_dr = static_cast<KK_FLOAT>(1.0) / r_env;
  const KK_FLOAT cutoff = static_cast<KK_FLOAT>(1.0) - dist / r_env;
  const KK_FLOAT f_env = cutoff * cutoff;
  const KK_FLOAT df_dr = static_cast<KK_FLOAT>(-2.0) * cutoff / r_env;
  KK_FLOAT basis[6];
  KK_FLOAT basis_der[6];
  bernstein_degree5_kk(y, basis, basis_der);
  for (int q = 0; q < kEnvGateChannels; ++q) {
    weighted_basis[q] = f_env * basis[q];
    weighted_basis_der[q] = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::eval_env_gate_rho(
    const int itype, const KK_FLOAT dist, KK_FLOAT &rho, KK_FLOAT &rho_der) const
{
  rho = static_cast<KK_FLOAT>(0.0);
  rho_der = static_cast<KK_FLOAT>(0.0);
  const KK_FLOAT r_env = env_gate_cutoff_ratio * max_cutoff;
  if (dist <= static_cast<KK_FLOAT>(0.0) || dist >= r_env) return;

  if (do_list && d_env_gate_rho_list.extent(0) > 0) {
    int r_list = (int) Kokkos::floor(dist * inv_dr);
    const int last_interval = list_grid_size - 2;
    if (r_list < 0) r_list = 0;
    if (r_list > last_interval) r_list = last_interval;
    const int r_next = r_list + 1;
    KK_FLOAT ddr = dist * inv_dr - r_list;
    if (ddr < static_cast<KK_FLOAT>(0.0)) ddr = static_cast<KK_FLOAT>(0.0);
    if (ddr > static_cast<KK_FLOAT>(1.0)) ddr = static_cast<KK_FLOAT>(1.0);
    const KK_FLOAT v1 = d_env_gate_rho_list(itype, r_list);
    const KK_FLOAT v2 = d_env_gate_rho_list(itype, r_next);
    const KK_FLOAT d1 = d_env_gate_rho_der_list(itype, r_list);
    const KK_FLOAT d2 = d_env_gate_rho_der_list(itype, r_next);
    rho = v1 + ddr * (v2 - v1);
    rho_der = d1 + ddr * (d2 - d1);
    return;
  }

  const KK_FLOAT y = dist / r_env;
  const KK_FLOAT dy_dr = static_cast<KK_FLOAT>(1.0) / r_env;
  const KK_FLOAT cutoff = static_cast<KK_FLOAT>(1.0) - dist / r_env;
  const KK_FLOAT f_env = cutoff * cutoff;
  const KK_FLOAT df_dr = static_cast<KK_FLOAT>(-2.0) * cutoff / r_env;
  KK_FLOAT basis[6];
  KK_FLOAT basis_der[6];
  bernstein_degree5_kk(y, basis, basis_der);
  for (int q = 0; q < kEnvGateChannels; ++q) {
    const KK_FLOAT density_coeff = env_gate_density_coeff_kk(
        d_regression_coeffs[env_gate_log_density_coeffs_offset + itype * env_gate_channel_count + q]);
    const KK_FLOAT weighted_basis = f_env * basis[q];
    const KK_FLOAT weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
    rho += density_coeff * weighted_basis;
    rho_der += density_coeff * weighted_basis_der;
  }
}

// Finds the maximum number of neighbours in all neigbhourhoods. This enables use to set the size (2nd index) of the jacobian. (Copied from other potentials)
template <class DeviceType> struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType> *nl) : k_list(*nl) {}
  ~FindMaxNumNeighs() { k_list.copymode = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &ii, int &max_neighs) const
  {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh(i);
    if (max_neighs < num_neighs) max_neighs = num_neighs;
  }
};

/* ----------------------------------------------------------------------
   This version is a straightforward implementation
   ---------------------------------------------------------------------- */

template <class DeviceType> void PairSUS2MTPKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  if (host_flag) {
    atomKK->sync(Host, X_MASK | F_MASK | TYPE_MASK);
    PairSUS2MTP::compute(eflag_in, vflag_in);
    atomKK->modified(Host, F_MASK);
    return;
  }

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }
  if (cvflag_atom) {
    memoryKK->destroy_kokkos(k_cvatom, cvatom);
    memoryKK->create_kokkos(k_cvatom, cvatom, maxcvatom, "pair:cvatom");
    d_cvatom = k_cvatom.view<DeviceType>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false) error->all(FLERR, "PairSUS2MTPKokkos requires 'newton on'.");

  // Now, ensure the atom data is synced
  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  need_dup = lmp->kokkos->need_dup<DeviceType>();
  // clang-format off
  if (need_dup) {
    dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
    dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
    if (cvflag_atom)
      dup_cvatom =
          Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                    Kokkos::Experimental::ScatterDuplicated>(d_cvatom);
  } else {
    ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
    ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
    if (cvflag_atom)
      ndup_cvatom =
          Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                    Kokkos::Experimental::ScatterNonDuplicated>(d_cvatom);
    // clang-format on
  }

  // Handling batching
  chunk_size =    // chunk_size is the working chunk size and may change per compute pass
      MIN(input_chunk_size,
          inum);    // chunksize is the maximum atoms per pass as defined by the user
  chunk_offset = 0;

  // Team sizes. We specify 32 for 1 warp per thread block.
  int team_size_default = 1;
  int vector_length_default = 1;
  if (!host_flag) team_size_default = 32;
  int alpha_basic_team_size = team_size_default;
  if (!host_flag) alpha_basic_team_size = 16;
  int alpha_basic_vector_length = vector_length_default;
  if (!host_flag) alpha_basic_vector_length = 4;

  // Resize the arrays to the chunksize if needed. Do not initialize values, we do so in the loop.
  if ((int) d_moment_tensor_vals.extent(0) < chunk_size) {
    Kokkos::realloc(Kokkos::WithoutInitializing, d_moment_tensor_vals, chunk_size,
                    alpha_moment_count);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_nbh_energy_ders_wrt_moments, chunk_size,
                    alpha_moment_count);
  }
  if (env_gate_enabled && (int) d_env_gate_values.extent(0) < chunk_size) {
    Kokkos::realloc(Kokkos::WithoutInitializing, d_env_gate_values, chunk_size);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_env_gate_der_prefactors, chunk_size);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_env_gate_rho_chain_values, chunk_size);
    Kokkos::realloc(Kokkos::WithoutInitializing, d_env_gate_activation_basic_vals, chunk_size,
                    alpha_index_basic_count);
  }

  EV_FLOAT ev;

  // ========== Begin Main Computation ==========
  while (chunk_offset < inum) {    // batching to prevent OOM on device
    EV_FLOAT ev_tmp;
    if (chunk_size > inum - chunk_offset) chunk_size = inum - chunk_offset;
    // ========== Init working views as 0  ==========
    {
      // These working buffers are dense and zero-filled every chunk, so let Kokkos map the
      // reset to the backend's bulk memset/deep-copy path instead of launching a custom kernel.
      Kokkos::deep_copy(d_moment_tensor_vals, moment_buffer_value_type(0));
      Kokkos::deep_copy(d_nbh_energy_ders_wrt_moments, nbh_der_buffer_value_type(0));
      if (env_gate_enabled)
        Kokkos::deep_copy(d_env_gate_activation_basic_vals, moment_buffer_value_type(0));
    }

    if (env_gate_enabled) {
      typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeEnvGate> policy_env_gate(
          0, chunk_size);
      Kokkos::parallel_for("ComputeEnvGate", policy_env_gate, *this);
    }

    // ========== Calculate the basic alphas ==========
    {
      int team_size = alpha_basic_team_size;
      int vector_length = alpha_basic_vector_length;
      int team_count = chunk_size;
      check_team_size_for<TagPairSUS2MTPComputeAlphaBasic>(team_count, team_size, vector_length);
      
      int radial_scratch_count = basic_mu_group_count;
      int dist_scratch_count = max_alpha_index_basic;           // s_dist_powers
      int coord_scratch_count = 3 * max_alpha_index_basic;      // s_coord_powers (3 dimensions)
      int scratch_size = scratch_size_helper<F_FLOAT>(
          team_size * (radial_scratch_count + dist_scratch_count + coord_scratch_count));
      Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeAlphaBasic> policy_basic_alpha(team_count,
                                                                                     team_size,
                                                                                     vector_length);
      policy_basic_alpha = policy_basic_alpha.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
      Kokkos::parallel_for("ComputeAlphaBasic", policy_basic_alpha, *this);
    }

    // ========== Calculate the non-elementary alphas  ==========
    {
      typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeAlphaTimes> policy_times(
          0, chunk_size);
      Kokkos::parallel_for("ComputeAlphaTimes", policy_times, *this);
    }
    // ========== Calc the nbh ders wrt moments ==========
    {
      typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeNbhDers> policy_nbh_calc(
          0, chunk_size);
      Kokkos::parallel_for("ComputeNbhDers", policy_nbh_calc, *this);
    }
    if (env_gate_enabled) {
      typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeEnvRhoChain>
          policy_env_rho_chain(0, chunk_size);
      Kokkos::parallel_for("ComputeEnvRhoChain", policy_env_rho_chain, *this);
    }
    // ========== Compute force (and dot product with alphas to get energy if needed) ==========
    {
      if (evflag) {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeForce<HALF, 1>> policy_force(
              0, chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeForce<HALFTHREAD, 1>>
              policy_force(0, chunk_size);
          Kokkos::parallel_reduce(policy_force, *this, ev_tmp);
        }
      } else {
        if (neighflag == HALF) {
          typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeForce<HALF, 0>> policy_force(
              0, chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        } else if (neighflag == HALFTHREAD) {
          typename Kokkos::RangePolicy<DeviceType, TagPairSUS2MTPComputeForce<HALFTHREAD, 0>>
              policy_force(0, chunk_size);
          Kokkos::parallel_for(policy_force, *this);
        }
      }
    }
    ev += ev_tmp;
    chunk_offset += chunk_size;    // Manage halt condition
  }    // end batching while loop

  // ========== End Main Computation ==========

  if (need_dup) Kokkos::Experimental::contribute(f, dup_f);

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_vatom, dup_vatom);
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }
  if (cvflag_atom) {
    if (need_dup) Kokkos::Experimental::contribute(d_cvatom, dup_cvatom);
    k_cvatom.template modify<DeviceType>();
    k_cvatom.template sync<LMPHostType>();
  }

  atomKK->modified(execution_space, F_MASK);

  copymode = 0;

  // free duplicated memory
  if (need_dup) {
    dup_f = decltype(dup_f)();
    dup_vatom = decltype(dup_vatom)();
    dup_cvatom = decltype(dup_cvatom)();
  }
}

// Device-side radial basic-mu evaluation for non-table execution paths.
// The helper supports the existing Chebyshev-sss path and the Laguerre-log1p basis.

// ========== Kernels ==========

// Inits the working arrays: moment and ders, jacobian not needed.
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(TagPairSUS2MTPInitMomentValsDers,
                                                                  const int &ii, const int &k) const
{
  d_moment_tensor_vals(ii, k) = 0;
  d_nbh_energy_ders_wrt_moments(ii, k) = 0;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPComputeEnvGate, const int &ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i] - 1;
  const int jnum = d_numneigh(i);
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const F_FLOAT r_env = env_gate_cutoff_ratio * max_cutoff;
  const F_FLOAT r_env_sq = r_env * r_env;
  F_FLOAT rho = 0.0;

  for (int jj = 0; jj < jnum; jj++) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = Kokkos::fma(r[0], r[0], Kokkos::fma(r[1], r[1], r[2] * r[2]));
    if (rsq <= static_cast<F_FLOAT>(0.0) || rsq >= r_env_sq) continue;
    const F_FLOAT dist = Kokkos::sqrt(rsq);
    F_FLOAT local_rho = 0.0;
    F_FLOAT local_rho_der = 0.0;
    eval_env_gate_rho(itype, dist, local_rho, local_rho_der);
    rho += local_rho;
  }

  const F_FLOAT lambda = stable_sigmoid_kk(d_regression_coeffs[env_gate_lambda_raw_offset + itype]);
  const F_FLOAT tanh_rho = Kokkos::tanh(rho);
  const F_FLOAT sech2_rho = static_cast<F_FLOAT>(1.0) - tanh_rho * tanh_rho;
  d_env_gate_values(ii) = lambda * tanh_rho;
  d_env_gate_der_prefactors(ii) = -lambda * sech2_rho;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPApplyEnvGate, const int &ii) const
{
  (void) ii;
  // Pair-distance env-gate is applied inside ComputeAlphaBasic. Keep this tag as a
  // compatibility no-op so older launch sequencing stays stable.
}

// Calculates the basic alphas using fused operations where possible
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPComputeAlphaBasic,
    const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeAlphaBasic>::member_type &team)
    const
{
  shared_double_2d s_radial_vals(team.team_scratch(0), team.team_size(), basic_mu_group_count);
  shared_double_2d s_dist_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic);
  shared_double_3d s_coord_powers(team.team_scratch(0), team.team_size(), max_alpha_index_basic);

  const int ii = team.league_rank();
  if (ii >= chunk_size) return;

  const int i = d_ilist[ii + chunk_offset];
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const int itype = type[i] - 1;    // switch to zero indexing
  const int jnum = d_numneigh(i);
  const int thread = team.team_rank();

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, jnum), [&](const int jj) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const int jtype = type[j] - 1;    // switch to zero indexing
    const F_FLOAT r[3] = {Kokkos::fma(-1.0, xi[0], x(j, 0)), Kokkos::fma(-1.0, xi[1], x(j, 1)),
                          Kokkos::fma(-1.0, xi[2], x(j, 2))};
    const F_FLOAT rsq = Kokkos::fma(r[0], r[0], Kokkos::fma(r[1], r[1], r[2] * r[2]));

    if (rsq >= max_cutoff_sq) return;
    const F_FLOAT dist = Kokkos::sqrt(rsq);
    bool used_precomputed_table = false;
    if (do_list) {
      // Use preinterpolation table with linear interpolation
      int r_list = (int) Kokkos::floor(dist * inv_dr);
      const int last_interval = list_grid_size - 2;
      if (r_list < 0) r_list = 0;
      if (r_list > last_interval) r_list = last_interval;
      int r_next = r_list + 1;
      int shift = itype * species_count + jtype;  // Species pair index
      int table_index = d_pair_to_table_index(shift);
      if (table_index >= 0) {
        double ddr = dist * inv_dr - r_list;
        if (ddr < 0.0) ddr = 0.0;
        if (ddr > 1.0) ddr = 1.0;

        for (int mu_group = 0; mu_group < basic_mu_group_count; mu_group++) {
          const F_FLOAT v1 = d_radial_list(table_index, r_list, mu_group);
          const F_FLOAT v2 = d_radial_list(table_index, r_next, mu_group);
          s_radial_vals(thread, mu_group) = v1 + ddr * (v2 - v1);
        }
        used_precomputed_table = true;
      }
    }
    if (!used_precomputed_table) {
      for (int mu_group = 0; mu_group < basic_mu_group_count; mu_group++) {
        F_FLOAT der_unused = 0.0;
        eval_radial_basic_mu_group(itype, jtype, dist, mu_group, s_radial_vals(thread, mu_group),
                                   der_unused);
      }
    }

    F_FLOAT env_activation = 0.0;
    F_FLOAT env_activation_der = 0.0;
    F_FLOAT env_pair_gate = 1.0;
    if (env_gate_enabled) {
      env_gate_activation_kk(dist, env_gate_cutoff_ratio * max_cutoff,
                             env_gate_activation_on_ratio, env_activation,
                             env_activation_der);
      (void) env_activation_der;
      env_pair_gate = static_cast<F_FLOAT>(1.0) - d_env_gate_values(ii) * env_activation;
    }

    s_dist_powers(thread, 0) = 1.0;
    s_coord_powers(thread, 0, 0) = 1.0;
    s_coord_powers(thread, 0, 1) = 1.0;
    s_coord_powers(thread, 0, 2) = 1.0;

    for (int k = 1; k < max_alpha_index_basic; k++) {
      s_dist_powers(thread, k) = s_dist_powers(thread, k - 1) * dist;
      for (int a = 0; a < 3; a++)
        s_coord_powers(thread, k, a) = s_coord_powers(thread, k - 1, a) * r[a];
    }

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, alpha_index_basic_count), [&](const int k) {
      int mu_group = d_alpha_basic_mu_group(k);
      int a0 = d_alpha_index_basic(k, 1);
      int a1 = d_alpha_index_basic(k, 2);
      int a2 = d_alpha_index_basic(k, 3);

      F_FLOAT val = s_radial_vals(thread, mu_group);
      int norm_rank = a0 + a1 + a2;
      F_FLOAT norm_fac = 1.0 / s_dist_powers(thread, norm_rank);
      val *= norm_fac;

      F_FLOAT pow0 = s_coord_powers(thread, a0, 0);
      F_FLOAT pow1 = s_coord_powers(thread, a1, 1);
      F_FLOAT pow2 = s_coord_powers(thread, a2, 2);
      F_FLOAT pow = pow0 * pow1 * pow2;
      const F_FLOAT raw_contrib = val * pow;
      if (env_gate_enabled) {
        Kokkos::atomic_add(&d_moment_tensor_vals(ii, k), env_pair_gate * raw_contrib);
        Kokkos::atomic_add(&d_env_gate_activation_basic_vals(ii, k),
                           env_activation * raw_contrib);
      } else {
        Kokkos::atomic_add(&d_moment_tensor_vals(ii, k), raw_contrib);
      }

    });
  });
}

// Calculates the non-elementary alpha from the basic alphas
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(TagPairSUS2MTPComputeAlphaTimes,
                                                                  const int &ii) const
{
  // Traverse all edges in the alpha times compute graph
  for (int k = 0; k < alpha_index_times_count; k++) {
    int a0 = d_alpha_index_times(k, 0);
    int a1 = d_alpha_index_times(k, 1);
    int mult = d_alpha_index_times(k, 2);
    int a3 = d_alpha_index_times(k, 3);

    F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
    F_FLOAT val1 = d_moment_tensor_vals(ii, a1);

    d_moment_tensor_vals(ii, a3) += mult * val0 * val1;
  }
}

// Sets the nbh energy ders as the linear coeffs
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(TagPairSUS2MTPSetScalarNbhDers,
                                                                  const int &ii, const int &k) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i] - 1;
  d_nbh_energy_ders_wrt_moments(ii, d_alpha_moment_mapping(k)) =
      d_linear_coeffs(k) * d_species_coeffs[itype];
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(TagPairSUS2MTPComputeNbhDers,
                                                                  const int &ii) const
{
  const int i = d_ilist[ii + chunk_offset];
  const int itype = type[i] - 1;
  const F_FLOAT species_coeff = d_species_coeffs[itype];

  for (int k = 0; k < alpha_scalar_count; k++) {
    d_nbh_energy_ders_wrt_moments(ii, d_alpha_moment_mapping(k)) = d_linear_coeffs(k) * species_coeff;
  }

  for (int k = alpha_index_times_count - 1; k >= 0; k--) {
    int a0 = d_alpha_index_times(k, 0);
    int a1 = d_alpha_index_times(k, 1);
    int mult = d_alpha_index_times(k, 2);
    int a3 = d_alpha_index_times(k, 3);

    F_FLOAT val0 = d_moment_tensor_vals(ii, a0);
    F_FLOAT val1 = d_moment_tensor_vals(ii, a1);
    F_FLOAT val3 = d_nbh_energy_ders_wrt_moments(ii, a3);

    d_nbh_energy_ders_wrt_moments(ii, a1) += val3 * mult * val0;
    d_nbh_energy_ders_wrt_moments(ii, a0) += val3 * mult * val1;
  }
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPComputeEnvRhoChain, const int &ii) const
{
  F_FLOAT dot = 0.0;
  for (int k = 0; k < alpha_index_basic_count; k++) {
    dot += d_nbh_energy_ders_wrt_moments(ii, k) * d_env_gate_activation_basic_vals(ii, k);
  }
  d_env_gate_rho_chain_values(ii) = d_env_gate_der_prefactors(ii) * dot;
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPComputeForce<NEIGHFLAG, EVFLAG>,
    const int &ii,
    EV_FLOAT &ev) const
{
  // The f array is duplicated for OpenMP, atomic for GPU, and neither for Serial
  auto v_f =
      ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_f), decltype(ndup_f)>::get(
          dup_f, ndup_f);
  auto a_f = v_f.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  const int i = d_ilist[ii + chunk_offset];
  const int jnum = d_numneigh(i);
  const bool need_energy_tally = EVFLAG && eflag_either;
  const bool need_virial_tally = EVFLAG && (vflag_global || vflag_atom || cvflag_atom);
  
  // FIXED: Cache itype and species_coeff to reduce global memory access
  const int itype = type[i] - 1;
  const F_FLOAT species_coeff = d_species_coeffs[itype];
  const F_FLOAT xi[3] = {x(i, 0), x(i, 1), x(i, 2)};
  const bool use_env_gate = env_gate_enabled;
  const F_FLOAT env_screen_strength =
      use_env_gate ? d_env_gate_values(ii) : static_cast<F_FLOAT>(0.0);
  const F_FLOAT env_rho_chain_prefactor =
      use_env_gate ? d_env_gate_rho_chain_values(ii) : static_cast<F_FLOAT>(0.0);
  const F_FLOAT r_env = env_gate_cutoff_ratio * max_cutoff;

  for (int jj = 0; jj < jnum; jj++) {
    const int j = d_neighbors(i, jj) & NEIGHMASK;
    const int jtype = type[j] - 1;
    const F_FLOAT r[3] = {x(j, 0) - xi[0], x(j, 1) - xi[1], x(j, 2) - xi[2]};
    const F_FLOAT rsq = Kokkos::fma(r[0], r[0], Kokkos::fma(r[1], r[1], r[2] * r[2]));
    if (rsq >= max_cutoff_sq) continue;
    const F_FLOAT dist = Kokkos::sqrt(rsq);
    F_FLOAT temp_force[3] = {0, 0, 0};
    F_FLOAT env_rho_dr = 0.0;
    F_FLOAT env_activation = 0.0;
    F_FLOAT env_activation_der = 0.0;
    F_FLOAT env_pair_gate = 1.0;
    if (use_env_gate && dist > static_cast<F_FLOAT>(0.0) && dist < r_env) {
      F_FLOAT local_rho = 0.0;
      eval_env_gate_rho(itype, dist, local_rho, env_rho_dr);
    }
    if (use_env_gate) {
      env_gate_activation_kk(dist, r_env, env_gate_activation_on_ratio,
                             env_activation, env_activation_der);
      env_pair_gate = static_cast<F_FLOAT>(1.0) - env_screen_strength * env_activation;
    }
    int table_index = -1;
    int r_list = 0;
    int r_next = 0;
    F_FLOAT ddr = 0.0;
    if (do_list) {
      r_list = (int) Kokkos::floor(dist * inv_dr);
      const int last_interval = list_grid_size - 2;
      if (r_list < 0) r_list = 0;
      if (r_list > last_interval) r_list = last_interval;
      r_next = r_list + 1;
      const int shift = itype * species_count + jtype;
      table_index = d_pair_to_table_index(shift);
      if (table_index >= 0) {
        ddr = dist * inv_dr - r_list;
        if (ddr < 0.0) ddr = 0.0;
        if (ddr > 1.0) ddr = 1.0;
      }
    }

    for (int mu_group = 0; mu_group < basic_mu_group_count; mu_group++) {
      F_FLOAT val = 0.0;
      F_FLOAT der = 0.0;
      if (table_index >= 0) {
        const F_FLOAT v1 = d_radial_list(table_index, r_list, mu_group);
        const F_FLOAT v2 = d_radial_list(table_index, r_next, mu_group);
        const F_FLOAT d1 = d_radial_der_list(table_index, r_list, mu_group);
        const F_FLOAT d2 = d_radial_der_list(table_index, r_next, mu_group);
        val = v1 + ddr * (v2 - v1);
        der = d1 + ddr * (d2 - d1);
      } else {
        eval_radial_basic_mu_group(itype, jtype, dist, mu_group, val, der);
      }

      for (int grouped_idx = d_basic_mu_offsets(mu_group);
           grouped_idx < d_basic_mu_offsets(mu_group + 1); grouped_idx++) {
        const int k = d_basic_grouped_indices(grouped_idx);
        const F_FLOAT coeff = d_nbh_energy_ders_wrt_moments(ii, k);
        const int a0 = d_alpha_index_basic(k, 1);
        const int a1 = d_alpha_index_basic(k, 2);
        const int a2 = d_alpha_index_basic(k, 3);
        const int norm_rank = a0 + a1 + a2;

        const F_FLOAT norm_fac = 1.0 / int_pow(dist, norm_rank);
        const F_FLOAT val_scaled = val * norm_fac;
        const F_FLOAT der_scaled =
            Kokkos::fma(norm_fac, der, -norm_rank * val_scaled / dist);

        const F_FLOAT pow0 = int_pow(r[0], a0);
        const F_FLOAT pow1 = int_pow(r[1], a1);
        const F_FLOAT pow2 = int_pow(r[2], a2);
        const F_FLOAT pow = pow0 * pow1 * pow2;
        const F_FLOAT raw_contrib = val_scaled * pow;
        F_FLOAT common = pow * der_scaled / dist;

        F_FLOAT jac0 = common * r[0];
        F_FLOAT jac1 = common * r[1];
        F_FLOAT jac2 = common * r[2];

        if (a0 != 0)
          jac0 = Kokkos::fma(val_scaled * a0, int_pow(r[0], a0 - 1) * pow1 * pow2, jac0);
        if (a1 != 0)
          jac1 = Kokkos::fma(val_scaled * a1, pow0 * int_pow(r[1], a1 - 1) * pow2, jac1);
        if (a2 != 0)
          jac2 = Kokkos::fma(val_scaled * a2, pow0 * pow1 * int_pow(r[2], a2 - 1), jac2);

        if (use_env_gate) {
          const F_FLOAT inv_dist = static_cast<F_FLOAT>(1.0) / dist;
          const F_FLOAT activation_der_factor =
              -env_screen_strength * env_activation_der * raw_contrib * inv_dist;
          temp_force[0] += coeff * (env_pair_gate * jac0 + activation_der_factor * r[0]);
          temp_force[1] += coeff * (env_pair_gate * jac1 + activation_der_factor * r[1]);
          temp_force[2] += coeff * (env_pair_gate * jac2 + activation_der_factor * r[2]);
        } else {
          temp_force[0] += coeff * jac0;
          temp_force[1] += coeff * jac1;
          temp_force[2] += coeff * jac2;
        }
      }
    }
    if (use_env_gate && env_rho_dr != static_cast<F_FLOAT>(0.0)) {
      const F_FLOAT rho_der_factor = env_rho_chain_prefactor * env_rho_dr / dist;
      temp_force[0] += rho_der_factor * r[0];
      temp_force[1] += rho_der_factor * r[1];
      temp_force[2] += rho_der_factor * r[2];
    }
    
    // SUS2-MLIP: Multiply force by species_coeffs (use cached variable)
    a_f(i, 0) += temp_force[0];
    a_f(i, 1) += temp_force[1];
    a_f(i, 2) += temp_force[2];

    a_f(j, 0) -= temp_force[0];
    a_f(j, 1) -= temp_force[1];
    a_f(j, 2) -= temp_force[2];

    if (need_virial_tally) {
      v_tally_xyz<NEIGHFLAG>(ev, i, j, temp_force[0], temp_force[1], temp_force[2], r[0], r[1],
                             r[2]);
    }
  }

  if (need_energy_tally) {
    // FIXED: Use cached itype and species_coeff instead of recalculating
    // SUS2-MLIP: Include shift_coeffs in energy calculation
    // CRITICAL FIX: Use d_shift_coeffs instead of d_regression_coeffs to avoid NaN
    F_FLOAT scalar_sum = 1.0;

    // Take the linear combination of the basis set and the learned coefficients.
    for (int k = 0; k < alpha_scalar_count; k++) {
      int basis_member_index = d_alpha_moment_mapping(k);
      scalar_sum += d_linear_coeffs(k) * d_moment_tensor_vals(ii, basis_member_index);
    }
    F_FLOAT nbh_energy = species_coeff * scalar_sum + d_shift_coeffs[itype];
    
    // FIXED: Removed the duplicate multiplication that was causing energy explosion
    // nbh_energy *= d_species_coeffs[itype];

    if (eflag_global) ev.evdwl += nbh_energy;
    if (eflag_atom) d_eatom[i] += nbh_energy;
  }
}

template <class DeviceType>
template <int NEIGHFLAG, int EVFLAG>
KOKKOS_INLINE_FUNCTION void
PairSUS2MTPKokkos<DeviceType>::operator()(
    TagPairSUS2MTPComputeForce<NEIGHFLAG, EVFLAG>,
    const int &ii) const
{
  EV_FLOAT ev;
  this->template operator()<NEIGHFLAG, EVFLAG>(TagPairSUS2MTPComputeForce<NEIGHFLAG, EVFLAG>(), ii, ev);
}

// =========== Helper Functions (Also used in other Kokkos potentials)===========
template <class DeviceType>
template <int NEIGHFLAG>
KOKKOS_INLINE_FUNCTION void
PairSUS2MTPKokkos<DeviceType>::v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j, const F_FLOAT &fx,
                                       const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                                       const F_FLOAT &dely, const F_FLOAT &delz) const
{
  // The vatom array is duplicated for OpenMP, atomic for GPU, and neither for Serial

  auto v_vatom = ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_vatom),
                                   decltype(ndup_vatom)>::get(dup_vatom, ndup_vatom);
  auto a_vatom = v_vatom.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();

  // Match the CPU reference implementation:
  // global virial uses -f \otimes r on the directed i->j edge,
  // centroid stress writes only to atom j with the full 9 components.
  const KK_ACC_FLOAT cpu_v0 = -fx * delx;
  const KK_ACC_FLOAT cpu_v1 = -fy * dely;
  const KK_ACC_FLOAT cpu_v2 = -fz * delz;
  const KK_ACC_FLOAT cpu_v3 = -fx * dely;
  const KK_ACC_FLOAT cpu_v4 = -fx * delz;
  const KK_ACC_FLOAT cpu_v5 = -fy * delz;
  const KK_ACC_FLOAT cpu_v6 = -fy * delx;
  const KK_ACC_FLOAT cpu_v7 = -fz * delx;
  const KK_ACC_FLOAT cpu_v8 = -fz * dely;

  // Keep the conventional symmetric pair-stress path for vatom.
  const KK_ACC_FLOAT v0 = delx * fx;
  const KK_ACC_FLOAT v1 = dely * fy;
  const KK_ACC_FLOAT v2 = delz * fz;
  const KK_ACC_FLOAT v3 = delx * fy;
  const KK_ACC_FLOAT v4 = delx * fz;
  const KK_ACC_FLOAT v5 = dely * fz;

  if (vflag_global) {
    ev.v[0] += cpu_v0;
    ev.v[1] += cpu_v1;
    ev.v[2] += cpu_v2;
    ev.v[3] += cpu_v3;
    ev.v[4] += cpu_v4;
    ev.v[5] += cpu_v5;
  }

  if (vflag_atom) {
    a_vatom(i, 0) += 0.5 * v0;
    a_vatom(i, 1) += 0.5 * v1;
    a_vatom(i, 2) += 0.5 * v2;
    a_vatom(i, 3) += 0.5 * v3;
    a_vatom(i, 4) += 0.5 * v4;
    a_vatom(i, 5) += 0.5 * v5;
    a_vatom(j, 0) += 0.5 * v0;
    a_vatom(j, 1) += 0.5 * v1;
    a_vatom(j, 2) += 0.5 * v2;
    a_vatom(j, 3) += 0.5 * v3;
    a_vatom(j, 4) += 0.5 * v4;
    a_vatom(j, 5) += 0.5 * v5;
  }

  if (cvflag_atom) {
    auto v_cvatom =
        ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, decltype(dup_cvatom),
                          decltype(ndup_cvatom)>::get(dup_cvatom, ndup_cvatom);
    auto a_cvatom = v_cvatom.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();
    a_cvatom(j, 0) += cpu_v0;
    a_cvatom(j, 1) += cpu_v1;
    a_cvatom(j, 2) += cpu_v2;
    a_cvatom(j, 3) += cpu_v3;
    a_cvatom(j, 4) += cpu_v4;
    a_cvatom(j, 5) += cpu_v5;
    a_cvatom(j, 6) += cpu_v6;
    a_cvatom(j, 7) += cpu_v7;
    a_cvatom(j, 8) += cpu_v8;
  }
}

template <class DeviceType>
template <class TagStyle>
void PairSUS2MTPKokkos<DeviceType>::check_team_size_for(int inum, int &team_size, int vector_length)
{
  int team_size_max;

  team_size_max = Kokkos::TeamPolicy<DeviceType, TagStyle>(inum, Kokkos::AUTO)
                      .team_size_max(*this, Kokkos::ParallelForTag());

  if (team_size * vector_length > team_size_max) team_size = team_size_max / vector_length;
}

template <class DeviceType>
template <typename scratch_type>
int PairSUS2MTPKokkos<DeviceType>::scratch_size_helper(int values_per_team)
{
  typedef Kokkos::View<scratch_type *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ScratchViewType;

  return ScratchViewType::shmem_size(values_per_team);
}    // namespace LAMMPS_NS

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairSUS2MTPKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairSUS2MTPKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS

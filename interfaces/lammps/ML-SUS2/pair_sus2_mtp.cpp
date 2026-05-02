/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#include "pair_sus2_mtp.h"

#include "sus2_mtp_radial_basis.h"
#include "sus2_mtp_rb_chebyshev_basis.h"
#include "sus2_mtp_rb_chebyshev_ss_basis.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>
//#define LAMMPS_VERSION_NUMBER 20220324 // use the new neighbor list starting from this version

using namespace LAMMPS_NS;

namespace {

constexpr int kEnvGateChannels = 6;
constexpr double kEnvGateMaxLogDensityCoeff = 6.0;
constexpr double kEnvGateDefaultActivationOnRatio = 0.5;

double stable_sigmoid(double value)
{
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

double env_gate_density_coeff(double raw_log_density)
{
  return std::exp(std::min(raw_log_density, kEnvGateMaxLogDensityCoeff));
}

double clamp01(double value)
{
  return std::max(0.0, std::min(1.0, value));
}

void bernstein_degree5(double y, double *basis, double *basis_der)
{
  y = clamp01(y);
  const double u = 1.0 - y;
  const double y2 = y * y;
  const double y3 = y2 * y;
  const double y4 = y3 * y;
  const double y5 = y4 * y;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;

  basis[0] = u5;
  basis[1] = 5.0 * y * u4;
  basis[2] = 10.0 * y2 * u3;
  basis[3] = 10.0 * y3 * u2;
  basis[4] = 5.0 * y4 * u;
  basis[5] = y5;

  const double degree4[5] = {u4, 4.0 * y * u3, 6.0 * y2 * u2, 4.0 * y3 * u, y4};
  basis_der[0] = -5.0 * degree4[0];
  for (int q = 1; q < 5; ++q) basis_der[q] = 5.0 * (degree4[q - 1] - degree4[q]);
  basis_der[5] = 5.0 * degree4[4];
}

void env_gate_activation(double r,
                         double r_env,
                         double activation_on_ratio,
                         double *activation,
                         double *activation_der)
{
  const double r_on = activation_on_ratio * r_env;
  if (r <= r_on) {
    *activation = 0.0;
    *activation_der = 0.0;
    return;
  }
  if (r >= r_env) {
    *activation = 1.0;
    *activation_der = 0.0;
    return;
  }

  const double width = r_env - r_on;
  const double t = (r - r_on) / width;
  const double t2 = t * t;
  const double t3 = t2 * t;
  *activation = 10.0 * t3 - 15.0 * t3 * t + 6.0 * t3 * t2;
  *activation_der = 30.0 * t2 * (1.0 - t) * (1.0 - t) / width;
}

bool uses_preinterpolation_table(int basis_type)
{
  return basis_type == SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP ||
         basis_type == SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP ||
         basis_type == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP ||
         basis_type == SUS2RadialMTPBasis::JACOBI_SSS_LMP;
}

bool basis_requires_per_mu_sigma(int basis_type)
{
  return basis_type == SUS2RadialMTPBasis::JACOBI_SSS_LMP;
}

}    // namespace

PairSUS2MTP::PairSUS2MTP(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
//#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag = CENTROID_AVAIL;
//#else
//  centroidstressflag = 2;
//#endif
  //centroidstressflag = CENTROID_AVAIL;
}

/* ---------------------------------------------------------------------- */

PairSUS2MTP::~PairSUS2MTP()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(radial_list);
	    memory->destroy(radial_der_list);
	    memory->destroy(env_gate_radial_list);
	    memory->destroy(env_gate_radial_der_list);
	    memory->destroy(env_gate_rho_list);
	    memory->destroy(env_gate_rho_der_list);
	    memory->destroy(pair_to_table_index);
    memory->destroy(moment_tensor_vals);
    memory->destroy(regression_coeffs);
    memory->destroy(linear_coeffs);
    memory->destroy(species_coeffs);
    memory->destroy(alpha_index_basic);
    memory->destroy(alpha_index_times);
    memory->destroy(alpha_moment_mapping);
    memory->destroy(alpha_basic_mu);
    memory->destroy(alpha_basic_a0);
    memory->destroy(alpha_basic_a1);
    memory->destroy(alpha_basic_a2);
    memory->destroy(alpha_basic_norm_rank);
    memory->destroy(alpha_times_a0);
    memory->destroy(alpha_times_a1);
    memory->destroy(alpha_times_multiplier);
    memory->destroy(alpha_times_out);
    memory->destroy(moment_jacobian_x);
    memory->destroy(moment_jacobian_y);
    memory->destroy(moment_jacobian_z);
    memory->destroy(nbh_energy_ders_wrt_moments);
    memory->destroy(within_cutoff);
    memory->destroy(inv_dist_powers);
    memory->destroy(coord_powers_x);
    memory->destroy(coord_powers_y);
    memory->destroy(coord_powers_z);
    memory->destroy(radial_vals);
    memory->destroy(radial_ders);
    memory->destroy(weighted_basic_moment_ders);
    memory->destroy(env_rho_dr);
    memory->destroy(env_activation_basic_vals);
    memory->destroy(mu_to_K);
    memory->destroy(mu_to_sigma);

    delete radial_basis;
    radial_basis = nullptr;
  }
}

/* ----------------------------------------------------------------------
   Straightfoward SUS2-MTP implementation based on SUS2-MLIP-1.1
   ---------------------------------------------------------------------- */

void PairSUS2MTP::compute(int eflag, int vflag)
{
  ev_setup(eflag, vflag);

  // CRITICAL: `alpha_index_basic_count` must be valid. Working buffers
  // (`moment_jacobian`, `within_cutoff`, etc.) are lazily allocated below
  // when a neighbourhood is encountered (via memory->grow). Do not treat
  // unallocated buffers as an immediate fatal error here because in MPI
  // runs some ranks may not have allocated them yet.
  if (alpha_index_basic_count <= 0) {
    error->one(FLERR, "alpha_index_basic_count is invalid: {}. This indicates the MTP file was not read properly.", alpha_index_basic_count);
  }

  double **x = atom->x;      // atomic positons
  double **f = atom->f;      // atomic forces
  int *type = atom->type;    //atomic types

  int inum = list->inum;             // The number of central atoms (neigbhourhoods)
  int *ilist = list->ilist;          // List of central atom ids
  int *numneigh = list->numneigh;    // List of the number of neighbours for each central atom
  int **firstneigh =
      list->firstneigh;    //List  (head of array) of neighbours for a given central atom

  // Local constants for coefficient indexing
  const int C = species_count;
  const int R = radial_basis_size;
  const int pairs_count = C * C;
  const int K_ = K_scaling;


  // Loop over all provided neighbourhoods
  for (int ii = 0; ii < inum; ii++) {
    const int i = ilist[ii];          // Set central atom index
    const int itype = type[i] - 1;    // Set central atom type. Convert back to zero indexing.
    if (itype >= species_count)
      error->one(FLERR,
                 "Too few species count in the MTP potential!");
    
    int jnum = numneigh[i];                                         // Set number of neighbours
    double nbh_energy = 0;
    const double xi[3] = {x[i][0], x[i][1],
                          x[i][2]};    // Cache the position of the central atom for efficiency

    // Grow jacobian working arrays if needed
    if (jac_size < jnum) {
      memory->grow(moment_jacobian_x, jnum * alpha_index_basic_count,
                   "moment_jacobian_x");
      memory->grow(moment_jacobian_y, jnum * alpha_index_basic_count,
                   "moment_jacobian_y");
      memory->grow(moment_jacobian_z, jnum * alpha_index_basic_count,
                   "moment_jacobian_z");
      memory->grow(within_cutoff, jnum, "within_cutoff");    // Resize within cuf
      memory->grow(env_rho_dr, jnum, "env_rho_dr");
      jac_size = jnum;
    }
    std::fill(&moment_tensor_vals[0], &moment_tensor_vals[0] + alpha_moment_count,
              0.0);    //Fill moments with 0
    std::fill(&nbh_energy_ders_wrt_moments[0], &nbh_energy_ders_wrt_moments[0] + alpha_moment_count,
              0.0);    //Fill moment derivatives with 0

    double env_screen_strength = 0.0;
    double env_rho_factor = 0.0;
    const double r_env = env_gate_enabled ? env_gate_cutoff_ratio * max_cutoff : 0.0;
    const double r_env_sq = r_env * r_env;
    double env_density_coeffs[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (env_gate_enabled) {
      if (r_env <= 0.0) error->one(FLERR, "Invalid env_gate_cutoff_ratio in SUS2 model.");
      if (env_activation_basic_size < alpha_index_basic_count) {
        memory->grow(env_activation_basic_vals, alpha_index_basic_count,
                     "env_activation_basic_vals");
        env_activation_basic_size = alpha_index_basic_count;
      }
      if (jnum > 0) std::fill(env_rho_dr, env_rho_dr + jnum, 0.0);
      std::fill(env_activation_basic_vals,
                env_activation_basic_vals + alpha_index_basic_count, 0.0);
      for (int q = 0; q < env_gate_channel_count; ++q)
        env_density_coeffs[q] =
            env_gate_density_coeff(regression_coeffs[env_gate_log_density_coeffs_offset + itype * env_gate_channel_count + q]);

      double rho = 0.0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = firstneigh[i][jj];
        j &= NEIGHMASK;
        const int jtype = type[j] - 1;
        if (jtype >= species_count)
          error->one(FLERR, "Too few species count in the MTP potential!");
        const double rvec[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
        const double rsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
        if (rsq <= 0.0 || rsq >= r_env_sq) continue;
        const double dist = std::sqrt(rsq);
        double local_rho = 0.0;
        double local_rho_dr = 0.0;
        if (do_list && env_gate_rho_list) {
          int r_list = static_cast<int>(std::floor(dist * inv_dr));
          const int last_interval = list_grid_size - 2;
          if (r_list < 0) r_list = 0;
          if (r_list > last_interval) r_list = last_interval;
          const int r_next = r_list + 1;
          double ddr = dist * inv_dr - r_list;
          if (ddr < 0.0) ddr = 0.0;
          if (ddr > 1.0) ddr = 1.0;
          const double v1 = env_gate_rho_list[itype][r_list];
          const double v2 = env_gate_rho_list[itype][r_next];
          const double d1 = env_gate_rho_der_list[itype][r_list];
          const double d2 = env_gate_rho_der_list[itype][r_next];
          local_rho = v1 + ddr * (v2 - v1);
          local_rho_dr = d1 + ddr * (d2 - d1);
        } else {
          const double y = dist / r_env;
          const double dy_dr = 1.0 / r_env;
          const double cutoff = 1.0 - dist / r_env;
          const double f_env = cutoff * cutoff;
          const double df_dr = -2.0 * cutoff / r_env;
          double basis[6];
          double basis_der[6];
          bernstein_degree5(y, basis, basis_der);
          for (int q = 0; q < env_gate_channel_count; ++q) {
            const double weighted_basis = f_env * basis[q];
            const double weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
            local_rho += env_density_coeffs[q] * weighted_basis;
            local_rho_dr += env_density_coeffs[q] * weighted_basis_der;
          }
        }
        rho += local_rho;
        env_rho_dr[jj] = local_rho_dr;
      }
      const double lambda = stable_sigmoid(regression_coeffs[env_gate_lambda_raw_offset + itype]);
      const double tanh_rho = std::tanh(rho);
      const double sech2_rho = 1.0 - tanh_rho * tanh_rho;
      env_screen_strength = lambda * tanh_rho;
      env_rho_factor = -lambda * sech2_rho;
    }

    // ------------ Begin Alpha Basic Calc ------------
    // Loop over all neighbours
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];    //List of neighbours
      j &= NEIGHMASK;
      const int jtype = type[j] - 1;    // Convert back to zero indexing
      if (jtype >= species_count)
        error->one(FLERR,
                   "Too few species count in the MTP potential!");

      const double r[3] = {x[j][0] - xi[0], x[j][1] - xi[1], x[j][2] - xi[2]};
      const double rsq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      if (rsq > cutsq[itype + 1][jtype + 1]) {    //1 indexing
        within_cutoff[jj] = false;
        continue;
      }
      within_cutoff[jj] = true;

      const double dist = std::sqrt(rsq);
      const double inv_dist = 1.0 / dist;
      double env_activation = 0.0;
      double env_activation_der = 0.0;
      double pair_gate = 1.0;
      if (env_gate_enabled) {
        env_gate_activation(dist, r_env, env_gate_activation_on_ratio,
                            &env_activation, &env_activation_der);
        pair_gate = 1.0 - env_screen_strength * env_activation;
      }
      
      // SUS2-MLIP: Apply scaling transformation using regression_coeffs
      
      // Precompute the coord and distance powers
      inv_dist_powers[0] = 1.0;
      coord_powers_x[0] = 1.0;
      coord_powers_y[0] = 1.0;
      coord_powers_z[0] = 1.0;
      
      for (int k = 1; k < max_alpha_index_basic; k++) {
        inv_dist_powers[k] = inv_dist_powers[k - 1] * inv_dist;
        coord_powers_x[k] = coord_powers_x[k - 1] * r[0];
        coord_powers_y[k] = coord_powers_y[k - 1] * r[1];
        coord_powers_z[k] = coord_powers_z[k - 1] * r[2];
      }

      // SUS2-MLIP: Precompute radial basis function values for all k_ values
      // Use local temporary array as in reference implementation
      // This prevents issues with array indexing and memory overlap

      bool used_precomputed_table = false;
      if (do_list) {
        const int shift = C * itype + jtype;
        const int table_index = pair_to_table_index[shift];
        if (table_index >= 0) {
          int r_list = static_cast<int>(std::floor(dist * inv_dr));
          const int last_interval = list_grid_size - 2;
          if (r_list < 0) r_list = 0;
          if (r_list > last_interval) r_list = last_interval;
          int r_next = r_list + 1;
          double ddr = dist * inv_dr - r_list;
          if (ddr < 0.0) ddr = 0.0;
          if (ddr > 1.0) ddr = 1.0;
          double v1, v2, d1, d2;
          double *radial_row = radial_list[table_index][r_list];
          double *radial_next_row = radial_list[table_index][r_next];
          double *deriv_row = radial_der_list[table_index][r_list];
          double *deriv_next_row = radial_der_list[table_index][r_next];

          for (int m = 0; m < radial_func_count; m++) {
            v1 = radial_row[m];
            v2 = radial_next_row[m];
            d1 = deriv_row[m];
            d2 = deriv_next_row[m];
            radial_vals[m] = v1 + ddr * (v2 - v1);
            radial_ders[m] = d1 + ddr * (d2 - d1);
          }
          used_precomputed_table = true;
        }
      }

      if (!used_precomputed_table) {
      const bool per_mu_sigma = basis_requires_per_mu_sigma(radial_basis_type_index);
      const int radial_cache_count = per_mu_sigma ? radial_func_count : K_;
      int temp_size = radial_cache_count * R;
      std::vector<double> val_temp(temp_size, 0.0);
      std::vector<double> der_temp(temp_size, 0.0);

      if (per_mu_sigma) {
        for (int mu = 0; mu < radial_func_count; mu++) {
          const int k_ = mu_to_K[mu];
          const int sigma = mu_to_sigma[mu];
          double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
          double s = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];

          radial_basis->calc_radial_basis_ders(dist, scal, s, sigma);

          for (int ri = 0; ri < R; ri++) {
            val_temp[mu * R + ri] = radial_basis->radial_basis_vals[ri];
            der_temp[mu * R + ri] = radial_basis->radial_basis_ders[ri];
          }
        }
      } else {
        for (int k_ = 0; k_ < K_; k_++) {
          int sigma = mu_to_sigma[k_];

          // Read s1 and s2 from unified regression_coeffs array
          // SUS2-MLIP formula: regression_coeffs[C + 2*k_*C*C + C*type_central + type_outer]
          double scal = regression_coeffs[C + 2 * k_ * pairs_count + C * itype + jtype];
          double s = regression_coeffs[C + 2 * k_ * pairs_count + pairs_count + C * itype + jtype];

          // Call RB_Calc with s1 and s2 parameters
          radial_basis->calc_radial_basis_ders(dist, scal, s, sigma);

          // Store scaled values in temporary arrays
          for (int ri = 0; ri < R; ri++) {
            val_temp[k_ * R + ri] = radial_basis->radial_basis_vals[ri];
            der_temp[k_ * R + ri] = radial_basis->radial_basis_ders[ri];
          }
        }
      }

      // Now compute radial_vals for each mu

      for (int mu = 0; mu < radial_func_count; mu++) {

        double val = 0;
        double der = 0;
        // SUS2-MLIP: Map mu to K using mu_to_K
        int k_ = mu_to_K[mu];
        int radial_cache_index = per_mu_sigma ? mu : k_;
        int offset_mu = mu * (R + C);

        // SUS2-MLIP: Include species-dependent factors in radial basis calculation
        // Original: regression_coeffs[C+2*C*C*K_ + mu*(R+C) + R + type_central/outer]
        double species_factor = regression_coeffs[radial_coeffs_offset + R + itype] *
                              regression_coeffs[radial_coeffs_offset + R + jtype];

        #pragma omp simd reduction(+:val,der)
        for (int ri = 0; ri < R; ri++) {
          val += regression_coeffs[radial_coeffs_offset + offset_mu + ri] * species_factor *
                 val_temp[radial_cache_index * R + ri];
          der += regression_coeffs[radial_coeffs_offset + offset_mu + ri] * species_factor *
                 der_temp[radial_cache_index * R + ri];
        }

        radial_vals[mu] = val;
        radial_ders[mu] = der;
      }
      }


      //Calculate the alpha basics
      
      for (int k = 0; k < alpha_index_basic_count; k++) {
        const size_t jac_idx = static_cast<size_t>(jj) * alpha_index_basic_count + k;
        
        double val = 0;
        double der = 0;
        const int mu = alpha_basic_mu[k];

        val = radial_vals[mu];
        der = radial_ders[mu];
        // Normalize by the rank of alpha's corresponding tensor
        const int norm_rank = alpha_basic_norm_rank[k];
        double norm_fac = inv_dist_powers[norm_rank];
        val *= norm_fac;
        der = der * norm_fac - norm_rank * val * inv_dist;

        const int a0 = alpha_basic_a0[k];
        const int a1 = alpha_basic_a1[k];
        const int a2 = alpha_basic_a2[k];
        double pow0 = coord_powers_x[a0];
        double pow1 = coord_powers_y[a1];
        double pow2 = coord_powers_z[a2];
        const double pow = pow0 * pow1 * pow2;
        const double raw_contrib = val * pow;

        // Get the component's derivatives too
        // Simplified derivative calculation - full implementation with scaling would be more complex
        const double radial_der_pref = pow * der / dist;
        double raw_jac_x = radial_der_pref * r[0];
        double raw_jac_y = radial_der_pref * r[1];
        double raw_jac_z = radial_der_pref * r[2];

        if (a0 != 0) {
          raw_jac_x += val * a0 *
              coord_powers_x[a0 - 1] * pow1 * pow2;
        }    //Chain rule for nonzero rank
        if (a1 != 0) {
          raw_jac_y += val * a1 * pow0 *
              coord_powers_y[a1 - 1] * pow2;
        }    //Chain rule for nonzero rank
        if (a2 != 0) {
          raw_jac_z += val * a2 * pow0 * pow1 *
              coord_powers_z[a2 - 1];
        }    //Chain rule for nonzero rank

        if (env_gate_enabled) {
          const double activation_der_factor =
              -env_screen_strength * env_activation_der * raw_contrib * inv_dist;
          env_activation_basic_vals[k] += env_activation * raw_contrib;
          moment_tensor_vals[k] += pair_gate * raw_contrib;
          moment_jacobian_x[jac_idx] = pair_gate * raw_jac_x + activation_der_factor * r[0];
          moment_jacobian_y[jac_idx] = pair_gate * raw_jac_y + activation_der_factor * r[1];
          moment_jacobian_z[jac_idx] = pair_gate * raw_jac_z + activation_der_factor * r[2];
        } else {
          moment_tensor_vals[k] += raw_contrib;
          moment_jacobian_x[jac_idx] = raw_jac_x;
          moment_jacobian_y[jac_idx] = raw_jac_y;
          moment_jacobian_z[jac_idx] = raw_jac_z;
        }
      }
    }

    // ------------ Contruct Other Alphas  ------------
    const int * __restrict times_a0 = alpha_times_a0;
    const int * __restrict times_a1 = alpha_times_a1;
    const int * __restrict times_mult = alpha_times_multiplier;
    const int * __restrict times_out = alpha_times_out;
    double * __restrict moments = moment_tensor_vals;

    for (int k = 0; k < alpha_index_times_count; k++) {
      double val0 = moments[times_a0[k]];
      double val1 = moments[times_a1[k]];
      int val2 = times_mult[k];

      moments[times_out[k]] += val2 * val0 * val1;
    }
    // ------------ Compute Basis Set From Alpha Map ------------
    if (eflag_atom || eflag_global) {
      // SUS2-MLIP: Include shift_coeffs in energy calculation
      nbh_energy = shift_coeffs[itype] + species_coeffs[itype];
      for (int k = 0; k < alpha_scalar_count; k++)
        nbh_energy += linear_coeffs[k] * moment_tensor_vals[alpha_moment_mapping[k]] * species_coeffs[itype];

      // Tally energies per flags
      if (eflag_atom) eatom[i] = nbh_energy;
      if (eflag_global) eng_vdwl += nbh_energy;
    }

    // =========== Begin Backpropogation ===========

    //------------ Step 1: NBH energy derivative is the corresponding linear combination------------
    for (int k = 0; k < alpha_scalar_count; k++)
      nbh_energy_ders_wrt_moments[alpha_moment_mapping[k]] = linear_coeffs[k];

    //------------ Step 2: Propogate chain rule through the alpha times to the alpha basics ------------
    const int * __restrict times_a0_back = alpha_times_a0;
    const int * __restrict times_a1_back = alpha_times_a1;
    const int * __restrict times_mult_back = alpha_times_multiplier;
    const int * __restrict times_out_back = alpha_times_out;
    double * __restrict moment_ders = nbh_energy_ders_wrt_moments;

    for (int k = alpha_index_times_count - 1; k >= 0; k--) {
      int a0 = times_a0_back[k];
      int a1 = times_a1_back[k];
      int multipiler = times_mult_back[k];
      int a3 = times_out_back[k];

      double val0 = moment_tensor_vals[a0];
      double val1 = moment_tensor_vals[a1];
      double val3 = moment_ders[a3];

      moment_ders[a1] += val3 * multipiler * val0;
      moment_ders[a0] += val3 * multipiler * val1;
    }
    const double species_weight = species_coeffs[itype];
    for (int k = 0; k < alpha_index_basic_count; k++)
      weighted_basic_moment_ders[k] = nbh_energy_ders_wrt_moments[k] * species_weight;

    double env_rho_force_prefactor = 0.0;
    if (env_gate_enabled) {
      double env_activation_derivative_weight = 0.0;
      for (int k = 0; k < alpha_index_basic_count; k++)
        env_activation_derivative_weight += weighted_basic_moment_ders[k] * env_activation_basic_vals[k];
      env_rho_force_prefactor = env_rho_factor * env_activation_derivative_weight;
    }

    //------------ Step 3: Multiply energy ders wrt moment by the Jacobian to get forces ------------
    for (int jj = 0; jj < jnum; jj++) {
      int j = firstneigh[i][jj];
      j &= NEIGHMASK;
      if (!within_cutoff[jj] &&
          (!env_gate_enabled || env_rho_dr[jj] == 0.0))
        continue;

      double temp_force[3] = {0, 0, 0};
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      if (within_cutoff[jj]) {
        const size_t jac_offset = static_cast<size_t>(jj) * alpha_index_basic_count;
        const double * __restrict jac_x = moment_jacobian_x + jac_offset;
        const double * __restrict jac_y = moment_jacobian_y + jac_offset;
        const double * __restrict jac_z = moment_jacobian_z + jac_offset;
        const double * __restrict weighted = weighted_basic_moment_ders;

        #pragma omp simd reduction(+:fx,fy,fz)
        for (int k = 0; k < alpha_index_basic_count; k++) {
          const double pref = weighted[k];
          fx += pref * jac_x[k];
          fy += pref * jac_y[k];
          fz += pref * jac_z[k];
        }
      }
      double r[3] = {0.0, 0.0, 0.0};
      bool have_r = false;
      if (env_gate_enabled && env_rho_dr[jj] != 0.0) {
        r[0] = x[j][0] - xi[0];
        r[1] = x[j][1] - xi[1];
        r[2] = x[j][2] - xi[2];
        have_r = true;
        const double dist = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (dist > 0.0) {
          const double rho_der_factor = env_rho_force_prefactor * env_rho_dr[jj] / dist;
          fx += rho_der_factor * r[0];
          fy += rho_der_factor * r[1];
          fz += rho_der_factor * r[2];
        }
      }
      temp_force[0] = fx;
      temp_force[1] = fy;
      temp_force[2] = fz;

      f[i][0] += temp_force[0];
      f[i][1] += temp_force[1];
      f[i][2] += temp_force[2];

      f[j][0] -= temp_force[0];
      f[j][1] -= temp_force[1];
      f[j][2] -= temp_force[2];

      //Calculate virial stress
      if (vflag) {
        // We only need to calculate rel pos again if stress are needed
        if (!have_r) {
          r[0] = x[j][0] - xi[0];
          r[1] = x[j][1] - xi[1];
          r[2] = x[j][2] - xi[2];
        }
        virial[0] -= temp_force[0] * r[0];    //xx
        virial[1] -= temp_force[1] * r[1];    //yy
        virial[2] -= temp_force[2] * r[2];    //zz

        virial[3] -= temp_force[0] * r[1] ;    //xy
        virial[4] -= temp_force[0] * r[2] ;    //xz
        virial[5] -= temp_force[1] * r[2] ;    //yz

        if (cvflag_atom) {
          cvatom[j][0] -= temp_force[0] * r[0];    //xx
          cvatom[j][1] -= temp_force[1] * r[1];    //yy
          cvatom[j][2] -= temp_force[2] * r[2];    //zz

          cvatom[j][3] -= temp_force[0] * r[1] ;    //xy
          cvatom[j][4] -= temp_force[0] * r[2] ;    //xz
          cvatom[j][5] -= temp_force[1] * r[2] ;    //yz

          cvatom[j][6] -= temp_force[1] * r[0] ;    //yx
          cvatom[j][7] -= temp_force[2] * r[0] ;    //zx
          cvatom[j][8] -= temp_force[2] * r[1] ;    //zy
        }
      }
    }
  }
}
/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSUS2MTP::settings(int narg, char **arg)
{
  requested_tabstep = 1.0e-4;
  tabstep_set_by_user = false;

  if (narg < 1) error->all(FLERR, "Pair sus2mtp requires a potential file.");
  if ((narg - 1) % 2 != 0)
    error->all(FLERR,
               "Pair sus2mtp optional arguments must be specified as keyword/value pairs.");

  for (int i = 1; i < narg; i += 2) {
    const std::string keyword = LAMMPS_NS::utils::lowercase(arg[i]);
    if (keyword == "tabstep") {
      requested_tabstep = utils::numeric(FLERR, arg[i + 1], true, lmp);
      if (requested_tabstep <= 0.0)
        error->all(FLERR, "Pair sus2mtp tabstep must be greater than 0.");
      tabstep_set_by_user = true;
      continue;
    }

    error->all(FLERR,
               "Pair sus2mtp only supports the optional keyword \"tabstep\".");
  }

  FILE *mtp_file = utils::open_potential(arg[0], lmp, nullptr);
  read_file(mtp_file);
  fclose(mtp_file);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSUS2MTP::coeff(int narg, char **arg)
{
  // The potential file is specified in the setting function instead.
  if (narg != 2) error->all(FLERR, "Only \"pair_coeff * *\" is permitted");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSUS2MTP::init_style()
{
  if (force->newton_pair != 1) error->all(FLERR, "Pair style SUS2MTP requires Newton Pair on");

  // Request a full neighbourhood list which is needed for MTP
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSUS2MTP::init_one(int i, int j)
{
  if (i > species_count || j > species_count)
    error->all(FLERR,
               "SUS2MTP model has {} species, but LAMMPS data contains atom type pair {}-{}.",
               species_count, i, j);

  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set. See types {}-{}.", i, j);

  return radial_basis->max_cutoff;
}

/* ----------------------------------------------------------------------
   SUS2-MTP file parsing helper function. 
   Based on SUS2-MLIP-1.1/dev_src/mtpr.cpp Load() function.
   Supports L parameter, scaling_map, shift_coeffs, scal_coeffs, and multiple radial basis types.
------------------------------------------------------------------------- */
void PairSUS2MTP::read_file(FILE *mtp_file)
{
  /*NOTE: TextFileReader is used in lieu of PotentialFileReader to ensure compatability 
with the MLIP-3 package. The alpha indicies in this format are all in one line, requiring
access to the buffer size that is not provided in PFR.
*/

  //Open the MTP file on proc 0
  if (comm->me == 0) {
    TextFileReader tfr(mtp_file, "ml-mtp");
    tfr.ignore_comments = true;
    std::string new_separators = "=, ";
    std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

    ValueTokenizer line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    std::string keyword = line_tokens.next_string();

    if (keyword != "MTP")    // Files checking
      error->one(FLERR, "Only MTP potential files are accepted.");
    std::string version_line = std::string(tfr.next_line());
    if (version_line.find("version = 1.1.0") != std::string::npos) {
      env_gate_enabled = false;
    } else if (version_line.find("version = 2.0.0") != std::string::npos) {
      env_gate_enabled = true;
      env_gate_cutoff_ratio = 0.5;
      env_gate_activation_on_ratio = kEnvGateDefaultActivationOnRatio;
      env_gate_channel_count = kEnvGateChannels;
    } else {
      error->one(FLERR, "MTP file must have version \"1.1.0\" or \"2.0.0\"");
    }

    // Read the potential name (optional)
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();

    if (keyword == "potential_name") {
      try {
        potential_name = line_tokens.next_string();
      } catch (TokenizerException e) {
        potential_name = "";
      }
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }

    //Check scaling (optional)
    if (keyword == "scaling") {
      scaling = line_tokens.next_double();
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      scaling = 1;
    }

    utils::logmesg(lmp, "The scaling is : {:.2e}.\n", scaling);

    // Read L parameter (SUS2-MLIP specific, optional)
    if (keyword == "L") {
      L_max = line_tokens.next_int();
      L_max += 1;  // Convert from 0-indexed to count (L=0,1,...,L_max)
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      L_max = 1;  // Default value
    }

    // Read scaling_map (SUS2-MLIP specific)
    if (keyword == "scaling_map") {
      scaling_map = line_tokens.next_string();
      if (scaling_map != "K" && scaling_map != "L" && scaling_map != "LK")
        error->one(FLERR, "scaling_map must be 'K', 'L', or 'LK'");
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      scaling_map = "LK";  // Default
    }

    utils::logmesg(lmp, "L_max = {}, scaling_map = {}\n", L_max, scaling_map);

    // Read the species count
    if (keyword != "species_count")
      error->one(FLERR, "Error reading MTP file. Species count not found.");
    species_count = line_tokens.next_int();
    utils::logmesg(lmp, "There are {} species.\n", species_count);

    int np1 = ((atom && atom->ntypes > species_count) ? atom->ntypes : species_count) + 1;
    memory->create(setflag, np1, np1, "pair:setflag");
    memory->create(cutsq, np1, np1, "pair:cutsq");
    for (int i = 0; i < np1; ++i) {
      for (int j = 0; j < np1; ++j) {
        setflag[i][j] = 0;
        cutsq[i][j] = 0.0;
      }
    }

    // Read the potential tag (also optional)
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword == "potential_tag") {
      try {
        potential_tag = line_tokens.next_string();
      } catch (TokenizerException e) {
        potential_tag = "";
      }
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    }

    // Read the radial basis type
    if (keyword != "radial_basis_type")
      error->one(FLERR, "Error reading MTP file. No radial basis set type is specified.");
    std::string radial_basis_type = line_tokens.next_string();
    radial_basis_type_str = radial_basis_type;

    // Set the type of radial basis based on SUS2-MLIP supported types
    if (radial_basis_type == "RBChebyshev") {
      radial_basis = new RBChebyshev(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_s") {
      radial_basis = new RBChebyshev_s(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_S;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_ss") {
      radial_basis = new RBChebyshev_ss(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SS;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_sss") {
      radial_basis = new RBChebyshev_sss(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBChebyshev_sss_lmp") {
      radial_basis = new RBChebyshev_sss_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p") {
      radial_basis = new RBLaguerre_log1p(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_lmp") {
      radial_basis = new RBLaguerre_log1p_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_pos") {
      radial_basis = new RBLaguerre_log1p_pos(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBLaguerre_log1p_pos_lmp") {
      radial_basis = new RBLaguerre_log1p_pos_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP;
      radial_basis->scaling = scaling;
    } else if (radial_basis_type == "RBJacobi_sss_lmp") {
      radial_basis = new RBJacobi_sss_lmp(tfr, lmp);
      radial_basis_type_index = SUS2RadialMTPBasis::JACOBI_SSS_LMP;
      radial_basis->scaling = scaling;
    } else


      error->one(FLERR,
                 "Error reading MTP file. The specified radial basis set type, {}, was not found..",
                 radial_basis_type);

    radial_basis_size = radial_basis->size;
    
    utils::logmesg(lmp, "Radial basis type: {}, size: {}, index: {} \n", radial_basis_type, radial_basis_size,radial_basis_type_index);

    // Read the basis function count
    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
    if (keyword != "radial_funcs_count")
      lmp->error->one(FLERR, "Error in reading MTP file. Cannot read radial function count.");
    radial_func_count = line_tokens.next_int();    // Assuming count is an int

    // Compute K_scaling based on scaling_map
    if (scaling_map == "K") {
      K_scaling = radial_func_count / L_max;
    } else if (scaling_map == "L") {
      K_scaling = L_max;
    } else {  // LK
      K_scaling = radial_func_count;
    }

    utils::logmesg(lmp, "radial_funcs_count = {}, K_scaling = {}\n", radial_func_count, K_scaling);

    // Allocate mu_to_K and mu_to_sigma arrays
    memory->create(mu_to_K, radial_func_count, "mu_to_K");
    memory->create(mu_to_sigma, radial_func_count, "mu_to_sigma");
    
    // Initialize mu_to_sigma (always n/L)
    for (int n = 0; n < radial_func_count; n++) {
      mu_to_sigma[n] = n / L_max;
    }
    
    // Initialize mu_to_K based on scaling_map
    if (scaling_map == "K") {
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n / L_max;
      }
    } else if (scaling_map == "L") {
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n % L_max;
      }
    } else {  // LK
      for (int n = 0; n < radial_func_count; n++) {
        mu_to_K[n] = n;
      }
    }

    // Check for magnetic basis which is currently unsupported.
    // CRITICAL FIX: Determine expected scal_coeffs count BEFORE reading the line to set proper buffer size
    int pairs_count_temp = species_count * species_count;
    int scal_coeffs_count_temp = 2 * pairs_count_temp * K_scaling;
    int scal_coeffs_buffer_size = scal_coeffs_count_temp * 30 + 500;
    
    // Set large buffer size BEFORE reading to ensure scal_coeffs line is not truncated
    tfr.set_bufsize(scal_coeffs_buffer_size);
    
    // Save the raw line content before parsing for potential re-parsing with different separators
    std::string current_raw_line = std::string(tfr.next_line());
    utils::logmesg(lmp, "Raw line content: '{}'\n", current_raw_line);
    
    line_tokens = ValueTokenizer(current_raw_line, separators);
    keyword = line_tokens.next_string();
    utils::logmesg(lmp, "Next keyword after radial_funcs_count: '{}'\n", keyword);
    if (keyword != "radial_coeffs" && keyword != "shift_coeffs" && keyword != "scal_coeffs") {
      lmp->error->one(FLERR, "Error in reading MTP file. Expected 'radial_coeffs', 'shift_coeffs', or 'scal_coeffs', but found '{}'.", keyword);
    }

    // Apply the robust keyword detection strategy from the reference code
    // but without additional memory allocation - use existing unified array
    utils::logmesg(lmp, "Applying robust keyword detection for MTP file reading\n");

    // Initialize saved_line and saved_line_valid for use across multiple sections
    std::string saved_line;
    bool saved_line_valid = false;

    // Allocate unified regression_coeffs array
    // SUS2-MLIP-1.1: regression_coeffs.resize(C + 2*C*C*K_ + radial_func_count*(rb_size + species_count));
    shift_coeffs_offset = 0;
    scal_coeffs_offset = species_count;
    int pairs_count = species_count * species_count;
    int scal_coeffs_count_local = 2 * pairs_count * K_scaling;
    int radial_coeff_count_local = radial_func_count * (radial_basis_size + species_count);
    int env_gate_coeff_count_local =
        env_gate_enabled ? species_count * (1 + env_gate_channel_count) : 0;
    env_gate_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local + radial_coeff_count_local;
    env_gate_lambda_raw_offset = env_gate_coeffs_offset;
    env_gate_log_density_coeffs_offset = env_gate_coeffs_offset + species_count;
    regression_coeffs_count = env_gate_coeffs_offset + env_gate_coeff_count_local;
    
    memory->create(regression_coeffs, regression_coeffs_count, "regression_coeffs");

    // Set offset pointers
    radial_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local;
    
    shift_coeffs = &regression_coeffs[shift_coeffs_offset];
    scal_coeffs = &regression_coeffs[scal_coeffs_offset];
    radial_basis_coeffs = &regression_coeffs[radial_coeffs_offset];

    // Read shift_coeffs (SUS2-MLIP specific, optional) - robust detection without extra memory allocation
    // FIXED: Parse values from the same line (keyword = {values} format) and remove trailing braces
    // CRITICAL FIX: Use custom separators that don't split scientific notation (e.g., e+00)
    if (keyword == "shift_coeffs") {
      utils::logmesg(lmp, "Found shift_coeffs keyword, reading coefficients...\n");
      
      // Re-parse the line with custom separators that preserve scientific notation
      // Don't include +, -, * as separators since they appear in scientific notation
      // Use the previously saved raw line content
      std::string shift_separators = " ={},\t\r\n";
      ValueTokenizer shift_tokens(current_raw_line, shift_separators);
      
      // Skip "shift_coeffs" keyword
      shift_tokens.next_string();
      
      // The equal sign is skipped automatically by tokenizer
      // Now parse the values inside braces
      for (int i = 0; i < species_count; i++) {
        try {
          // Get the next token - should be the full scientific notation number
          std::string token = shift_tokens.next_string();
          
          // Remove possible braces and trim whitespace
          while (!token.empty() && (token.front() == '{' || token.front() == ' ' || token.front() == '\t')) {
            token.erase(0, 1);
          }
          while (!token.empty() && (token.back() == '}' || token.back() == ',' || token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
          }
          
          // Now parse the clean token
          shift_coeffs[i] = std::stod(token);
        } catch (const std::exception& e) {
          error->one(FLERR, "Error parsing shift_coeffs[{}]: error={}", i, e.what());
        }
        if (i < 5) {  // Log first few coefficients for debugging
          utils::logmesg(lmp, "shift_coeffs[{}] = {}\n", i, shift_coeffs[i]);
        }
      }
      // Move to next line after successful read
      // CRITICAL FIX: Update current_raw_line to the newly read line for proper scal_coeffs parsing
      current_raw_line = std::string(tfr.next_line());
      line_tokens = ValueTokenizer(current_raw_line, separators);
      keyword = line_tokens.next_string();
      utils::logmesg(lmp, "shift_coeffs read for {} species, next keyword: '{}'\n", species_count, keyword);
    } else {
      // Default values if shift_coeffs not present
      utils::logmesg(lmp, "shift_coeffs keyword not found (current keyword: '{}'), using defaults\n", keyword);
      for (int i = 0; i < species_count; i++) {
        shift_coeffs[i] = -1.0;
      }
    }

    // Read scal_coeffs (SUS2-MLIP specific, optional) - robust detection with error handling
    // FIXED: Parse values from the same line (keyword = {values} format)
    // CRITICAL FIX: Use custom separators that don't split scientific notation (e.g., e+00)
    if (keyword == "scal_coeffs") {
      utils::logmesg(lmp, "Found scal_coeffs keyword, reading coefficients...\n");
      
      // CRITICAL FIX: The scal_coeffs values are on the SAME line as the keyword, not on the next line!
      // Use the previously saved current_raw_line which contains the full scal_coeffs line
      std::string scal_raw_line = current_raw_line;
      
      try {
      // Re-parse the line with custom separators that preserve scientific notation
      // Don't include +, -, * as separators since they appear in scientific notation
      // CRITICAL FIX: Don't include "=" as a separator to avoid skipping the first value
      std::string scal_separators = " {},\t\r\n";
      ValueTokenizer scal_tokens(scal_raw_line, scal_separators);
      
      // Skip "scal_coeffs" keyword (the "=" will be handled by next_string automatically)
      scal_tokens.next_string();  // Skip "scal_coeffs"
      scal_tokens.next_string();  // This will skip the "=" token
        
        // Read all scal_coeffs directly into the existing unified regression_coeffs array
        // with error handling for each coefficient
        for (int i = 0; i < scal_coeffs_count_local; i++) {
          std::string token = scal_tokens.next_string();
          
          // Remove possible braces and trim whitespace
          while (!token.empty() && (token.front() == '{' || token.front() == ' ' || token.front() == '\t')) {
            token.erase(0, 1);
          }
          while (!token.empty() && (token.back() == '}' || token.back() == ',' || token.back() == ' ' || token.back() == '\t')) {
            token.pop_back();
          }
          
          // Parse the clean token
          scal_coeffs[i] = std::stod(token);
          if (i < 5) {  // Log first few coefficients for debugging
            utils::logmesg(lmp, "scal_coeffs[{}] = {}\n", i, scal_coeffs[i]);
          }
        }
        
        utils::logmesg(lmp, "scal_coeffs read successfully, count = {}\n", scal_coeffs_count_local);
      } catch (const std::exception& e) {
        error->one(FLERR, "MTP error: Exception while reading scal_coeffs: {}", e.what());
      } catch (...) {
        error->one(FLERR, "MTP error: Unknown exception while reading scal_coeffs");
      }
      
      // Move to next line after successful read
      line_tokens = ValueTokenizer(tfr.next_line(), separators);
      keyword = line_tokens.next_string();
    } else {
      // Default values if scal_coeffs not present - use existing unified array
      utils::logmesg(lmp, "WARNING: scal_coeffs keyword not found (current keyword: '{}'), using defaults\n", keyword);
      
      // CRITICAL: Validate radial_basis before accessing its members
      if (!radial_basis) {
        error->one(FLERR, "MTP error: radial_basis is null when calculating default scaling coefficients");
      }
      
      double cutoff_range = radial_basis->max_cutoff - radial_basis->min_cutoff;
      if (cutoff_range <= 0.0) {
        error->one(FLERR, "MTP error: Invalid cutoff range (max={}, min={}) for scaling coefficients",
                   radial_basis->max_cutoff, radial_basis->min_cutoff);
      }
      
      double s1_default = 3.3 * 2 / cutoff_range;
      double s2_default = radial_basis->min_cutoff;
      utils::logmesg(lmp, "Default values: s1 = {}, s2 = {}, K_scaling = {}\n", s1_default, s2_default, K_scaling);
      
      // Use existing unified array structure - no additional memory allocation needed
      for (int j = 0; j < K_scaling; j++) {
        for (int i = 0; i < pairs_count; i++) {
          scal_coeffs[j * 2 * pairs_count + i] = s1_default;                    // s1 coefficients
          scal_coeffs[j * 2 * pairs_count + pairs_count + i] = s2_default;      // s2 coefficients
        }
      }
      utils::logmesg(lmp, "scal_coeffs initialized to defaults using existing unified array\n");
    }

    // Read radial basis coeffs (SUS2-MLIP format) - robust detection using existing unified array
    // CRITICAL: Only read 0-0 pair coefficients (all pairs share these coefficients)
    if (keyword == "radial_coeffs") {
      utils::logmesg(lmp, "Found radial_coeffs keyword, reading coefficients...\n");
      
      bool found_zero_zero = false;
      bool reached_next_section = false;
      while (!reached_next_section) {
        std::string line = std::string(tfr.next_line());
        
        // Check if we've reached the next section (alpha_moments_count)
        if (line.find("alpha_moments_count") != std::string::npos ||
            line.find("env_gate_type") != std::string::npos) {
          saved_line = line;  // Save this line for later processing
          saved_line_valid = true;
          reached_next_section = true;
          break;
        }
        
        // Skip empty lines
        size_t pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        
        // Check if this is a pair header (starts with digit or sign)
        char c = line[pos];
        if (!((c >= '0' && c <= '9') || c == '+' || c == '-')) continue;
        
        // Parse pair type
        line_tokens = ValueTokenizer(line, separators + "-");
        int type1, type2;
        try {
          type1 = line_tokens.next_int();
          type2 = line_tokens.next_int();
        } catch (...) { continue; }
        
        utils::logmesg(lmp, "Found pair header: {}-{}\n", type1, type2);
        
        // Only process 0-0 pair (all pairs share these coefficients)
        if (type1 == 0 && type2 == 0) {
          found_zero_zero = true;
          
          for (int mu = 0; mu < radial_func_count; mu++) {
            std::string coeff_line = std::string(tfr.next_line());
            line_tokens = ValueTokenizer(coeff_line, separators + "{,}");
            for (int ri = 0; ri < radial_basis_size + species_count; ri++) {
              std::string token = line_tokens.next_string();
              
              // Remove possible braces
              if (!token.empty() && token.front() == '{') token.erase(0, 1);
              if (!token.empty() && token.back() == '}') token.pop_back();
              
              try {
                radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = std::stod(token);
                if (mu < 2 && ri < 3) {  // Log first few coefficients for debugging
                  utils::logmesg(lmp, "radial_basis_coeffs[mu={}, ri={}] = {}\n", mu, ri,
                               radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri]);
                }
              } catch (const std::exception& e) {
                error->one(FLERR, "Error parsing radial coefficient at mu={}, ri={}: {}", mu, ri, e.what());
              }
            }
          }
          utils::logmesg(lmp, "Successfully read 0-0 pair radial coefficients\n");
        } else {
          // Skip other pairs - skip all their coefficient lines
          utils::logmesg(lmp, "Skipping pair {}-{} (only 0-0 needed)\n", type1, type2);
          for (int j = 0; j < radial_func_count; j++) {
            std::string(tfr.next_line());  // Skip coefficient lines
          }
        }
      }
      
      if (!found_zero_zero) {
        error->one(FLERR, "MTP error: No 0-0 radial coefficients found in file");
      }
      
      // Update keyword for next section
      if (reached_next_section && saved_line_valid) {
        line_tokens = ValueTokenizer(saved_line, separators);
        keyword = line_tokens.next_string();
      }
      
      utils::logmesg(lmp, "radial_coeffs read successfully (0-0 pair only)\n");
    } else {
      // Initialize default values if not present - use existing unified array
      utils::logmesg(lmp, "radial_coeffs keyword not found, using defaults\n");
      for (int mu = 0; mu < radial_func_count; mu++) {
        for (int ri = 0; ri < radial_basis_size + species_count; ri++) {
          if (ri < radial_basis_size) {
            radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = 1e-2;
          } else {
            radial_basis_coeffs[mu * (radial_basis_size + species_count) + ri] = 1.1;
          }
        }
      }
      utils::logmesg(lmp, "radial_basis_coeffs initialized to defaults using existing unified array\n");
    }

    if (env_gate_enabled) {
      if (!saved_line_valid || saved_line.find("env_gate_type") == std::string::npos)
        error->one(FLERR, "version = 2.0.0 model must contain env_gate_type section.");

      line_tokens = ValueTokenizer(saved_line, separators);
      keyword = line_tokens.next_string();
      std::string env_gate_type = line_tokens.next_string();
      if (env_gate_type != "centered_tanh_screen" && env_gate_type != "centered_exp_screen")
        error->one(FLERR, "Unsupported env_gate_type: {}", env_gate_type);

      line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
      keyword = line_tokens.next_string();
      if (keyword != "env_gate_cutoff_ratio")
        error->one(FLERR, "Error reading env_gate_cutoff_ratio.");
      env_gate_cutoff_ratio = line_tokens.next_double();

      line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
      keyword = line_tokens.next_string();
      if (keyword == "env_gate_activation_on_ratio") {
        env_gate_activation_on_ratio = line_tokens.next_double();
        if (!std::isfinite(env_gate_activation_on_ratio) ||
            env_gate_activation_on_ratio < 0.0 ||
            env_gate_activation_on_ratio >= 1.0)
          error->one(FLERR, "env_gate_activation_on_ratio should be in [0, 1).");
        line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
        keyword = line_tokens.next_string();
      } else {
        env_gate_activation_on_ratio = kEnvGateDefaultActivationOnRatio;
      }
      if (keyword != "env_gate_channel_count")
        error->one(FLERR, "Error reading env_gate_channel_count.");
      env_gate_channel_count = line_tokens.next_int();
      if (env_gate_channel_count != kEnvGateChannels)
        error->one(FLERR, "centered_tanh_screen env-gate requires 6 channels.");

      std::string gate_line = std::string(tfr.next_line());
      std::string gate_separators = " ={},\t\r\n";
      line_tokens = ValueTokenizer(gate_line, gate_separators);
      keyword = line_tokens.next_string();
      if (keyword == "env_gate_scal_coeffs") {
        for (int i = 0; i < 2 * pairs_count; ++i) {
          (void)std::stod(line_tokens.next_string());
        }
        gate_line = std::string(tfr.next_line());
        line_tokens = ValueTokenizer(gate_line, separators);
        keyword = line_tokens.next_string();
      }
      if (keyword != "env_gate_lambda_raw")
        error->one(FLERR, "Error reading env_gate_lambda_raw.");
      {
        std::string lambda_values = gate_line;
        const std::size_t equals_pos = lambda_values.find('=');
        if (equals_pos != std::string::npos)
          lambda_values = lambda_values.substr(equals_pos + 1);
        for (char &ch : lambda_values) {
          if (ch == '{' || ch == '}' || ch == ',') ch = ' ';
        }
        std::istringstream lambda_stream(lambda_values);
        std::vector<double> lambda_raw_values;
        double lambda_value = 0.0;
        while (lambda_stream >> lambda_value)
          lambda_raw_values.push_back(lambda_value);
        if (lambda_raw_values.empty())
          error->one(FLERR, "Error reading env_gate_lambda_raw.");
        if (lambda_raw_values.size() == 1) {
          for (int type_i = 0; type_i < species_count; ++type_i)
            regression_coeffs[env_gate_lambda_raw_offset + type_i] = lambda_raw_values[0];
        } else {
          if (static_cast<int>(lambda_raw_values.size()) != species_count)
            error->one(FLERR, "env_gate_lambda_raw count should be 1 or species_count.");
          for (int type_i = 0; type_i < species_count; ++type_i)
            regression_coeffs[env_gate_lambda_raw_offset + type_i] = lambda_raw_values[type_i];
        }
      }

      std::string density_line = std::string(tfr.next_line());
      std::string density_separators = " ={},\t\r\n";
      ValueTokenizer density_tokens(density_line, density_separators);
      keyword = density_tokens.next_string();
      if (keyword != "env_gate_log_density_coeffs")
        error->one(FLERR, "Error reading env_gate_log_density_coeffs.");
      for (int type_i = 0; type_i < species_count; ++type_i)
        for (int q = 0; q < env_gate_channel_count; ++q)
          regression_coeffs[env_gate_log_density_coeffs_offset + type_i * env_gate_channel_count + q] =
              std::stod(density_tokens.next_string());

      saved_line = std::string(tfr.next_line());
      saved_line_valid = true;
    }

    // Setup cutoff values
    double rcutmaxsq = radial_basis->max_cutoff * radial_basis->max_cutoff;
    for (int i = 0; i < species_count; i++) {
      for (int j = 0; j < species_count; j++) {
        setflag[i + 1][j + 1] = 1;
        cutsq[i + 1][j + 1] = rcutmaxsq;
      }
    }

    // CRITICAL FIX: Robust search for alpha_moments_count with saved_line and attempt limits
    // This handles cases where the keyword might be embedded in other content or on unexpected lines
    bool found_alpha_moments = false;
    int max_attempts = 1000;  // Prevent infinite loops
    int attempt_count = 0;
    
    utils::logmesg(lmp, "Searching for alpha_moments_count in MTP file...\n");
    if (saved_line_valid && saved_line.find("alpha_moments_count") != std::string::npos) {
      found_alpha_moments = true;
      utils::logmesg(lmp, "Using saved alpha_moments_count line: '{}'\n", saved_line);
    }
    
    // First, try to find alpha_moments_count using saved_line approach
    // This handles cases where we might have already read past the line
    while (!found_alpha_moments && attempt_count < max_attempts) {
      attempt_count++;
      try {
        std::string next_line = std::string(tfr.next_line());
        
        // Check for empty lines
        if (next_line.empty()) {
          utils::logmesg(lmp, "WARNING: Empty line encountered while searching for alpha_moments_count (attempt {})\n", attempt_count);
          continue;
        }
        
        // Check if this line contains alpha_moments_count
        if (next_line.find("alpha_moments_count") != std::string::npos) {
          saved_line = next_line;
          saved_line_valid = true;
          found_alpha_moments = true;
          utils::logmesg(lmp, "Found alpha_moments_count in line (attempt {}): '{}'\n", attempt_count, saved_line);
          break;
        }
        
        // Trim leading whitespace and check if line starts with a digit (might be pair header)
        size_t pos = next_line.find_first_not_of(" \t\r\n");
        if (pos != std::string::npos) {
          char c = next_line[pos];
          // Skip lines that look like pair headers (start with digit or sign)
          if ((c >= '0' && c <= '9') || c == '+' || c == '-') {
            // This might be from radial_coeffs section, skip it
            continue;
          }
        }
        
      } catch (const std::exception& e) {
        error->one(FLERR, std::string("MTP error: Exception while searching for alpha_moments_count: ") + e.what());
        break;
      } catch (...) {
        error->one(FLERR, "MTP error: Unknown exception while searching for alpha_moments_count (attempt {})", attempt_count);
        break;
      }
    }
    
    // Check if we found alpha_moments_count
    if (!found_alpha_moments) {
      if (attempt_count >= max_attempts) {
        error->one(FLERR, "MTP error: Could not find alpha_moments_count after {} attempts. File may be corrupted or missing required field.", max_attempts);
      } else {
        error->one(FLERR, "Error reading MTP file. Alpha moment count not found.");
      }
    }
    
    // Parse alpha_moments_count from saved_line with validation
    if (saved_line_valid) {
      std::istringstream iss(saved_line);
      std::string keyword_str, equals_str;
      int value;
      iss >> keyword_str >> equals_str >> value;
      
      // Validate parsed values
      if (keyword_str != "alpha_moments_count" || equals_str != "=") {
        error->one(FLERR, "MTP error: Invalid alpha_moments_count format. Expected 'alpha_moments_count = value', got: '{}'", saved_line);
      }
      
      // CRITICAL: Validate the value - alpha_moment_count must be positive
      if (value <= 0) {
        error->one(FLERR, "MTP error: Invalid alpha_moments_count value: {}. Must be positive integer.", value);
      }
      
      // Check for unreasonably large values
      if (value > 1000000) {
        utils::logmesg(lmp, "WARNING: alpha_moments_count is unusually large: {}. File may have issues.\n", value);
      }
      
      alpha_moment_count = value;
      utils::logmesg(lmp, "Successfully parsed alpha_moments_count = {}\n", alpha_moment_count);
    } else {
      error->one(FLERR, "MTP error: alpha_moments_count found but could not be parsed from: '{}'", saved_line);
    }
    
    memory->create(moment_tensor_vals, alpha_moment_count, "moment_tensor_vals");
    memory->create(nbh_energy_ders_wrt_moments, alpha_moment_count, "nbh_energy_ders_wrt_moments");

    // Get the basic alpha count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic_count")
      error->one(FLERR, "Error reading MTP file. Alpha moment count not found.");
    alpha_index_basic_count = line_tokens.next_int();

    // Read the basic alphas
    int radial_func_max = 0;
    tfr.set_bufsize(
        (alpha_index_basic_count * 20 + 20) *
        sizeof(
            char));    // Adjust the buffer size. This needed to ensure cross-compatability since the MLIP files stores all the alpha indicies on the same line.
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_basic")
      error->one(FLERR, "Error reading MTP file. Alpha index basic not found.");
    memory->create(alpha_index_basic, alpha_index_basic_count, 4, "alpha_index_basic");
    for (int i = 0; i < alpha_index_basic_count; i++) {
      for (int j = 0; j < 4; j++) {
        int index = line_tokens.next_int();
        alpha_index_basic[i][j] = index;
      }
      if (alpha_index_basic[i][0] > radial_func_max) radial_func_max = alpha_index_basic[i][0];
    }
    if (radial_func_max != radial_func_count - 1)    //Index validity check
      error->one(FLERR, "Wrong number of radial functions specified!");

    //Precompute the maximum alpha basic index
    max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; i++)
      max_alpha_index_basic =
          std::max(max_alpha_index_basic,
                   alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
    max_alpha_index_basic++;    // Add 1 to account for zeroth order indicies

    // Get the alpha times count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times_count")
      error->one(FLERR, "Error reading MTP file. Alpha index times count not found.");
    alpha_index_times_count = line_tokens.next_int();

    // Read the alphas times
    tfr.set_bufsize(
        (alpha_index_times_count * 32 + 20) *
        sizeof(
            char));    // Adjust the buffer size. This needed to ensure cross-compatability since the MLIP files stores all the alpha indicies on the same line.
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_index_times")
      error->one(FLERR, "Error reading MTP file. Alpha index times not found.");
    memory->create(alpha_index_times, alpha_index_times_count, 4, "alpha_index_times");
    for (int i = 0; i < alpha_index_times_count; i++) {
      for (int j = 0; j < 4; j++) { alpha_index_times[i][j] = line_tokens.next_int(); }
    }

    // Get the alpha scalar count
    line_tokens = ValueTokenizer(tfr.next_line(), separators);
    keyword = line_tokens.next_string();
    if (keyword != "alpha_scalar_moments")
      error->one(FLERR, "Error reading MTP file. Alpha scalar moment count not found.");
    alpha_scalar_count = line_tokens.next_int();

    //Read the alpha moment mappings
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "alpha_moment_mapping")
      error->one(FLERR, "Error reading MTP file. Alpha moment mappings not found.");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");
    for (int i = 0; i < alpha_scalar_count; i++) {
      alpha_moment_mapping[i] = line_tokens.next_int();
    }

    //Read the species coefficients
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "species_coeffs")
      error->one(FLERR, "Error reading MTP file. Species coefficients not found.");
    memory->create(species_coeffs, species_count, "species_coeffs");
    for (int i = 0; i < species_count; i++) { species_coeffs[i] = line_tokens.next_double(); }

    //Read the linear MTP basis coefficients
    line_tokens = ValueTokenizer(tfr.next_line(), separators + "{},");

    keyword = line_tokens.next_string();
    if (keyword != "moment_coeffs")
      error->one(FLERR, "Error reading MTP file. Moment coefficients not found.");
    memory->create(linear_coeffs, alpha_scalar_count, "moment_coeffs");
    for (int i = 0; i < alpha_scalar_count; i++) { linear_coeffs[i] = line_tokens.next_double(); }
  }    // Proc 0

  // ---------- CRITICAL FIX: Broadcast ALL single values BEFORE allocating arrays ----------
  // This ensures all procs have correct parameter values before they are used
  //Radial Basis Set Type First
  MPI_Bcast(&radial_basis_type_index, 1, MPI_INT, 0, world);

  //Then Single Values
  MPI_Bcast(&scaling, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&species_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&L_max, 1, MPI_INT, 0, world);
  MPI_Bcast(&K_scaling, 1, MPI_INT, 0, world);
  MPI_Bcast(&radial_basis_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&radial_func_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_moment_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_index_basic_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&max_alpha_index_basic, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_index_times_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&alpha_scalar_count, 1, MPI_INT, 0, world);
  MPI_Bcast(&regression_coeffs_count, 1, MPI_INT, 0, world);
  int env_gate_enabled_int = env_gate_enabled ? 1 : 0;
  MPI_Bcast(&env_gate_enabled_int, 1, MPI_INT, 0, world);
  env_gate_enabled = (env_gate_enabled_int != 0);
  MPI_Bcast(&env_gate_cutoff_ratio, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&env_gate_activation_on_ratio, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&env_gate_channel_count, 1, MPI_INT, 0, world);

  // Broadcast scaling_map string
  int scaling_map_len = scaling_map.length();
  MPI_Bcast(&scaling_map_len, 1, MPI_INT, 0, world);
  if (comm->me != 0) scaling_map.resize(scaling_map_len);
  MPI_Bcast(&scaling_map[0], scaling_map_len, MPI_CHAR, 0, world);

  // ---------- Now all procs have correct parameter values ----------
  // Pre-allocate arrays on ALL procs (including proc 0)
  int np1 = ((atom && atom->ntypes > species_count) ? atom->ntypes : species_count) + 1;

  //Working buffers
  memory->create(inv_dist_powers, max_alpha_index_basic, "inv_dist_powers");
  memory->create(coord_powers_x, max_alpha_index_basic, "coord_powers_x");
  memory->create(coord_powers_y, max_alpha_index_basic, "coord_powers_y");
  memory->create(coord_powers_z, max_alpha_index_basic, "coord_powers_z");
  memory->create(alpha_basic_mu, alpha_index_basic_count, "alpha_basic_mu");
  memory->create(alpha_basic_a0, alpha_index_basic_count, "alpha_basic_a0");
  memory->create(alpha_basic_a1, alpha_index_basic_count, "alpha_basic_a1");
  memory->create(alpha_basic_a2, alpha_index_basic_count, "alpha_basic_a2");
  memory->create(alpha_basic_norm_rank, alpha_index_basic_count, "alpha_basic_norm_rank");
  memory->create(alpha_times_a0, alpha_index_times_count, "alpha_times_a0");
  memory->create(alpha_times_a1, alpha_index_times_count, "alpha_times_a1");
  memory->create(alpha_times_multiplier, alpha_index_times_count, "alpha_times_multiplier");
  memory->create(alpha_times_out, alpha_index_times_count, "alpha_times_out");
  memory->create(weighted_basic_moment_ders, alpha_index_basic_count, "weighted_basic_moment_ders");
  
  // Allocate radial_vals/ders for final storage only (one value per mu)
  // Temporary arrays are now created locally in compute() function
  memory->create(radial_vals, radial_func_count, "radial_vals");
  memory->create(radial_ders, radial_func_count, "radial_ders");


  // Now allocate arrays that depend on broadcasted values
  if (comm->me != 0) {    // Non-zero proc
    if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV) {
      radial_basis = new RBChebyshev(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_S) {
      radial_basis = new RBChebyshev_s(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SS) {
      radial_basis = new RBChebyshev_ss(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS) {
      radial_basis = new RBChebyshev_sss(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::CHEBYSHEV_SSS_LMP) {
      radial_basis = new RBChebyshev_sss_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P) {
      radial_basis = new RBLaguerre_log1p(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_LMP) {
      radial_basis = new RBLaguerre_log1p_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS) {
      radial_basis = new RBLaguerre_log1p_pos(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::LAGUERRE_LOG1P_POS_LMP) {
      radial_basis = new RBLaguerre_log1p_pos_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    } else if (radial_basis_type_index == SUS2RadialMTPBasis::JACOBI_SSS_LMP) {
      radial_basis = new RBJacobi_sss_lmp(radial_basis_size, lmp);
      radial_basis->scaling = scaling;
    }
    //Flags
    memory->create(cutsq, np1, np1, "pair:cutsq");
    memory->create(setflag, np1, np1, "pair:setflag");
    for (int i = 0; i < np1; ++i) {
      for (int j = 0; j < np1; ++j) {
        cutsq[i][j] = 0.0;
        setflag[i][j] = 0;
      }
    }

    //Alpha indicies
    memory->create(alpha_index_basic, alpha_index_basic_count, 4, "alpha_index_basic");
    memory->create(alpha_index_times, alpha_index_times_count, 4, "alpha_index_times");
    memory->create(alpha_moment_mapping, alpha_scalar_count, "alpha_moment_mapping");

    //Working buffers
    memory->create(moment_tensor_vals, alpha_moment_count, "moment_tensor_vals");
    memory->create(nbh_energy_ders_wrt_moments, alpha_moment_count, "nbh_energy_ders_wrt_moments");
    // CRITICAL FIX: Initialize Jacobian pointers to nullptr for all MPI procs
    // These arrays will be allocated with memory->grow during calculation
    moment_jacobian_x = nullptr;
    moment_jacobian_y = nullptr;
    moment_jacobian_z = nullptr;
    within_cutoff = nullptr;
    radial_list = nullptr;
    radial_der_list = nullptr;
    jac_size = 0;

    //Coefficients - allocate unified array
    memory->create(regression_coeffs, regression_coeffs_count, "regression_coeffs");
    
    // Set offset pointers
    shift_coeffs_offset = 0;
    scal_coeffs_offset = species_count;
    int pairs_count = species_count * species_count;
    int scal_coeffs_count_local = 2 * pairs_count * K_scaling;
    radial_coeffs_offset = scal_coeffs_offset + scal_coeffs_count_local;
    const int radial_coeff_count_local = radial_func_count * (radial_basis_size + species_count);
    env_gate_coeffs_offset = radial_coeffs_offset + radial_coeff_count_local;
    env_gate_lambda_raw_offset = env_gate_coeffs_offset;
    env_gate_log_density_coeffs_offset = env_gate_coeffs_offset + species_count;
    
    shift_coeffs = &regression_coeffs[shift_coeffs_offset];
    scal_coeffs = &regression_coeffs[scal_coeffs_offset];
    radial_basis_coeffs = &regression_coeffs[radial_coeffs_offset];
    
    memory->create(linear_coeffs, alpha_scalar_count, "linear_coeffs");
    memory->create(species_coeffs, species_count, "species_coeffs");
    
    // SUS2-MLIP mapping arrays
    memory->create(mu_to_K, radial_func_count, "mu_to_K");
    memory->create(mu_to_sigma, radial_func_count, "mu_to_sigma");
    
  }

  //We can then populate the cutoffs
  MPI_Bcast(&radial_basis->min_cutoff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&radial_basis->max_cutoff, 1, MPI_DOUBLE, 0, world);
  min_cutoff = radial_basis->min_cutoff;
  max_cutoff = radial_basis->max_cutoff;
  max_cutoff_sq = max_cutoff * max_cutoff;

  //Now we B Cast into arrays
  //Flags
  MPI_Bcast(&cutsq[0][0], np1 * np1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&setflag[0][0], np1 * np1, MPI_INT, 0, world);

  // Alphas
  MPI_Bcast(&alpha_index_basic[0][0], alpha_index_basic_count * 4, MPI_INT, 0, world);
  MPI_Bcast(&alpha_index_times[0][0], alpha_index_times_count * 4, MPI_INT, 0, world);
  MPI_Bcast(alpha_moment_mapping, alpha_scalar_count, MPI_INT, 0, world);
  for (int i = 0; i < alpha_index_basic_count; i++) {
    alpha_basic_mu[i] = alpha_index_basic[i][0];
    alpha_basic_a0[i] = alpha_index_basic[i][1];
    alpha_basic_a1[i] = alpha_index_basic[i][2];
    alpha_basic_a2[i] = alpha_index_basic[i][3];
    alpha_basic_norm_rank[i] = alpha_basic_a0[i] + alpha_basic_a1[i] + alpha_basic_a2[i];
  }
  for (int i = 0; i < alpha_index_times_count; i++) {
    alpha_times_a0[i] = alpha_index_times[i][0];
    alpha_times_a1[i] = alpha_index_times[i][1];
    alpha_times_multiplier[i] = alpha_index_times[i][2];
    alpha_times_out[i] = alpha_index_times[i][3];
  }

  //Working buffers
  //Preassign constant values for dist powers and coord powers. Other buffers can be uninited.
  inv_dist_powers[0] = 1.0;
  if (max_alpha_index_basic > 0) {
    coord_powers_x[0] = 1.0;
    coord_powers_y[0] = 1.0;
    coord_powers_z[0] = 1.0;
  }

  // Coefficients - broadcast the entire unified regression_coeffs array
  MPI_Bcast(regression_coeffs, regression_coeffs_count, MPI_DOUBLE, 0, world);
  MPI_Bcast(linear_coeffs, alpha_scalar_count, MPI_DOUBLE, 0, world);
  MPI_Bcast(species_coeffs, species_count, MPI_DOUBLE, 0, world);
  
  // SUS2-MLIP mapping arrays
  MPI_Bcast(mu_to_K, radial_func_count, MPI_INT, 0, world);
  MPI_Bcast(mu_to_sigma, radial_func_count, MPI_INT, 0, world);

  do_list = false;
  list_grid_size = 0;
  actual_dr = 0.0;
  inv_dr = 0.0;
  total_list_elements = 0;
  used_species_count = species_count;
  used_pair_count = species_count * species_count;
  memory->create(pair_to_table_index, species_count * species_count, "pair_to_table_index");

  std::vector<int> used_species_flags(species_count, 1);
  if (atom && atom->type && atom->natoms > 0) {
    std::fill(used_species_flags.begin(), used_species_flags.end(), 0);
    for (int i = 0; i < atom->nlocal; i++) {
      const int atype = atom->type[i] - 1;
      if (atype >= 0 && atype < species_count) {
        used_species_flags[atype] = 1;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, used_species_flags.data(), species_count, MPI_INT, MPI_MAX, world);

    used_species_count = 0;
    for (int i = 0; i < species_count; i++) {
      used_species_count += used_species_flags[i];
    }
    if (used_species_count == 0) {
      std::fill(used_species_flags.begin(), used_species_flags.end(), 1);
      used_species_count = species_count;
    }
  }

  used_pair_count = 0;
  for (int i = 0; i < species_count; i++) {
    for (int j = 0; j < species_count; j++) {
      const int shift = i * species_count + j;
      if (used_species_flags[i] && used_species_flags[j]) {
        pair_to_table_index[shift] = used_pair_count++;
      } else {
        pair_to_table_index[shift] = -1;
      }
    }
  }

  if (uses_preinterpolation_table(radial_basis_type_index)) {
    do_list = true;
    list_grid_size = static_cast<int>(std::ceil(max_cutoff / requested_tabstep)) + 1;
    if (list_grid_size < 2) list_grid_size = 2;
    actual_dr = max_cutoff / static_cast<double>(list_grid_size - 1);
    inv_dr = 1.0 / actual_dr;

    if (comm->me == 0) {
      std::ostringstream used_species_stream;
      bool first_used_species = true;
      for (int i = 0; i < species_count; i++) {
        if (!used_species_flags[i]) continue;
        if (!first_used_species) used_species_stream << ", ";
        used_species_stream << (i + 1);
        first_used_species = false;
      }
      utils::logmesg(
          lmp,
          "Building Radial List, requested_tabstep={} A, actual_dr={} A, grid_points={}, max_cutoff={} A\n",
          requested_tabstep, actual_dr, list_grid_size, max_cutoff);
      utils::logmesg(
          lmp,
          "  - Precomputing {} of {} species pairs from used atom types [{}]\n",
          used_pair_count, species_count * species_count, used_species_stream.str());
    }

    memory->create(radial_list, used_pair_count, list_grid_size, radial_func_count,
                   "radial_list");
    memory->create(radial_der_list, used_pair_count, list_grid_size, radial_func_count,
                   "radial_der_list");
    total_list_elements =
        static_cast<size_t>(used_pair_count) * list_grid_size * radial_func_count;
    std::fill_n(&radial_list[0][0][0], total_list_elements, 0.0);
    std::fill_n(&radial_der_list[0][0][0], total_list_elements, 0.0);

    const int C = species_count;
    const int R = radial_basis_size;
    const double dr = actual_dr;
    double factor;
    for (int i = 0; i < C; i++) {
      for (int j = 0; j < C; j++) {
        const int table_index = pair_to_table_index[i * C + j];
        if (table_index < 0) continue;
        for (int n = 0; n < list_grid_size; n++) {
          const double dist = dr * n;
          for (int mu = 0; mu < radial_func_count; mu++) {
            const int k_ = mu_to_K[mu];
            const int sigma = mu_to_sigma[mu];

            radial_basis->calc_radial_basis_ders(
                dist, regression_coeffs[C + 2 * k_ * C * C + C * i + j],
                regression_coeffs[C + 2 * k_ * C * C + C * C + C * i + j], sigma);

            for (int xi = 0; xi < R; xi++) {
              factor = regression_coeffs[C + 2 * C * C * K_scaling + mu * (R + C) + xi] *
                       regression_coeffs[C + 2 * C * C * K_scaling + R + i] *
                       regression_coeffs[C + 2 * C * C * K_scaling + R + j];
              radial_list[table_index][n][mu] += radial_basis->radial_basis_vals[xi] * factor;
              radial_der_list[table_index][n][mu] += radial_basis->radial_basis_ders[xi] * factor;
            }
          }
        }
      }
    }

    if (env_gate_enabled) {
      memory->create(env_gate_rho_list, C, list_grid_size, "env_gate_rho_list");
      memory->create(env_gate_rho_der_list, C, list_grid_size, "env_gate_rho_der_list");
      for (int i = 0; i < C; i++) {
        for (int n = 0; n < list_grid_size; n++) {
          env_gate_rho_list[i][n] = 0.0;
          env_gate_rho_der_list[i][n] = 0.0;
        }
      }

      const double r_env = env_gate_cutoff_ratio * max_cutoff;
      if (r_env <= 0.0) error->all(FLERR, "Invalid env_gate_cutoff_ratio in SUS2 model.");
      for (int i = 0; i < C; i++) {
        double density_coeffs[6];
        for (int q = 0; q < env_gate_channel_count; ++q)
          density_coeffs[q] =
              env_gate_density_coeff(regression_coeffs[env_gate_log_density_coeffs_offset + i * env_gate_channel_count + q]);
        for (int n = 0; n < list_grid_size; n++) {
          const double dist = dr * n;
          if (dist <= 0.0 || dist >= r_env) continue;
          const double y = dist / r_env;
          const double dy_dr = 1.0 / r_env;
          const double cutoff = 1.0 - dist / r_env;
          const double f_env = cutoff * cutoff;
          const double df_dr = -2.0 * cutoff / r_env;
          double basis[6];
          double basis_der[6];
          bernstein_degree5(y, basis, basis_der);
          for (int q = 0; q < env_gate_channel_count; ++q) {
            const double weighted_basis = f_env * basis[q];
            const double weighted_basis_der = df_dr * basis[q] + f_env * basis_der[q] * dy_dr;
            env_gate_rho_list[i][n] += density_coeffs[q] * weighted_basis;
            env_gate_rho_der_list[i][n] += density_coeffs[q] * weighted_basis_der;
          }
        }
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (comm->me == 0) {
      utils::logmesg(lmp, "Radial List Completed\n");
    }
  } else if (tabstep_set_by_user && comm->me == 0) {
    utils::logmesg(
        lmp,
        "Ignoring tabstep={} A because radial_basis_type={} does not use the preinterpolation table.\n",
        requested_tabstep, radial_basis_type_str);
  }

  allocated = 1;
}

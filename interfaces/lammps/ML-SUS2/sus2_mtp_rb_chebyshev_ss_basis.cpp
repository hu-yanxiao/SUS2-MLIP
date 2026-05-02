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
// Contributing author: Based on SUS2-MLIP-1.1 super-linear radial basis functions
//

#include "sus2_mtp_rb_chebyshev_ss_basis.h"
#include "error.h"
#include "memory.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <utility>

using namespace LAMMPS_NS;

namespace {

constexpr double kLaguerreMinRho = 1.0e-8;
constexpr double kLaguerrePositiveParamFloor = 1.0e-6;
constexpr double kJacobiEndpointEps = 1.0e-12;
constexpr int kJacobiMaxIndexedBlock = 5;

double stable_softplus(double x)
{
  if (x > 40.0) return x;
  if (x < -40.0) return exp(x);
  return log1p(exp(x));
}

double laguerre_positive_param(double raw)
{
  return kLaguerrePositiveParamFloor + stable_softplus(raw);
}

void laguerre_log1p_positive_calc(SUS2RadialMTPBasis &basis, double dist, double scal_raw,
                                  double s_raw, bool include_distance_derivative)
{
  const double scal = laguerre_positive_param(scal_raw);
  const double rho = laguerre_positive_param(s_raw);
  const double u = scal * log1p(dist / rho);
  const double u_r = scal / (rho + dist);
  const double Dr = dist - basis.max_cutoff;
  const double cutoff = Dr * Dr;
  const double cutoff_der = 2.0 * Dr;
  const double exp_factor = exp(-0.5 * u);

  double phi_prev = 0.0;
  double dphi_prev = 0.0;
  double phi_curr = basis.scaling * cutoff * exp_factor;
  double dphi_curr = basis.scaling * cutoff_der * exp_factor - 0.5 * u_r * phi_curr;

  basis.radial_basis_vals[0] = phi_curr;
  if (include_distance_derivative) basis.radial_basis_ders[0] = dphi_curr;

  for (int n = 0; n < basis.size - 1; ++n) {
    const double inv_np1 = 1.0 / (n + 1.0);
    const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
    const double prev_coeff = n * inv_np1;
    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;
    double dphi_next = 0.0;

    if (include_distance_derivative) {
      dphi_next = -u_r * inv_np1 * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;
      basis.radial_basis_ders[n + 1] = dphi_next;
    }

    basis.radial_basis_vals[n + 1] = phi_next;
    phi_prev = phi_curr;
    dphi_prev = dphi_curr;
    phi_curr = phi_next;
    dphi_curr = dphi_next;
  }
}

std::pair<int, int> jacobi_alpha_beta_for_block(int k, LAMMPS *lmp)
{
  static const int kAlphaBetaTable[kJacobiMaxIndexedBlock + 1][2] = {
      {0, 0},
      {1, 0},
      {1, 1},
      {2, 0},
      {2, 1},
      {2, 2},
  };

  if (k < 0 || k > kJacobiMaxIndexedBlock) {
    lmp->error->one(
        FLERR,
        "RBJacobi_sss_lmp supports only six indexed blocks: "
        "k=0..5 -> (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)");
  }
  return std::make_pair(kAlphaBetaTable[k][0], kAlphaBetaTable[k][1]);
}

void jacobi_sss_calc_lmp(SUS2RadialMTPBasis &basis, LAMMPS *lmp, double dist, double scal,
                         double s, int k)
{
  const std::pair<int, int> alpha_beta = jacobi_alpha_beta_for_block(k, lmp);
  const double alpha = static_cast<double>(alpha_beta.first);
  const double beta = static_cast<double>(alpha_beta.second);

  const double z = 0.5 * scal * (dist - s);
  double x = tanh(z);
  x = std::max(-1.0 + kJacobiEndpointEps, std::min(1.0 - kJacobiEndpointEps, x));

  const double sech_sq = 1.0 - x * x;
  const double x_r = 0.5 * scal * sech_sq;

  const double one_minus_x = std::max(kJacobiEndpointEps, 1.0 - x);
  const double one_plus_x = std::max(kJacobiEndpointEps, 1.0 + x);

  double sqrt_weight = 1.0;
  if (alpha != 0.0) sqrt_weight *= pow(one_minus_x, 0.5 * alpha);
  if (beta != 0.0) sqrt_weight *= pow(one_plus_x, 0.5 * beta);

  const double log_weight_x = -0.5 * alpha / one_minus_x + 0.5 * beta / one_plus_x;

  double y_prev = 0.0;
  double y_prev_x = 0.0;

  double y_curr = sqrt_weight;
  double y_curr_x = sqrt_weight * log_weight_x;

  const double dr = dist - basis.max_cutoff;
  const double cutoff = dr * dr;
  const double cutoff_der = 2.0 * dr;

  auto store_basis = [&](int index, double y, double y_x) {
    basis.radial_basis_vals[index] = basis.scaling * cutoff * y;
    basis.radial_basis_ders[index] = basis.scaling * (cutoff_der * y + cutoff * y_x * x_r);
  };

  store_basis(0, y_curr, y_curr_x);
  if (basis.size == 1) return;

  const double linear = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);
  const double linear_x = 0.5 * (alpha + beta + 2.0);

  double y_next = linear * y_curr;
  double y_next_x = linear_x * y_curr + linear * y_curr_x;

  store_basis(1, y_next, y_next_x);

  y_prev = y_curr;
  y_prev_x = y_curr_x;
  y_curr = y_next;
  y_curr_x = y_next_x;

  for (int order = 2; order < basis.size; ++order) {
    const double n = static_cast<double>(order);
    const double denom = 2.0 * n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
    const double b = 2.0 * n + alpha + beta - 1.0;
    const double c = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
    const double d = alpha * alpha - beta * beta;
    const double e = 2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + alpha + beta);
    const double coeff = b * (c * x + d) / denom;
    const double coeff_x = b * c / denom;
    const double prev_coeff = e / denom;

    y_next = coeff * y_curr - prev_coeff * y_prev;
    y_next_x = coeff_x * y_curr + coeff * y_curr_x - prev_coeff * y_prev_x;

    store_basis(order, y_next, y_next_x);

    y_prev = y_curr;
    y_prev_x = y_curr_x;
    y_curr = y_next;
    y_curr_x = y_next_x;
  }
}

}    // namespace

// RBChebyshev_ss implementation - matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_ss::RB_Calc
void RBChebyshev_ss::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double ksi = x / sq;  // Basis function ksi = x/√(x²+1)
  
  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base functions (i=0)
  radial_basis_vals[0] = scaling * cutoff;
  
  // First Chebyshev polynomial (i=1)
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  // Chebyshev recursion: psi_n = 2*ksi*psi_{n-1} - psi_{n-2}
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2];
  }
}

void RBChebyshev_ss::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);
  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double ksi = x / sq;  // Basis function ksi = x/√(x²+1)
  double der = 1.0 / (denom * sq);  // dksi/dx
  double dder = -3.0 * ksi * der / sq;  // d²ksi/dx²
  
  // Multipliers for different derivatives (matching SUS2-MLIP-1.1)
  double mult = der * scal / 2.0;  // dksi/dr
  double mult_s_r = -dder * scal * scal / 4.0;  // d²ksi/dscal*dr
  double mult_scal_r = der / 2.0 + dder * x / 2.0;  // d²ksi/dr*dscal
  double mult_scal = der * (dist - s) / 2.0;  // dksi/dscal
  double mult_s = -mult;  // dksi/ds
  
  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base derivatives (i=0)
  radial_basis_ders[0] = 2.0 * Dr * scaling;
  radial_basis_ders[0 + size] = 0.0;
  radial_basis_ders[0 + 2*size] = 0.0;
  radial_basis_ders[0 + 3*size] = 0.0;
  radial_basis_ders[0 + 4*size] = 0.0;
  
  // First Chebyshev polynomial derivatives (i=1)
  radial_basis_ders[1] = scaling * (mult * cutoff + 2.0 * ksi * Dr);
  radial_basis_ders[1 + size] = scaling * mult_scal * cutoff;
  radial_basis_ders[1 + 2*size] = scaling * (mult_scal_r * cutoff + 2.0 * mult_scal * Dr);
  radial_basis_ders[1 + 3*size] = scaling * mult_s * cutoff;
  radial_basis_ders[1 + 4*size] = scaling * (mult_s_r * cutoff + 2.0 * mult_s * Dr);
  
  // Chebyshev recursion for all 5 derivatives (matching SUS2-MLIP-1.1)
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - radial_basis_ders[i-2];
    radial_basis_ders[i + size] = 2.0 * (mult_scal * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1+size]) - radial_basis_ders[i-2+size];
    radial_basis_ders[i + 2*size] = 2.0 * (mult_scal_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i-1+size] + 
                                        ksi * radial_basis_ders[i-1+2*size] + mult_scal * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i-2+2*size];
    radial_basis_ders[i + 3*size] = 2.0 * (mult_s * radial_basis_vals[i-1] + ksi * radial_basis_ders[i+3*size-1]) - 
                                    radial_basis_ders[i+3*size-2];
    radial_basis_ders[i + 4*size] = 2.0 * (mult_s_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i+3*size-1] + 
                                        ksi * radial_basis_ders[i+4*size-1] + mult_s * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i+4*size-2];
  }
}



// RBChebyshev_sss_lmp implementation - matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_sss::RB_Calc
void RBChebyshev_sss_lmp::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);  // Basis function ksi = tanh(x) (SUS2-MLIP-1.1 standard)

  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;

  // Base functions (i=0)
  radial_basis_vals[0] = scaling * cutoff;

  // First Chebyshev polynomial (i=1)
  radial_basis_vals[1] = scaling * (ksi * cutoff);

  // Chebyshev recursion: psi_n = 2*ksi*psi_{n-1} - psi_{n-2}
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2];
  }
}

void RBChebyshev_sss_lmp::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);  // Basis function ksi = tanh(x)
  double der = 1.0 - ksi * ksi;  // dksi/dx = 1 - tanh²(x) = sech²(x)
  double dder = -2.0 * ksi * der;  // d²ksi/dx² = -2*tanh(x)*sech²(x)

  // Multipliers for different derivatives (matching SUS2-MLIP-1.1)
  double mult = der * scal / 2.0;  // dksi/dr
  //double mult_s_r = -dder * scal * scal / 4.0;  // d²ksi/dscal*dr
  //double mult_scal_r = der / 2.0 + dder * (dist - s) * scal / 4.0;  // d²ksi/dr*dscal
  //double mult_scal = der * (dist - s) / 2.0;  // dksi/dscal
  //double mult_s = -mult;  // dksi/ds

  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;

  // Base derivatives (i=0)
  radial_basis_ders[0] = 2.0 * Dr * scaling;
  //radial_basis_ders[0 + size] = 0.0;
  //radial_basis_ders[0 + 2*size] = 0.0;
  //radial_basis_ders[0 + 3*size] = 0.0;
  //radial_basis_ders[0 + 4*size] = 0.0;

  // First Chebyshev polynomial derivatives (i=1)
  radial_basis_ders[1] = scaling * (mult * cutoff + 2.0 * ksi * Dr);
  //radial_basis_ders[1 + size] = scaling * mult_scal * cutoff;
  //radial_basis_ders[1 + 2*size] = scaling * (mult_scal_r * cutoff + 2.0 * mult_scal * Dr);
  //radial_basis_ders[1 + 3*size] = scaling * mult_s * cutoff;
  //radial_basis_ders[1 + 4*size] = scaling * (mult_s_r * cutoff + 2.0 * mult_s * Dr);

  // Chebyshev recursion for all 5 derivatives (matching SUS2-MLIP-1.1)
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - radial_basis_ders[i-2];
    //radial_basis_ders[i + size] = 2.0 * (mult_scal * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1+size]) - radial_basis_ders[i-2+size];
    //radial_basis_ders[i + 2*size] = 2.0 * (mult_scal_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i-1+size] +
     //                                   ksi * radial_basis_ders[i-1+2*size] + mult_scal * radial_basis_ders[i-1]) -
     //                                   radial_basis_ders[i-2+2*size];
    //radial_basis_ders[i + 3*size] = 2.0 * (mult_s * radial_basis_vals[i-1] + ksi * radial_basis_ders[i+3*size-1]) -
    //                                radial_basis_ders[i+3*size-2];
    //radial_basis_ders[i + 4*size] = 2.0 * (mult_s_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i+3*size-1] +
    //                                    ksi * radial_basis_ders[i+4*size-1] + mult_s * radial_basis_ders[i-1]) -
     //                                   radial_basis_ders[i+4*size-2];
  }
}







// RBChebyshev_sss implementation - matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_sss::RB_Calc
void RBChebyshev_sss::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);  // Basis function ksi = tanh(x) (SUS2-MLIP-1.1 standard)
  
  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base functions (i=0)
  radial_basis_vals[0] = scaling * cutoff;
  
  // First Chebyshev polynomial (i=1)
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  // Chebyshev recursion: psi_n = 2*ksi*psi_{n-1} - psi_{n-2}
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2];
  }
}

void RBChebyshev_sss::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);  // Basis function ksi = tanh(x)
  double der = 1.0 - ksi * ksi;  // dksi/dx = 1 - tanh²(x) = sech²(x)
  double dder = -2.0 * ksi * der;  // d²ksi/dx² = -2*tanh(x)*sech²(x)
  
  // Multipliers for different derivatives (matching SUS2-MLIP-1.1)
  double mult = der * scal / 2.0;  // dksi/dr
  double mult_s_r = -dder * scal * scal / 4.0;  // d²ksi/dscal*dr
  double mult_scal_r = der / 2.0 + dder * (dist - s) * scal / 4.0;  // d²ksi/dr*dscal
  double mult_scal = der * (dist - s) / 2.0;  // dksi/dscal
  double mult_s = -mult;  // dksi/ds
  
  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base derivatives (i=0)
  radial_basis_ders[0] = 2.0 * Dr * scaling;
  radial_basis_ders[0 + size] = 0.0;
  radial_basis_ders[0 + 2*size] = 0.0;
  radial_basis_ders[0 + 3*size] = 0.0;
  radial_basis_ders[0 + 4*size] = 0.0;
  
  // First Chebyshev polynomial derivatives (i=1)
  radial_basis_ders[1] = scaling * (mult * cutoff + 2.0 * ksi * Dr);
  radial_basis_ders[1 + size] = scaling * mult_scal * cutoff;
  radial_basis_ders[1 + 2*size] = scaling * (mult_scal_r * cutoff + 2.0 * mult_scal * Dr);
  radial_basis_ders[1 + 3*size] = scaling * mult_s * cutoff;
  radial_basis_ders[1 + 4*size] = scaling * (mult_s_r * cutoff + 2.0 * mult_s * Dr);
  
  // Chebyshev recursion for all 5 derivatives (matching SUS2-MLIP-1.1)
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - radial_basis_ders[i-2];
    radial_basis_ders[i + size] = 2.0 * (mult_scal * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1+size]) - radial_basis_ders[i-2+size];
    radial_basis_ders[i + 2*size] = 2.0 * (mult_scal_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i-1+size] + 
                                        ksi * radial_basis_ders[i-1+2*size] + mult_scal * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i-2+2*size];
    radial_basis_ders[i + 3*size] = 2.0 * (mult_s * radial_basis_vals[i-1] + ksi * radial_basis_ders[i+3*size-1]) - 
                                    radial_basis_ders[i+3*size-2];
    radial_basis_ders[i + 4*size] = 2.0 * (mult_s_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i+3*size-1] + 
                                        ksi * radial_basis_ders[i+4*size-1] + mult_s * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i+4*size-2];
  }
}

void RBLaguerre_log1p::calc_radial_basis(double dist, double scal, double s, int k)
{
  const double rho = (s > kLaguerreMinRho) ? s : kLaguerreMinRho;
  const double u = scal * log1p(dist / rho);
  const double Dr = dist - max_cutoff;
  const double cutoff = Dr * Dr;
  const double exp_factor = exp(-0.5 * u);

  double phi_prev = 0.0;
  double phi_curr = scaling * cutoff * exp_factor;
  radial_basis_vals[0] = phi_curr;

  for (int n = 0; n < size - 1; ++n) {
    const double inv_np1 = 1.0 / (n + 1.0);
    const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
    const double prev_coeff = n * inv_np1;
    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;

    radial_basis_vals[n + 1] = phi_next;
    phi_prev = phi_curr;
    phi_curr = phi_next;
  }
}

void RBLaguerre_log1p::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  const bool rho_is_active = s > kLaguerreMinRho;
  const double rho = rho_is_active ? s : kLaguerreMinRho;
  const double log_term = log1p(dist / rho);
  const double u = scal * log_term;
  const double u_r = scal / (rho + dist);
  const double u_scal = log_term;
  const double u_scal_r = 1.0 / (rho + dist);
  const double u_rho = rho_is_active ? (-scal * dist / (rho * (rho + dist))) : 0.0;
  const double u_rho_r = rho_is_active ? (-scal / ((rho + dist) * (rho + dist))) : 0.0;

  const double Dr = dist - max_cutoff;
  const double cutoff = Dr * Dr;
  const double cutoff_der = 2.0 * Dr;

  const double exp_factor = exp(-0.5 * u);
  double phi_prev = 0.0;
  double dphi_prev = 0.0;
  double dphi_scal_prev = 0.0;
  double dphi_scal_r_prev = 0.0;
  double dphi_rho_prev = 0.0;
  double dphi_rho_r_prev = 0.0;

  double phi_curr = scaling * cutoff * exp_factor;
  double dphi_curr = scaling * cutoff_der * exp_factor - 0.5 * u_r * phi_curr;
  double dphi_scal_curr = -0.5 * u_scal * phi_curr;
  double dphi_scal_r_curr = -0.5 * u_scal_r * phi_curr - 0.5 * u_scal * dphi_curr;
  double dphi_rho_curr = -0.5 * u_rho * phi_curr;
  double dphi_rho_r_curr = -0.5 * u_rho_r * phi_curr - 0.5 * u_rho * dphi_curr;

  radial_basis_vals[0] = phi_curr;
  radial_basis_ders[0] = dphi_curr;
  radial_basis_ders[size] = dphi_scal_curr;
  radial_basis_ders[2 * size] = dphi_scal_r_curr;
  radial_basis_ders[3 * size] = dphi_rho_curr;
  radial_basis_ders[4 * size] = dphi_rho_r_curr;

  for (int n = 0; n < size - 1; ++n) {
    const double inv_np1 = 1.0 / (n + 1.0);
    const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
    const double prev_coeff = n * inv_np1;

    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;
    const double dphi_next = -u_r * inv_np1 * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;
    const double dphi_scal_next =
        -u_scal * inv_np1 * phi_curr + coeff * dphi_scal_curr - prev_coeff * dphi_scal_prev;
    const double dphi_scal_r_next =
        (-u_scal_r * phi_curr - u_scal * dphi_curr - u_r * dphi_scal_curr) * inv_np1 +
        coeff * dphi_scal_r_curr - prev_coeff * dphi_scal_r_prev;
    const double dphi_rho_next =
        -u_rho * inv_np1 * phi_curr + coeff * dphi_rho_curr - prev_coeff * dphi_rho_prev;
    const double dphi_rho_r_next =
        (-u_rho_r * phi_curr - u_rho * dphi_curr - u_r * dphi_rho_curr) * inv_np1 +
        coeff * dphi_rho_r_curr - prev_coeff * dphi_rho_r_prev;

    radial_basis_vals[n + 1] = phi_next;
    radial_basis_ders[n + 1] = dphi_next;
    radial_basis_ders[n + 1 + size] = dphi_scal_next;
    radial_basis_ders[n + 1 + 2 * size] = dphi_scal_r_next;
    radial_basis_ders[n + 1 + 3 * size] = dphi_rho_next;
    radial_basis_ders[n + 1 + 4 * size] = dphi_rho_r_next;

    phi_prev = phi_curr;
    dphi_prev = dphi_curr;
    dphi_scal_prev = dphi_scal_curr;
    dphi_scal_r_prev = dphi_scal_r_curr;
    dphi_rho_prev = dphi_rho_curr;
    dphi_rho_r_prev = dphi_rho_r_curr;
    phi_curr = phi_next;
    dphi_curr = dphi_next;
    dphi_scal_curr = dphi_scal_next;
    dphi_scal_r_curr = dphi_scal_r_next;
    dphi_rho_curr = dphi_rho_next;
    dphi_rho_r_curr = dphi_rho_r_next;
  }
}

void RBLaguerre_log1p_lmp::calc_radial_basis(double dist, double scal, double s, int k)
{
  const double rho = (s > kLaguerreMinRho) ? s : kLaguerreMinRho;
  const double u = scal * log1p(dist / rho);
  const double Dr = dist - max_cutoff;
  const double cutoff = Dr * Dr;
  const double exp_factor = exp(-0.5 * u);

  double phi_prev = 0.0;
  double phi_curr = scaling * cutoff * exp_factor;
  radial_basis_vals[0] = phi_curr;

  for (int n = 0; n < size - 1; ++n) {
    const double inv_np1 = 1.0 / (n + 1.0);
    const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
    const double prev_coeff = n * inv_np1;
    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;

    radial_basis_vals[n + 1] = phi_next;
    phi_prev = phi_curr;
    phi_curr = phi_next;
  }
}

void RBLaguerre_log1p_lmp::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  const double rho = (s > kLaguerreMinRho) ? s : kLaguerreMinRho;
  const double u = scal * log1p(dist / rho);
  const double u_r = scal / (rho + dist);
  const double Dr = dist - max_cutoff;
  const double cutoff = Dr * Dr;
  const double cutoff_der = 2.0 * Dr;
  const double exp_factor = exp(-0.5 * u);

  double phi_prev = 0.0;
  double dphi_prev = 0.0;
  double phi_curr = scaling * cutoff * exp_factor;
  double dphi_curr = scaling * cutoff_der * exp_factor - 0.5 * u_r * phi_curr;

  radial_basis_vals[0] = phi_curr;
  radial_basis_ders[0] = dphi_curr;

  for (int n = 0; n < size - 1; ++n) {
    const double inv_np1 = 1.0 / (n + 1.0);
    const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
    const double prev_coeff = n * inv_np1;
    const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;
    const double dphi_next = -u_r * inv_np1 * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;

    radial_basis_vals[n + 1] = phi_next;
    radial_basis_ders[n + 1] = dphi_next;
    phi_prev = phi_curr;
    dphi_prev = dphi_curr;
    phi_curr = phi_next;
    dphi_curr = dphi_next;
  }
}

void RBLaguerre_log1p_pos::calc_radial_basis(double dist, double scal, double s, int k)
{
  laguerre_log1p_positive_calc(*this, dist, scal, s, false);
}

void RBLaguerre_log1p_pos::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  laguerre_log1p_positive_calc(*this, dist, scal, s, true);
}

void RBLaguerre_log1p_pos_lmp::calc_radial_basis(double dist, double scal, double s, int k)
{
  laguerre_log1p_positive_calc(*this, dist, scal, s, false);
}

void RBLaguerre_log1p_pos_lmp::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  laguerre_log1p_positive_calc(*this, dist, scal, s, true);
}

void RBJacobi_sss_lmp::calc_radial_basis(double dist, double scal, double s, int k)
{
  jacobi_sss_calc_lmp(*this, lmp, dist, scal, s, k);
}

void RBJacobi_sss_lmp::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  jacobi_sss_calc_lmp(*this, lmp, dist, scal, s, k);
}

// RBChebyshev_tanhexp implementation - matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_tanhexp::RB_Calc
void RBChebyshev_tanhexp::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double expm = exp(-x);
  double expp = exp(x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * (exp((x_plus) * expm - 1.0)) + 1.0;  // Special transformation (SUS2-MLIP-1.1)
  
  // Apply cutoff function: (r - max_dist)² (matching SUS2-MLIP-1.1)
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base functions (i=0)
  radial_basis_vals[0] = scaling * cutoff;
  
  // First Chebyshev polynomial (i=1)
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  // Chebyshev recursion: psi_n = 2*ksi*psi_{n-1} - psi_{n-2}
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2];
  }
}

void RBChebyshev_tanhexp::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double expm = exp(-x);
  double expp = exp(x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * (exp((x_plus) * expm - 1.0)) + 1.0;  // Special transformation
  double der = 2.0 * exp((-1.0 + expm) * x_plus) * x;  // dksi/dx (SUS2-MLIP-1.1)
  double dder = -2.0 * exp(-1.0 - 2.0 * x + expm * x_plus) * (expp * (-1.0 + x) + x * x);  // d²ksi/dx²
  
  // Multipliers for different derivatives (matching SUS2-MLIP-1.1)
  double mult = der * scal * 0.5;  // dksi/dr
  double mult_s_r = -dder * scal * scal / 4;  // d²ksi/dscal*dr (with 0.05² = 0.0025)
  double mult_scal_r = der / 2.0 + dder * scal * (dist - s) / 4.0;  // d²ksi/dr*dscal
  double mult_scal = der * (dist - s) / 2.0;  // dksi/dscal
  double mult_s = -mult;  // dksi/ds
  
  // Apply cutoff function: (r - max_dist)²
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Base derivatives (i=0)
  radial_basis_ders[0] = 2.0 * Dr * scaling;
  radial_basis_ders[0 + size] = 0.0;
  radial_basis_ders[0 + 2*size] = 0.0;
  radial_basis_ders[0 + 3*size] = 0.0;
  radial_basis_ders[0 + 4*size] = 0.0;
  
  // First Chebyshev polynomial derivatives (i=1)
  radial_basis_ders[1] = scaling * (mult * cutoff + 2.0 * ksi * Dr);
  radial_basis_ders[1 + size] = scaling * mult_scal * cutoff;
  radial_basis_ders[1 + 2*size] = scaling * (mult_scal_r * cutoff + 2.0 * mult_scal * Dr);
  radial_basis_ders[1 + 3*size] = scaling * mult_s * cutoff;
  radial_basis_ders[1 + 4*size] = scaling * (mult_s_r * cutoff + 2.0 * mult_s * Dr);
  
  // Chebyshev recursion for all 5 derivatives (matching SUS2-MLIP-1.1)
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - radial_basis_ders[i-2];
    radial_basis_ders[i + size] = 2.0 * (mult_scal * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1+size]) - radial_basis_ders[i-2+size];
    radial_basis_ders[i + 2*size] = 2.0 * (mult_scal_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i-1+size] + 
                                        ksi * radial_basis_ders[i-1+2*size] + mult_scal * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i-2+2*size];
    radial_basis_ders[i + 3*size] = 2.0 * (mult_s * radial_basis_vals[i-1] + ksi * radial_basis_ders[i+3*size-1]) - 
                                    radial_basis_ders[i+3*size-2];
    radial_basis_ders[i + 4*size] = 2.0 * (mult_s_r * radial_basis_vals[i-1] + mult * radial_basis_ders[i+3*size-1] + 
                                        ksi * radial_basis_ders[i+4*size-1] + mult_s * radial_basis_ders[i-1]) - 
                                        radial_basis_ders[i+4*size-2];
  }
}

// RBBessel_sss implementation - matching SUS2-MLIP-1.1 RadialBasis_Bessel_sss::RB_Calc
void RBBessel_sss::calc_radial_basis(double dist, double scal, double s, int k)
{
  double pi = 3.141592654;
  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);
  double temp = ksi + 1.00001;  // Shifted tanh for Bessel calculation
  
  // Apply cutoff function with polynomial enhancement (matching SUS2-MLIP-1.1)
  double Dr = dist / max_cutoff - 1.0;
  double p = 6.0;
  double cutoff = 1.0 - (p+1.0)*(p+2.0)*pow(Dr,p)*0.5 + p*(p+2.0)*pow(Dr,p+1.0) - (p+1.0)*p*pow(Dr,p+2.0)*0.5;
  
  // All basis functions use Bessel form (matching SUS2-MLIP-1.1)
  for (int i = 0; i < size; i++) {
    double N = i + 1.0;
    double w = pi * N / 2.0;
    double sin_val = sin(w * temp);
    double bessel = sin_val / temp;
    radial_basis_vals[i] = scaling * bessel * cutoff;
  }
}

void RBBessel_sss::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) / 2 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) / 2.0;
  double ksi = tanh(x);
  double der = 1.0 - ksi * ksi;  // dksi/dx = 1 - tanh²(x)
  double dder = -2.0 * ksi * der;  // d²ksi/dx² = -2*tanh(x)*sech²(x)
  
  // Multipliers for different derivatives (matching SUS2-MLIP-1.1)
  double mult = der * scal / 2.0;  // dksi/dr
  double mult_s_r = -dder * scal * scal / 4.0;  // d²ksi/dscal*dr
  double mult_scal_r = der / 2.0 + dder * (dist - s) * scal / 4.0;  // d²ksi/dr*dscal
  double mult_scal = der * (dist - s) / 2.0;  // dksi/dscal
  double mult_s = -mult;  // dksi/ds
  
  // Cutoff function derivatives (matching SUS2-MLIP-1.1)
  double Dr = dist / max_cutoff - 1.0;
  double p = 6.0;
  double cutoff = 1.0 - (p+1.0)*(p+2.0)*pow(Dr,p)*0.5 + p*(p+2.0)*pow(Dr,p+1.0) - (p+1.0)*p*pow(Dr,p+2.0)*0.5;
  double cutoff_der = p*(p+1.0)*(p+2.0)*(pow(Dr,p) - 0.5*(pow(Dr,p+1.0) + pow(Dr,p-1.0))) / max_cutoff;
  
  double pi = 3.141592654;
  double temp = ksi + 1.00001;
  
  // All basis functions use Bessel derivative with all 5 derivatives (matching SUS2-MLIP-1.1)
  for (int i = 0; i < size; i++) {
    double N = i + 1.0;
    double w = pi * N / 2.0;
    double sin_val = sin(w * temp);
    double cos_val = cos(w * temp);
    double bessel = sin_val / temp;
    double bessel_der = w * cos_val / temp - sin_val / (temp * temp);
    double bessel_ddr = -(w * w * sin_val / temp) - 2.0 * w * cos_val / (temp * temp) + 2.0 * sin_val / (temp * temp * temp);
    
    // Calculate all 5 derivatives (matching SUS2-MLIP-1.1)
    radial_basis_ders[i] = scaling * (bessel_der * mult * cutoff + bessel * cutoff_der);
    radial_basis_ders[i + size] = scaling * bessel_der * mult_scal * cutoff;
    radial_basis_ders[i + 2*size] = scaling * (bessel_ddr * mult * mult_scal * cutoff + bessel_der * mult_scal_r * cutoff + cutoff_der * bessel_der * mult_scal);
    radial_basis_ders[i + 3*size] = scaling * bessel_der * mult_s * cutoff;
    radial_basis_ders[i + 4*size] = scaling * (bessel_ddr * mult * mult_s * cutoff + bessel_der * mult_s_r * cutoff + cutoff_der * bessel_der * mult_s);
  }
}

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
// Extended radial basis functions implementation from SUS2-MLIP-1.1
// Contributing author: Transplanted from SUS2-MLIP-1.1
//

#include "sus2_mtp_rb_extended_basis.h"
#include "error.h"
#include "memory.h"
#include "utils.h"
#include <cmath>

using namespace LAMMPS_NS;

void RBChebyshev_ssss::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  
  // First transformation: exponential (matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_s)
  double expm = exp(-x);
  double expp = exp(x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * (exp((x_plus) * expm - 1.0)) + 1.0;
  
  // Second transformation: hyperbolic (matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_ss)
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double _ksi = x / sq;
  
  // Apply cutoff function: (r - max_dist)² (matching SUS2-MLIP-1.1)
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  
  // Dual scale system: split basis functions via shift (matching SUS2-MLIP-1.1)
  int shift = size / 2;
  
  // Initialize first half (using ksi transformation)
  radial_basis_vals[0] = scaling * cutoff;
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  // Initialize second half (using _ksi transformation)
  radial_basis_vals[shift] = scaling * cutoff;
  radial_basis_vals[shift + 1] = scaling * (_ksi * cutoff);
  
  // Chebyshev recursion: first half uses ksi (matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
  
  // Chebyshev recursion: second half uses _ksi (matching SUS2-MLIP-1.1)
  for (int i = shift + 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * _ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_ssss::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s, k);
  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  
  // First transformation: exponential (matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_s)
  double expm = exp(-x);
  double expp = exp(x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * (exp((x_plus) * expm - 1.0)) + 1.0;
  double der = 2.0 * exp((-1.0 + expm) * (x_plus)) * x;
  double dder = -2.0 * exp(-1.0 - 2.0 * x + expm * (x_plus)) * (expp * (-1.0 + x) + x * x);
  double mult = der * scal / 2.0;
  double mult_s_r = -dder * scal * scal / 4.0;
  double mult_scal_r = der / 2.0 + dder * x / 2.0;
  double mult_scal = der * (dist - s) / 2.0;
  double mult_s = -mult;
  
  // Second transformation: hyperbolic (matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_ss)
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double _ksi = x / sq;
  double _der = 1.0 / (denom * sq);
  double _dder = -3.0 * _ksi * _der / sq;
  double _mult = _der * scal / 2.0;
  double _mult_s_r = -_dder * scal * scal / 4.0;
  double _mult_scal_r = _der / 2.0 + _dder * x / 2.0;
  double _mult_scal = _der * (dist - s) / 2.0;
  double _mult_s = -_mult;
  
  // Apply cutoff function: (r - max_dist)² (matching SUS2-MLIP-1.1)
  double Dr = dist - max_cutoff;
  double cutoff = Dr * Dr;
  double cutoff_der = 2.0 * Dr;
  
  // Dual scale system: split basis functions via shift (matching SUS2-MLIP-1.1)
  int shift = size / 2;
  
  // Initialize derivatives: first half (using ksi transformation)
  radial_basis_ders[0] = scaling * cutoff_der;
  radial_basis_ders[1] = scaling * (mult * cutoff + 2.0 * ksi * Dr);
  
  // Initialize derivatives: second half (using _ksi transformation)
  radial_basis_ders[shift] = scaling * cutoff_der;
  radial_basis_ders[shift + 1] = scaling * (_mult * cutoff + 2.0 * _ksi * Dr);
  
  // Chebyshev recursion: first half uses ksi (matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                                         radial_basis_ders[i-2]);
  }
  
  // Chebyshev recursion: second half uses _ksi (matching SUS2-MLIP-1.1)
  for (int i = shift + 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (_mult * radial_basis_vals[i-1] + _ksi * radial_basis_ders[i-1]) - 
                                         radial_basis_ders[i-2]);
  }
}

void RBChebyshev_sssss::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5 (fully matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double ksi = x / sq;  // First: hyperbolic
  double _ksi = 2.0 * ksi * ksi - 1.0;  // Second: second-order Chebyshev
  double __ksi = ksi * ksi * ksi;  // Third: cubic (note: not Chebyshev)
  
  // Triple scale system: split basis functions via shift (fully matching SUS2-MLIP-1.1)
  int shift = size / 3;      // First part start index
  int shift_ = 2 * shift;    // Second part start index
  
  // Apply cutoff function: (r - max_dist)² (fully matching SUS2-MLIP-1.1)
  double Dr = dist - max_cutoff;
  double cutoff_f = Dr * Dr;
  
  // Initialize three parts (fully matching SUS2-MLIP-1.1)
  radial_basis_vals[0] = scaling * cutoff_f;
  radial_basis_vals[shift] = scaling * cutoff_f;
  radial_basis_vals[shift_] = scaling * cutoff_f;
  
  radial_basis_vals[1] = scaling * (ksi * cutoff_f);
  radial_basis_vals[1 + shift] = scaling * (_ksi * cutoff_f);
  radial_basis_vals[1 + shift_] = scaling * (__ksi * cutoff_f);
  
  // Chebyshev recursion: first part uses ksi (fully matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2]);
  }
  
  // Chebyshev recursion: second part uses _ksi (fully matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    radial_basis_vals[i + shift] = scaling * (2.0 * _ksi * radial_basis_vals[shift + i - 1] - radial_basis_vals[shift + i - 2]);
  }
  
  // Chebyshev recursion: third part uses __ksi (fully matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    radial_basis_vals[i + shift_] = scaling * (2.0 * __ksi * radial_basis_vals[shift_ + i - 1] - radial_basis_vals[shift_ + i - 2]);
  }
}

void RBChebyshev_sssss::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) * 0.5 (fully matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double denom = x * x + 1.0;
  double sq = sqrt(denom);
  double ksi = x / sq;  // First: hyperbolic
  
  // Calculate ksi derivative (matching SUS2-MLIP-1.1)
  double der = 1.0 / (denom * sq);
  double dder = -3.0 * ksi * der / sq;
  
  // Calculate _ksi derivative (second-order Chebyshev derivative, matching SUS2-MLIP-1.1)
  double _ksi = 2.0 * ksi * ksi - 1.0;
  double _der = 4.0 * ksi * der;
  double _dder = 4.0 * (der * der + ksi * dder);
  
  // Calculate __ksi derivative (cubic derivative, note: non-standard, matching SUS2-MLIP-1.1)
  double __ksi = ksi * ksi * ksi;
  double __der = 0.75 * ksi * _der;  // Warning: non-standard, should be 3*ksi*ksi*der
  double __dder = 6.0 * ksi * der * der + 3.0 * ksi * ksi * dder;
  
  // Calculate mult coefficients (matching SUS2-MLIP-1.1)
  double mult = der * scal / 2.0;
  double mult_s_r = -dder * scal * scal / 4.0;
  double mult_scal_r = der / 2.0 + dder * x / 2.0;
  double mult_scal = der * (dist - s) / 2.0;
  double mult_s = -mult;
  
  double _mult = _der * scal / 2.0;
  double _mult_s_r = -_dder * scal * scal / 4.0;
  double _mult_scal_r = _der / 2.0 + _dder * x / 2.0;
  double _mult_scal = _der * (dist - s) / 2.0;
  double _mult_s = -_mult;
  
  double __mult = __der * scal / 2.0;
  double __mult_s_r = -__dder * scal * scal / 4.0;
  double __mult_scal_r = __der / 2.0 + __dder * x / 2.0;
  double __mult_scal = __der * (dist - s) / 2.0;
  double __mult_s = -__mult;
  
  // Apply cutoff function: (r - max_dist)² (matching SUS2-MLIP-1.1)
  double Dr = dist - max_cutoff;
  double cutoff_f = Dr * Dr;
  
  // Triple scale system: split basis functions via shift (fully matching SUS2-MLIP-1.1)
  int shift = size / 3;      // First part start index
  int shift_ = 2 * shift;    // Second part start index
  
  // Initialize derivatives: three parts (fully matching SUS2-MLIP-1.1)
  radial_basis_ders[0] = scaling * 2.0 * Dr;
  radial_basis_ders[shift] = scaling * 2.0 * Dr;
  radial_basis_ders[shift_] = scaling * 2.0 * Dr;
  
  radial_basis_ders[1] = scaling * (mult * cutoff_f + 2.0 * ksi * Dr);
  radial_basis_ders[1 + shift] = scaling * (_mult * cutoff_f + 2.0 * _ksi * Dr);
  radial_basis_ders[1 + shift_] = scaling * (__mult * cutoff_f + 2.0 * __ksi * Dr);
  
  // Chebyshev recursion: three parts independently calculate derivatives (fully matching SUS2-MLIP-1.1)
  for (int i = 2; i < shift; i++) {
    // First part (using ksi)
    radial_basis_ders[i] = scaling * (2.0 * (mult * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1]) - 
                                             radial_basis_ders[i - 2]);
    radial_basis_ders[i + size] = scaling * (2.0 * (mult_scal * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1 + size]) - 
                                                  radial_basis_ders[i - 2 + size]);
    radial_basis_ders[i + 2 * size] = scaling * (2.0 * (mult_scal_r * radial_basis_vals[i - 1] + mult * radial_basis_ders[i - 1 + size] + 
                                                               ksi * radial_basis_ders[i - 1 + 2 * size] + mult_scal * radial_basis_ders[i - 1]) - 
                                                            radial_basis_ders[i - 2 + 2 * size]);
    radial_basis_ders[i + 3 * size] = scaling * (2.0 * (mult_s * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i + 3 * size - 1]) - 
                                                  radial_basis_ders[i + 3 * size - 2]);
    radial_basis_ders[i + 4 * size] = scaling * (2.0 * (mult_s_r * radial_basis_vals[i - 1] + mult * radial_basis_ders[i + 3 * size - 1] + 
                                                               ksi * radial_basis_ders[i + 4 * size - 1] + mult_s * radial_basis_ders[i - 1]) - 
                                                            radial_basis_ders[i + 4 * size - 2]);
    
    // Second part (using _ksi)
    radial_basis_ders[i + shift] = scaling * (2.0 * (_mult * radial_basis_vals[i - 1 + shift] + _ksi * radial_basis_ders[i - 1 + shift]) - 
                                                     radial_basis_ders[i - 2 + shift]);
    radial_basis_ders[i + size + shift] = scaling * (2.0 * (_mult_scal * radial_basis_vals[i - 1 + shift] + _ksi * radial_basis_ders[i - 1 + size + shift]) - 
                                                          radial_basis_ders[i - 2 + size + shift]);
    radial_basis_ders[i + 2 * size + shift] = scaling * (2.0 * (_mult_scal_r * radial_basis_vals[i - 1 + shift] + _mult * radial_basis_ders[i - 1 + size + shift] + 
                                                                     _ksi * radial_basis_ders[i - 1 + 2 * size + shift] + _mult_scal * radial_basis_ders[i - 1 + shift]) - 
                                                                  radial_basis_ders[i - 2 + 2 * size + shift]);
    radial_basis_ders[i + 3 * size + shift] = scaling * (2.0 * (_mult_s * radial_basis_vals[i - 1 + shift] + _ksi * radial_basis_ders[i + 3 * size - 1 + shift]) - 
                                                          radial_basis_ders[i + 3 * size - 2 + shift]);
    radial_basis_ders[i + 4 * size + shift] = scaling * (2.0 * (_mult_s_r * radial_basis_vals[i - 1 + shift] + _mult * radial_basis_ders[i + 3 * size - 1 + shift] + 
                                                                     _ksi * radial_basis_ders[i + 4 * size - 1 + shift] + _mult_s * radial_basis_ders[i - 1 + shift]) - 
                                                                  radial_basis_ders[i + 4 * size - 2 + shift]);
    
    // Third part (using __ksi)
    radial_basis_ders[i + shift_] = scaling * (2.0 * (__mult * radial_basis_vals[i - 1 + shift_] + __ksi * radial_basis_ders[i - 1 + shift_]) - 
                                                      radial_basis_ders[i - 2 + shift_]);
    radial_basis_ders[i + size + shift_] = scaling * (2.0 * (__mult_scal * radial_basis_vals[i - 1 + shift_] + __ksi * radial_basis_ders[i - 1 + size + shift_]) - 
                                                           radial_basis_ders[i - 2 + size + shift_]);
    radial_basis_ders[i + 2 * size + shift_] = scaling * (2.0 * (__mult_scal_r * radial_basis_vals[i - 1 + shift_] + __mult * radial_basis_ders[i - 1 + size + shift_] + 
                                                                      __ksi * radial_basis_ders[i - 1 + 2 * size + shift_] + __mult_scal * radial_basis_ders[i - 1 + shift_]) - 
                                                                   radial_basis_ders[i - 2 + 2 * size + shift_]);
    radial_basis_ders[i + 3 * size + shift_] = scaling * (2.0 * (__mult_s * radial_basis_vals[i - 1 + shift_] + __ksi * radial_basis_ders[i + 3 * size - 1 + shift_]) - 
                                                           radial_basis_ders[i + 3 * size - 2 + shift_]);
    radial_basis_ders[i + 4 * size + shift_] = scaling * (2.0 * (__mult_s_r * radial_basis_vals[i - 1 + shift_] + __mult * radial_basis_ders[i + 3 * size - 1 + shift_] + 
                                                                      __ksi * radial_basis_ders[i + 4 * size - 1 + shift_] + __mult_s * radial_basis_ders[i - 1 + shift_]) - 
                                                                   radial_basis_ders[i + 4 * size - 2 + shift_]);
  }
}

void RBChebyshev_ssw::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double denom = x*x + exp(0);  // Weighting factor
  double sq = sqrt(denom);
  double ksi = x/sq;
  
  // Apply cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  
  radial_basis_vals[0] = scaling * cutoff;
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_ssw::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);

  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double denom = x*x + exp(0);
  double sq = sqrt(denom);
  double ksi = x/sq;
  double der = exp(0)/(denom*sq);
  
  double mult = der * dx_dr;
  
  // Derivative of cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double cutoff_der = -2.0 * (max_cutoff - dist);
  
  radial_basis_ders[0] = scaling * cutoff_der;
  radial_basis_ders[1] = scaling * (mult * cutoff + ksi * cutoff_der);
  
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                          radial_basis_ders[i-2]);
  }
}

void RBChebyshev_sssw::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double tanh_x = tanh(x);
  
  // Weighting processing - use k parameter to select weights
  const double arr[] = {1.0, 5.0/7.0, 9.0/7.0, 3.0/7.0, 11.0/7.0, 13.0/7.0};
  int idx = k % 6;  // Array has 6 elements
  double w = arr[idx];
  double ksi;
  if (w == 1.0) {
    ksi = tanh_x;
  } else {
    ksi = pow(tanh_x, w);
  }
  
  // Apply cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  
  radial_basis_vals[0] = scaling * cutoff;
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_sssw::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double tanh_x = tanh(x);
  double sech_x = 1.0 - tanh_x * tanh_x;
  
  // Weighting processing - use k parameter to select weights
  const double arr[] = {1.0, 5.0/7.0, 9.0/7.0, 3.0/7.0, 11.0/7.0, 13.0/7.0};
  int idx = k % 6;  // Array has 6 elements
  double w = arr[idx];
  double ksi;
  double der;
  if (w == 1.0) {
    ksi = tanh_x;
    der = sech_x;
  } else {
    ksi = pow(tanh_x, w);
    der = w * pow(tanh_x, w-1) * sech_x;
  }
  
  double mult = der * dx_dr;
  
  // Derivative of cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double cutoff_der = -2.0 * (max_cutoff - dist);
  
  radial_basis_ders[0] = scaling * cutoff_der;
  radial_basis_ders[1] = scaling * (mult * cutoff + ksi * cutoff_der);
  
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                          radial_basis_ders[i-2]);
  }
}

void RBChebyshev_tanhexp_w::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double exp_val = exp(-0.5559 - x);
  
  // Weighting processing - use passed k parameter to select sigma
  int sigma = k % 3;
  double u;
  if (sigma == 0) {
    u = exp_val + 0.02;
  } else if (sigma == 1) {
    u = exp_val + 0.15;
  } else {
    u = exp_val + 0.5;
  }
  
  double tanh_u = tanh(u);
  double temp = 2.0 * tanh_u - 1.0;
  double ksi = 2.0 * temp * temp - 1.0;  // Double transformation
  
  // Apply cutoff function with weighting
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double w = 1.0 - ksi;  // Weighting factor
  
  radial_basis_vals[0] = scaling * cutoff * w;
  radial_basis_vals[1] = scaling * (ksi * cutoff * w);
  
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_tanhexp_w::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double exp_val = exp(-0.5559 - x);
  
  // Weighting processing - use passed k parameter to select sigma
  int sigma = k % 3;
  double u;
  if (sigma == 0) {
    u = exp_val + 0.02;
  } else if (sigma == 1) {
    u = exp_val + 0.15;
  } else {
    u = exp_val + 0.5;
  }
  
  double tanh_u = tanh(u);
  double temp = 2.0 * tanh_u - 1.0;
  double dydu = 1.0 - tanh_u * tanh_u;
  double du_dr = -exp_val * dx_dr;
  double der = 4.0 * (2.0 * tanh_u - 1.0) * dydu * du_dr;
  
  double ksi = 2.0 * temp * temp - 1.0;
  double w = 1.0 - ksi;
  
  // Derivative of cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double cutoff_der = -2.0 * (max_cutoff - dist);
  
  radial_basis_ders[0] = scaling * (cutoff_der * w - der * cutoff);
  radial_basis_ders[1] = scaling * (der * cutoff * w + ksi * (cutoff_der * w - der * cutoff));
  
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (der * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                          radial_basis_ders[i-2]);
  }
}

void RBChebyshev_sigma::calc_radial_basis(double dist, double scal, double s, int k)
{

  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double ksi;
  
  // Multi-sigma processing - use passed k parameter to select transformation mode
  int sigma = k % 3;
  if (sigma == 0) {
    // First sigma: hyperbolic sine
    double denom = x*x + 1.0;
    double sq = sqrt(denom);
    ksi = x/sq;
  } else if (sigma == 1) {
    // Second sigma: exponential
    double expm = exp(-x);
    double expp = exp(x);
    double x_plus = 1.0 + x;
    ksi = -2.0 * (exp((x_plus)*expm-1.0)) + 1.0;
  } else {
    // Third sigma: hyperbolic tangent exponential
    double exp_val = exp(-0.5559 - x);
    double u = exp_val + 0.02;
    double tanh_u = tanh(u);
    double temp = 2.0 * tanh_u - 1.0;
    ksi = 2.0 * temp * temp - 1.0;
  }
  
  // Apply cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  
  radial_basis_vals[0] = scaling * cutoff;
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_sigma::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);

  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double ksi;
  double der;
  double dder;
  
  // Multi-sigma processing - use passed k parameter to select transformation mode
  int sigma = k % 3;
  if (sigma == 0) {
    // First sigma: hyperbolic sine
    double denom = x*x + 1.0;
    double sq = sqrt(denom);
    ksi = x/sq;
    der = 1.0/(denom*sq);
    dder = -3.0*ksi*der/sq;
  } else if (sigma == 1) {
    // Second sigma: exponential
    double expm = exp(-x);
    double expp = exp(x);
    double x_plus = 1.0 + x;
    ksi = -2.0 * (exp((x_plus)*expm-1.0)) + 1.0;
    der = 2.0 * exp((-1.0+expm)*(x_plus)) * x;
    dder = -2.0 * exp(-1.0-2.0*x+expm*(x_plus)) * (expp*(-1.0+x)+x*x);
  } else {
    // Third sigma: hyperbolic tangent exponential
    double exp_val = exp(-0.5559 - x);
    double u = exp_val + 0.02;
    double tanh_u = tanh(u);
    double temp = 2.0 * tanh_u - 1.0;
    ksi = 2.0 * temp * temp - 1.0;
    double dydu = 1.0 - tanh_u * tanh_u;
    double du_dr = -exp_val * dx_dr;
    der = 4.0 * (2.0 * tanh_u - 1.0) * dydu * du_dr;
    dder = 8.0 * (exp_val*exp_val*(2.0*dydu*dydu+temp*dydu) + temp*dydu*exp_val) * dx_dr * dx_dr;
  }
  
  double mult = der;
  double mult_s_r = dder;
  
  // Derivative of cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double cutoff_der = -2.0 * (max_cutoff - dist);
  
  radial_basis_ders[0] = scaling * cutoff_der;
  radial_basis_ders[1] = scaling * (mult * cutoff + ksi * cutoff_der);
  
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (mult * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                          radial_basis_ders[i-2]);
  }
}

void RBBessel_sssw::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double tanh_x = tanh(x);
  double temp = tanh_x + 1.00001;
  
  // Weighting processing - use k parameter to calculate weights
  double w = (1.0 + k) * 0.1;
  double x_weighted = w * x;
  double tanh_weighted = tanh(x_weighted);
  double temp_weighted = tanh_weighted + 1.00001;
  
  // Apply cutoff function with polynomial enhancement
  double Dr = dist / max_cutoff - 1.0;
  double p = 6.0;
  double cutoff = 1.0 - (p+1.0)*(p+2.0)*pow(Dr,p)*0.5 + p*(p+2.0)*pow(Dr,p+1.0) - (p+1.0)*p*pow(Dr,p+2.0)*0.5;
  
  radial_basis_vals[0] = scaling * cutoff;
  
  for (int i = 1; i < size; i++) {
    double N = i + 1.0;
    double Pi = 3.141592654;
    double w_bessel = Pi * N / 2.0;
    double sin_val = sin(w_bessel * temp_weighted);
    double bessel = sin_val / temp_weighted;
    radial_basis_vals[i] = scaling * bessel * cutoff;
  }
}

void RBBessel_sssw::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double tanh_x = tanh(x);
  double sech_x = 1.0 - tanh_x * tanh_x;
  
  // Weighting processing - use k parameter to calculate weights
  double w = (1.0 + k) * 0.1;
  double x_weighted = w * x;
  double tanh_weighted = tanh(x_weighted);
  double sech_weighted = 1.0 - tanh_weighted * tanh_weighted;
  double temp_weighted = tanh_weighted + 1.00001;
  double mult_weighted = w * sech_weighted * dx_dr;
  
  // Cutoff function derivatives
  double Dr = dist / max_cutoff - 1.0;
  double p = 6.0;
  double cutoff = 1.0 - (p+1.0)*(p+2.0)*pow(Dr,p)*0.5 + p*(p+2.0)*pow(Dr,p+1.0) - (p+1.0)*p*pow(Dr,p+2.0)*0.5;
  double cutoff_der = p*(p+1.0)*(p+2.0)*(pow(Dr,p) - 0.5*(pow(Dr,p+1.0) + pow(Dr,p-1.0))) / max_cutoff;
  
  radial_basis_ders[0] = scaling * cutoff_der;
  
  for (int i = 1; i < size; i++) {
    double N = i + 1.0;
    double Pi = 3.141592654;
    double w_bessel = Pi * N / 2.0;
    double sin_val = sin(w_bessel * temp_weighted);
    double cos_val = cos(w_bessel * temp_weighted);
    double bessel = sin_val / temp_weighted;
    double bessel_der = (w_bessel * cos_val / temp_weighted) - (sin_val / (temp_weighted * temp_weighted));
    double mult_combined = mult_weighted * bessel_der * cutoff + bessel * cutoff_der;
    
    radial_basis_ders[i] = scaling * mult_combined;
  }
}

void RBChebyshev_Tri::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5
  double x = scal * (dist - s) * 0.5;
  double Pi = 3.141592654;
  double tri_x = x * 0.05 * Pi;
  
  double cos_val = cos(tri_x);
  double sin_val = sin(tri_x);
  double ksi = sin_val;  // Using sine function as basis
  
  // Apply cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  
  radial_basis_vals[0] = scaling * cutoff;
  radial_basis_vals[1] = scaling * (ksi * cutoff);
  
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = scaling * (2.0 * ksi * radial_basis_vals[i-1] - radial_basis_vals[i-2]);
  }
}

void RBChebyshev_Tri::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  calc_radial_basis(dist, scal, s);

  // Nonlinear transformation: x = scal * (r - s) * 0.5
  // Derivative dx/dr = scal * 0.5
  double x = scal * (dist - s) * 0.5;
  double dx_dr = scal * 0.5;
  double Pi = 3.141592654;
  double tri_x = x * 0.05 * Pi;
  
  double cos_val = cos(tri_x);
  double sin_val = sin(tri_x);
  double ksi = sin_val;
  double der = cos_val * 0.05 * Pi * dx_dr;
  
  // Derivative of cutoff function
  double cutoff = (max_cutoff - dist) * (max_cutoff - dist);
  double cutoff_der = -2.0 * (max_cutoff - dist);
  
  radial_basis_ders[0] = scaling * cutoff_der;
  radial_basis_ders[1] = scaling * (der * cutoff + ksi * cutoff_der);
  
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = scaling * (2.0 * (der * radial_basis_vals[i-1] + ksi * radial_basis_ders[i-1]) - 
                          radial_basis_ders[i-2]);
  }
}

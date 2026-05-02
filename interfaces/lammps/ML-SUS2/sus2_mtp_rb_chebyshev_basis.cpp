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
// Contributing author, Richard Meng, Queen's University at Kingston, 22.11.24, contact@richardzjm.com
//

#include "sus2_mtp_radial_basis.h"
#include "sus2_mtp_rb_chebyshev_basis.h"

#include "error.h"
#include "memory.h"
#include "utils.h"

#include "cstring"

using namespace LAMMPS_NS;

void RBChebyshev::calc_radial_basis(double dist, double scal, double s, int k)
{
  double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

  radial_basis_vals[0] = scaling * (1 * (dist - max_cutoff) * (dist - max_cutoff));
  radial_basis_vals[1] = scaling * (ksi * (dist - max_cutoff) * (dist - max_cutoff));
  for (int i = 2; i < size; i++) {
    radial_basis_vals[i] = 2 * ksi * radial_basis_vals[i - 1] - radial_basis_vals[i - 2];
  }
}

void RBChebyshev::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  RBChebyshev::calc_radial_basis(dist, scal, s, k);

  double mult = 2.0 / (max_cutoff - min_cutoff);
  double ksi = (2 * dist - (min_cutoff + max_cutoff)) / (max_cutoff - min_cutoff);

  radial_basis_ders[0] = scaling * 2 * (dist - max_cutoff);
  radial_basis_ders[1] =
      scaling * (mult * (dist - max_cutoff) * (dist - max_cutoff) + 2 * ksi * (dist - max_cutoff));
  for (int i = 2; i < size; i++) {
    radial_basis_ders[i] = 2 * (mult * radial_basis_vals[i - 1] + ksi * radial_basis_ders[i - 1]) -
        radial_basis_ders[i - 2];
  }
}

// RBChebyshev_s implementation - matching SUS2-MLIP-1.1 RadialBasis_Chebyshev_s
void RBChebyshev_s::calc_radial_basis(double dist, double scal, double s, int k)
{
  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double expm = exp(-x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * exp((x_plus) * expm - 1.0) + 1.0;  // Special transformation (SUS2-MLIP-1.1)
  
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

void RBChebyshev_s::calc_radial_basis_ders(double dist, double scal, double s, int k)
{
  // First compute basis values (needed for derivative calculations)
  calc_radial_basis(dist, scal, s, k);

  // Nonlinear transformation: x = scal * (r - s) * 0.5 (matching SUS2-MLIP-1.1)
  double x = scal * (dist - s) * 0.5;
  double expm = exp(-x);
  double expp = exp(x);
  double x_plus = 1.0 + x;
  double ksi = -2.0 * exp((x_plus) * expm - 1.0) + 1.0;  // Special transformation
  double der = 2.0 * exp((-1.0 + expm) * x_plus) * x;  // dksi/dx (SUS2-MLIP-1.1)
  double dder = -2.0 * exp(-1.0 - 2.0 * x + expm * x_plus) * (expp * (-1.0 + x) + x * x);  // d²ksi/dx²
  
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

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

#ifndef LMP_SUS2_MTP_RADIAL_BASIS_H
#define LMP_SUS2_MTP_RADIAL_BASIS_H

#include "potential_file_reader.h"

namespace LAMMPS_NS {

class SUS2RadialMTPBasis {
 public:
  SUS2RadialMTPBasis(TextFileReader &tfr, LAMMPS *lmp);
  SUS2RadialMTPBasis(int size, LAMMPS *lmp);
  ~SUS2RadialMTPBasis();    // Needed to clear memory

  //Specifically reads the basis properties (ie. cutoffs and size) and not the radial parameters
  void ReadBasisProperties(TextFileReader &tfr);

  virtual void calc_radial_basis(double dist, double scal = 0.1, double s = 0.1, int k = 0) = 0;
  virtual void calc_radial_basis_ders(double dist, double scal = 0.1, double s = 0.1, int k = 0) = 0;

  int allocated = 0;

  int size;             // size of radial basis functions
  double min_cutoff;    //  Minimum radius value
  double max_cutoff;    // Cutoff radius
  double scaling;        // all functions are multiplied by scaling
  
  // SUS2-MLIP specific parameters
  enum BasisType {
    CHEBYSHEV = 1,
    CHEBYSHEV_SS = 2,
    CHEBYSHEV_SSS = 3,
    CHEBYSHEV_SSS_LMP = 15,
    LAGUERRE_LOG1P = 16,
    LAGUERRE_LOG1P_LMP = 17,
    JACOBI_SSS_LMP = 18,
    LAGUERRE_LOG1P_POS = 19,
    LAGUERRE_LOG1P_POS_LMP = 20,
    CHEBYSHEV_TANHEXP = 4,
    CHEBYSHEV_S = 14,
    BESSEL_SSS = 5,
    CHEBYSHEV_SSSS = 6,
    CHEBYSHEV_SSSSS = 7,
    CHEBYSHEV_SSW = 8,
    CHEBYSHEV_SSSW = 9,
    CHEBYSHEV_TANHEXP_W = 10,
    CHEBYSHEV_SIGMA = 11,
    BESSEL_SSSW = 12,
    CHEBYSHEV_TRI = 13
  };
  BasisType basis_type;   // type of radial basis function
  
  // Values and derivatives for radial basis functions
  double *radial_basis_vals;
  double *radial_basis_ders;

 protected:
  LAMMPS *lmp;    // LAMMPS reference
};

}    // namespace LAMMPS_NS

#endif

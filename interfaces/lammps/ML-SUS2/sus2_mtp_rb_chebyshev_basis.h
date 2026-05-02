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

#ifndef LMP_SUS2_MTP_RB_CHEBYSHEV_BASIS_H
#define LMP_SUS2_MTP_RB_CHEBYSHEV_BASIS_H

#include "sus2_mtp_radial_basis.h"
#include "text_file_reader.h"

namespace LAMMPS_NS {
class RBChebyshev : public SUS2RadialMTPBasis {
 public:
  RBChebyshev(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV;
  };
  RBChebyshev(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBChebyshev_s : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_s(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_S;
  };
  RBChebyshev_s(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_S;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};
}    // namespace LAMMPS_NS

#endif

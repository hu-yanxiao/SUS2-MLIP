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

#ifndef LMP_SUS2_MTP_RB_CHEBYSHEV_SS_BASIS_H
#define LMP_SUS2_MTP_RB_CHEBYSHEV_SS_BASIS_H

#include "sus2_mtp_radial_basis.h"
#include "text_file_reader.h"

namespace LAMMPS_NS {

class RBChebyshev_ss : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_ss(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SS;
  };
  RBChebyshev_ss(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBChebyshev_sss : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_sss(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSS;
  };
  RBChebyshev_sss(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBChebyshev_sss_lmp : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_sss_lmp(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSS_LMP;
  };
  RBChebyshev_sss_lmp(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSS_LMP;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBLaguerre_log1p : public SUS2RadialMTPBasis {
 public:
  RBLaguerre_log1p(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = LAGUERRE_LOG1P;
  };
  RBLaguerre_log1p(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = LAGUERRE_LOG1P;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBLaguerre_log1p_lmp : public SUS2RadialMTPBasis {
 public:
  RBLaguerre_log1p_lmp(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = LAGUERRE_LOG1P_LMP;
  };
  RBLaguerre_log1p_lmp(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = LAGUERRE_LOG1P_LMP;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBLaguerre_log1p_pos : public SUS2RadialMTPBasis {
 public:
  RBLaguerre_log1p_pos(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = LAGUERRE_LOG1P_POS;
  };
  RBLaguerre_log1p_pos(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = LAGUERRE_LOG1P_POS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBLaguerre_log1p_pos_lmp : public SUS2RadialMTPBasis {
 public:
  RBLaguerre_log1p_pos_lmp(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = LAGUERRE_LOG1P_POS_LMP;
  };
  RBLaguerre_log1p_pos_lmp(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = LAGUERRE_LOG1P_POS_LMP;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBJacobi_sss_lmp : public SUS2RadialMTPBasis {
 public:
  RBJacobi_sss_lmp(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = JACOBI_SSS_LMP;
  };
  RBJacobi_sss_lmp(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = JACOBI_SSS_LMP;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1,
                                 int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1,
                                      int k = 0) override;
};


class RBChebyshev_tanhexp : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_tanhexp(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_TANHEXP;
  };
  RBChebyshev_tanhexp(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_TANHEXP;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

class RBBessel_sss : public SUS2RadialMTPBasis {
 public:
  RBBessel_sss(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = BESSEL_SSS;
  };
  RBBessel_sss(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = BESSEL_SSS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

}    // namespace LAMMPS_NS

#endif

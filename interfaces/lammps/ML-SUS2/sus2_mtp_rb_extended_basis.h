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
// Extended radial basis functions from SUS2-MLIP-1.1
// Contributing author: Transplanted from SUS2-MLIP-1.1
//

#ifndef LMP_SUS2_MTP_RB_EXTENDED_BASIS_H
#define LMP_SUS2_MTP_RB_EXTENDED_BASIS_H

#include "sus2_mtp_radial_basis.h"
#include "text_file_reader.h"

namespace LAMMPS_NS {

// RBChebyshev_ssss - 四阶变换的多尺度切比雪夫基函数
class RBChebyshev_ssss : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_ssss(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSSS;
  };
  RBChebyshev_ssss(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSSS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_sssss - 五阶变换的多尺度切比雪夫基函数
class RBChebyshev_sssss : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_sssss(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSSSS;
  };
  RBChebyshev_sssss(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSSSS;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_ssw - 加权双尺度切比雪夫基函数
class RBChebyshev_ssw : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_ssw(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSW;
  };
  RBChebyshev_ssw(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSW;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_sssw - 加权三尺度切比雪夫基函数
class RBChebyshev_sssw : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_sssw(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SSSW;
  };
  RBChebyshev_sssw(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SSSW;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_tanhexp_w - 加权双曲正切指数基函数
class RBChebyshev_tanhexp_w : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_tanhexp_w(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_TANHEXP_W;
  };
  RBChebyshev_tanhexp_w(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_TANHEXP_W;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_sigma - 多σ基函数
class RBChebyshev_sigma : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_sigma(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_SIGMA;
  };
  RBChebyshev_sigma(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_SIGMA;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBBessel_sssw - 加权三阶贝塞尔基函数
class RBBessel_sssw : public SUS2RadialMTPBasis {
 public:
  RBBessel_sssw(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = BESSEL_SSSW;
  };
  RBBessel_sssw(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = BESSEL_SSSW;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

// RBChebyshev_Tri - 三角函数基
class RBChebyshev_Tri : public SUS2RadialMTPBasis {
 public:
  RBChebyshev_Tri(int size, LAMMPS *lmp) : SUS2RadialMTPBasis(size, lmp) {
    basis_type = CHEBYSHEV_TRI;
  };
  RBChebyshev_Tri(TextFileReader &tfr, LAMMPS *lmp) : SUS2RadialMTPBasis(tfr, lmp) {
    basis_type = CHEBYSHEV_TRI;
  }
  virtual void calc_radial_basis(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
  virtual void calc_radial_basis_ders(double val, double scal = 0.1, double s = 0.1, int k = 0) override;
};

}    // namespace LAMMPS_NS

#endif

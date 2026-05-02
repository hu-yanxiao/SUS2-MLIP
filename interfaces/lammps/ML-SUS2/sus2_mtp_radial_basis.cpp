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

#include "error.h"
#include "memory.h"
#include "potential_file_reader.h"
#include "text_file_reader.h"
#include "utils.h"

#include "csignal"
#include "cstring"

using namespace LAMMPS_NS;

SUS2RadialMTPBasis::SUS2RadialMTPBasis(TextFileReader &tfr, LAMMPS *lmp)
{
  this->lmp = lmp;

  //Clear out old arrays if any
  if (allocated) {
    lmp->memory->destroy(radial_basis_vals);
    lmp->memory->destroy(radial_basis_ders);
  }

  ReadBasisProperties(tfr);
}

SUS2RadialMTPBasis::SUS2RadialMTPBasis(int size, LAMMPS *lmp) : size(size), lmp(lmp)
{
  //Clear out old arrays if any
  if (allocated) {
    lmp->memory->destroy(radial_basis_vals);
    lmp->memory->destroy(radial_basis_ders);
  }
  
  //Allocate the memory for the basis set values and derivatives.
  // Note: radial_basis_ders stores 5 sets of derivatives (dr, dscal, dr*dscal, ds, ds*dr)
  lmp->memory->create(radial_basis_vals, size, "pair:sus2_mtp_radial_vals");
  lmp->memory->create(radial_basis_ders, 5*size, "pair:sus2_mtp_radial_ders");

  // Initialize allocated arrays to avoid uninitialised reads (Valgrind/ASan)
  for (int ii = 0; ii < size; ++ii) radial_basis_vals[ii] = 0.0;
  for (int ii = 0; ii < 5 * size; ++ii) radial_basis_ders[ii] = 0.0;

  // Initialize SUS2-MLIP specific parameters
  scaling = 1.0;
  basis_type = CHEBYSHEV;  // Default to standard Chebyshev
  
  allocated = 1;
}

void SUS2RadialMTPBasis::ReadBasisProperties(TextFileReader &tfr)
{

  std::string new_separators = "=, ";
  std::string separators = TOKENIZER_DEFAULT_SEPARATORS + new_separators;

  //Extact next line and it's tokens
  ValueTokenizer line_tokens{std::string(tfr.next_line()), separators};
  std::string keyword = line_tokens.next_string();

  // First check if scaling is available
  if (keyword == "scaling") {
    //If available alert the user and extract the next lines
    scaling = line_tokens.next_double();
    utils::logmesg(lmp, "MTP Radial Scaling Value = {} ", scaling);
    line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
    keyword = line_tokens.next_string();
  }

  // Read the lower cutoff
  if (keyword != "min_val" && keyword != "min_dist")
    lmp->error->all(FLERR, "Error in reading MTP file. Cannot read lower cutoff.");
  min_cutoff = line_tokens.next_double();

  // Read the upper cutoff
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "max_val" && keyword != "max_dist")
    lmp->error->all(FLERR, "Error in reading MTP file. Cannot read upper cutoff.");
  max_cutoff = line_tokens.next_double();

  // Read the basis size set value
  line_tokens = ValueTokenizer(std::string(tfr.next_line()), separators);
  keyword = line_tokens.next_string();
  if (keyword != "radial_basis_size")
    lmp->error->all(FLERR, "Error in reading MTP file. Cannot read radial basis set size.");
  size = line_tokens.next_int();    // Assuming size is an int

  //Allocate the memory for the basis set values and derivatives.
  // Note: radial_basis_ders stores 5 sets of derivatives (dr, dscal, dr*dscal, ds, ds*dr)
  lmp->memory->create(radial_basis_vals, size, "pair:sus2_mtp_radial_vals");
  lmp->memory->create(radial_basis_ders, 5*size, "pair:sus2_mtp_radial_ders");

  // Initialize allocated arrays to avoid uninitialised reads (Valgrind/ASan)
  for (int ii = 0; ii < size; ++ii) radial_basis_vals[ii] = 0.0;
  for (int ii = 0; ii < 5 * size; ++ii) radial_basis_ders[ii] = 0.0;

  allocated = 1;
}
SUS2RadialMTPBasis::~SUS2RadialMTPBasis()
{
  lmp->memory->destroy(radial_basis_vals);
  lmp->memory->destroy(radial_basis_ders);
}

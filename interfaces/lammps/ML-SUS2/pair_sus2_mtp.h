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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sus2mtp,PairSUS2MTP);
// clang-format on
#else

#ifndef LMP_PAIR_SUS2_MTP_H
#define LMP_PAIR_SUS2_MTP_H

#include "pair.h"
#include "sus2_mtp_radial_basis.h"
#include "sus2_mtp_zbl.h"
#include <atomic>    // 原子操作支持
#include <mutex>     // 互斥锁支持
#include <vector>

namespace LAMMPS_NS {

class PairSUS2MTP : public Pair {
 public:
  PairSUS2MTP(class LAMMPS *);
  ~PairSUS2MTP() override;
  void compute(int, int) override;         //Workhorse comuptation
  void settings(int, char **) override;    // Reads args from "pair_style"
  void coeff(int, char **) override;       // Reads args from "pair_coeff" (only * * for mtp)
  void init_style() override;              //Init style
  double init_one(int, int) override;      // Checks that species are inited
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;



 protected:

       // radial list
  double ***radial_list = nullptr;    // First created during read_file
	  double ***radial_der_list = nullptr;    // First created during read_file
	  double ***two_layer_gate_radial_list = nullptr;
	  double ***two_layer_gate_radial_der_list = nullptr;
	  double ***env_gate_radial_list = nullptr;
	  double ***env_gate_radial_der_list = nullptr;
	  double **env_gate_rho_list = nullptr;
	  double **env_gate_rho_der_list = nullptr;
  size_t total_list_elements=0;
  bool do_list=false;
  double inv_dr=0 ;
  double requested_tabstep = 1.0e-4;
  double actual_dr = 0.0;
  int list_grid_size = 0;
  bool tabstep_set_by_user = false;
  int *pair_to_table_index = nullptr;
  int used_species_count = 0;
  int used_pair_count = 0;

  void read_file(FILE *, const char *);       //Parsing file using LAMMPS utils
  std::string potential_name = "Untitled";    //An optional name which isn't currently used.
  std::string potential_tag = "";    //An optional tag/description which isn't currently used.
  bool is_sh_model = false;           // SUS2-SH uses real spherical harmonics for angular channels
  int sh_l_max = 0;
  int sh_k_max = 0;
  int sh_body_order = 0;

  int species_count = 0;     // Number of species (initialize to 0 to prevent uninitialized memory usage)
  double scaling =1;    // All forces are multiplied by scaling (global scaling)   // Initialize to 0
  int scal_coeffs_count = 0; // Initialize to 0
  int radial_coeff_count;      // Total radial basis coefficients (radial_func_count * radial_basis_size)


  // SUS2-MLIP physical scaling parameters
  int L_max = 0;         // Maximum moment tensor level
  int K_scaling = 1;     // Scaling dimension parameter
  std::string scaling_map = "LK";  // Scaling mode: "L", "K", or "LK"
  int *mu_to_K;          // Mapping from radial function to scaling dimension (fixed: int*)
  int *mu_to_sigma;      // Mapping from radial function to sigma index (fixed: int*)
  
  // SUS2-MLIP regression coefficients (managed by PairSUS2MTP, not by radial basis)
  // Unified array: [shift_coeffs][scal_coeffs][radial_basis_coeffs][optional env-gate coeffs]
  double *regression_coeffs;  // Unified regression coefficients array
  int regression_coeffs_count;    // Total size
  
  // For convenience, maintain pointers to sections (offsets into regression_coeffs)
  int shift_coeffs_offset;      // Offset to shift_coeffs section (= 0)
  int scal_coeffs_offset;      // Offset to scal_coeffs section (= species_count)
  int radial_coeffs_offset;     // Offset to radial_basis_coeffs section
  int env_gate_coeffs_offset = 0;
  int env_gate_lambda_raw_offset = 0;
  int env_gate_log_density_coeffs_offset = 0;
  bool env_gate_enabled = false;
  double env_gate_cutoff_ratio = 0.5;
  double env_gate_activation_on_ratio = 0.5;
  int env_gate_channel_count = 6;
  bool zbl_enabled = false;
  double zbl_inner = 0.7;
  double zbl_outer = 1.4;
  double zbl_outer_sq = 1.96;
  bool zbl_typewise_cutoff_enabled = false;
  double zbl_typewise_cutoff_factor = 0.7;
  double interaction_cutoff = 0.0;
  double interaction_cutoff_sq = 0.0;
  int *zbl_atomic_numbers = nullptr;
  double *zbl_pair_inner_cutoffs = nullptr;
  double *zbl_pair_outer_cutoffs = nullptr;
  double *zbl_pair_outer_sq = nullptr;
  
  double *shift_coeffs;  // Shift coefficients for each species (species_count elements)
  double *scal_coeffs;   // Scaling coefficients for coordinate transformation (scal_coeffs_count elements)

  // Radial basis
  // Basis types correspond to SUS2RadialMTPBasis::BasisType enum
  std::string radial_basis_type_str;   // String name of radial basis type for reading from file
  int radial_basis_type_index;         // Index for MPI Bcast
  SUS2RadialMTPBasis *radial_basis; // Pointer to basis object
  double *radial_basis_coeffs;        // These are the radial basis coeffs (c)
  int radial_func_count;              // Number of radial bases (mu_max)
  int radial_basis_size;              // Number of elements in bases
  
  // Cutoff values are now managed by radial_basis object
  // Access via radial_basis->min_cutoff and radial_basis->max_cutoff
  double min_cutoff;
  double max_cutoff;
  double max_cutoff_sq;    // Maximum radial cutoff squared (The MTP only supports one cutoff for all species combinations)

  double *linear_coeffs;     // These are the moment tensor basis coeffs (xi)
  double *species_coeffs;    // For the species coefficients (0th rank moment tensor)
  bool two_layer_gate_enabled = false;
  bool two_layer_gate_shared_radial = false;
  bool two_layer_residual_enabled = false;
  bool two_layer_gate_direct_scale = false;
  double two_layer_gate_bias = 1.0;
  int two_layer_gate_body_order_max = 0;
  int two_layer_gate_weight_count = 0;
  std::vector<int> sh_scalar_body_order;
  std::vector<int> two_layer_gate_scalar_indices;
  std::vector<double> two_layer_gate_weights;
  std::vector<double> two_layer_gate_radial_coeffs;
  std::vector<double> two_layer_gate_additive_coeffs;
  std::vector<double> two_layer_gate_additive_ratios;
  std::vector<unsigned char> two_layer_gate_additive_ratio_valid;
  std::vector<size_t> two_layer_gate_edge_offsets;
  std::vector<int> two_layer_gate_edge_neighbors;
  std::vector<int> two_layer_gate_edge_types;
  std::vector<double> two_layer_gate_edge_dx;
  std::vector<double> two_layer_gate_edge_dy;
  std::vector<double> two_layer_gate_edge_dz;
  std::vector<double> two_layer_gate_edge_dist;
  std::vector<double> two_layer_gate_edge_deriv_x;
  std::vector<double> two_layer_gate_edge_deriv_y;
  std::vector<double> two_layer_gate_edge_deriv_z;
  int alpha_moment_count, alpha_index_basic_count, alpha_index_times_count, alpha_scalar_count,
      max_alpha_index_basic;    // Counts of various alpha indicies
  int **alpha_index_basic;      // Indicies how to construct elementary moments from coords and dist
  int **alpha_index_times;      // Indicies to combine existing moments into knew ones
  int *alpha_moment_mapping;    // Selects the basis values from completed moments
  int *alpha_basic_mu;          // Cached radial function index for each basic alpha
  int *alpha_basic_a0;          // Cached x exponent for each basic alpha
  int *alpha_basic_a1;          // Cached y exponent for each basic alpha
  int *alpha_basic_a2;          // Cached z exponent for each basic alpha
  int *alpha_basic_norm_rank;   // Cached a0+a1+a2 for each basic alpha
  int *alpha_basic_sh_index;    // Cached real-SH flat index for SUS2-SH basic alpha
  int *alpha_times_a0;          // Cached lhs moment index for each alpha-times entry
  int *alpha_times_a1;          // Cached rhs moment index for each alpha-times entry
  int *alpha_times_multiplier;  // Cached multiplier for each alpha-times entry
  double *alpha_times_coeff;    // Cached tensor-product coefficient for each alpha-times entry
  int *alpha_times_out;         // Cached output moment index for each alpha-times entry

  // Other working buffers
  int jac_size = 0;         // Size of the jacobian (jnum dim)




  // Concurrency optimization: simple flag for buffer initialization
  volatile int buffers_initialized;  // 使用volatile确保内存可见性
  
  double *inv_dist_powers;  // Buffer used for inverse powers of dist (eg. d^-i)
  double *coord_powers_x;   // Buffer used for powers of dx
  double *coord_powers_y;   // Buffer used for powers of dy
  double *coord_powers_z;   // Buffer used for powers of dz
  double *radial_vals;      // Buffer used for radial basis function values for each mu
  double *radial_ders;      // Buffer used for radial basis function derivatives for each mu
  double *two_layer_raw_basic_vals = nullptr;
  double *two_layer_gate_residual_radial_vals = nullptr;
  double *two_layer_gate_values = nullptr;
  double *two_layer_gate_adjoints = nullptr;
  double *two_layer_radial_cache_vals = nullptr;
  double *two_layer_radial_cache_ders = nullptr;
  int two_layer_raw_jac_size = 0;
  int two_layer_atom_buffer_size = 0;
  int two_layer_radial_cache_size = 0;
  double *moment_jacobian_x = nullptr;    // SoA layout for x-component
  double *moment_jacobian_y = nullptr;    // SoA layout for y-component
  double *moment_jacobian_z = nullptr;    // SoA layout for z-component
  double *moment_tensor_vals;             //Buffer to hold the moments
  double *nbh_energy_ders_wrt_moments;    // Same as above except for ders
  double *weighted_basic_moment_ders;     // Basic-moment derivatives with species coeff prefactor applied
  double *env_rho_dr = nullptr;           // Reused env-gate density derivative per neighbor
  double *env_activation_basic_vals = nullptr;  // Reused env-gate chain accumulator per basic moment
  double *zbl_force_prefactors = nullptr; // Reused 0.5*dE/dr/r for full-neighbor ZBL
  int env_activation_basic_size = 0;

  // Cache whether to calculate forces based on cutoff as calculated in alpha basics
  bool *within_cutoff = nullptr;    // First created during compute using grow

  bool has_nonzero_two_layer_gate_weights() const;
  bool requires_two_layer_gate_sh() const;
  void prepare_two_layer_gate_additive_ratios();
  void compute_two_layer_gate_sh(int, int);
  int two_layer_gate_additive_coeff_index(int, int) const;
  double two_layer_gate_additive_coeff(int, int) const;
  void calc_pair_radial_values(int, int, double, bool, double = 0.0, bool = false);
  void accumulate_sh_basic_edge(int, const double *, double, double, bool, int, bool = false);
  void forward_sh_products();
  void backprop_sh_products();
  void ensure_two_layer_atom_buffers();
  void ensure_two_layer_edge_buffer(int);
};

}    // namespace LAMMPS_NS

#endif
#endif

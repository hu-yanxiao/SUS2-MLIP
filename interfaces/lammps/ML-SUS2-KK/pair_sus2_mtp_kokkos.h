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
// Contributing author, Richard Meng, Queen's University at Kingston, 21.01.24, contact@richardzjm.com
//

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sus2mtp/kk,PairSUS2MTPKokkos<LMPDeviceType>);
PairStyle(sus2mtp/kk/device,PairSUS2MTPKokkos<LMPDeviceType>);
PairStyle(sus2mtp/kk/host,PairSUS2MTPKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PAIR_SUS2MTP_KOKKOS_H
#define LMP_PAIR_SUS2MTP_KOKKOS_H

	#include "kokkos_type.h"
	#include "kokkos_base.h"
	#include "neigh_list_kokkos.h"
	#include "pair_kokkos.h"
#include "../ML-SUS2/pair_sus2_mtp.h"

namespace LAMMPS_NS {

	template <class DeviceType> class PairSUS2MTPKokkos : public PairSUS2MTP, public KokkosBase {
	 public:
	  // Structs for kernels
	  struct TagPairSUS2MTPInitMomentValsDers {};
	  struct TagPairSUS2MTPPackForwardComm {};
	  struct TagPairSUS2MTPUnpackForwardComm {};
	  struct TagPairSUS2MTPPackReverseComm {};
	  struct TagPairSUS2MTPUnpackReverseComm {};
	  struct TagPairSUS2MTPComputeEnvGate {};
	  struct TagPairSUS2MTPApplyEnvGate {};
	  struct TagPairSUS2MTPComputeGateFirstLayer {};
	  struct TagPairSUS2MTPComputeGateFirstFinalize {};
	  struct TagPairSUS2MTPComputeGateFirstDerivs {};
	  struct TagPairSUS2MTPComputeGateFirstDerivsTeam {};
	  struct TagPairSUS2MTPComputeGateMainAlphaBasic {};
	  struct TagPairSUS2MTPComputeAlphaBasic {};
	  struct TagPairSUS2MTPComputeAlphaTimes {};
	  struct TagPairSUS2MTPSetScalarNbhDers {};
	  struct TagPairSUS2MTPComputeNbhDers {};
	  struct TagPairSUS2MTPComputeEnvRhoChain {};
	  template <int NEIGHFLAG, int EVFLAG> struct TagPairSUS2MTPComputeGateMainForce {};
	  template <int NEIGHFLAG, int EVFLAG> struct TagPairSUS2MTPComputeGateChainForce {};
	  template <int NEIGHFLAG> struct TagPairSUS2MTPComputeGateMainForceTeam {};
	  template <int NEIGHFLAG> struct TagPairSUS2MTPComputeGateChainForceTeam {};
	  template <int NEIGHFLAG, int EVFLAG> struct TagPairSUS2MTPComputeZBLForce {};
	  template <int NEIGHFLAG, int EVFLAG> struct TagPairSUS2MTPComputeForce {};
	  template <int NEIGHFLAG> struct TagPairSUS2MTPComputeForceTeam {};

  enum { EnabledNeighFlags = HALF | HALFTHREAD };
  enum { COUL_FLAG = 0 };
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  using F_FLOAT = KK_FLOAT;

  PairSUS2MTPKokkos(class LAMMPS *);
  ~PairSUS2MTPKokkos() override;
  void compute(int, int) override;
  void settings(int, char **) override;
	  void coeff(int, char **) override;
	  void init_style() override;
	  double init_one(int, int) override;
	  void build_preinterpolation_table();
	  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &,
	                               int, int *) override;
	  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
	  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
	  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d,
	                                  DAT::tdual_double_1d &) override;

  // ========== Kokkos kernels ==========
  //Utility routines
  template <class TagStyle> void check_team_size_for(int, int &, int);

  template <typename scratch_type>
  int scratch_size_helper(int values_per_team);    // Helps calcs scratch size for calcalphabasic

  template <int NEIGHFLAG>
  KOKKOS_INLINE_FUNCTION void v_tally_xyz(EV_FLOAT &ev, const int &i, const int &j,
                                          const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz,
                                          const F_FLOAT &delx, const F_FLOAT &dely,
                                          const F_FLOAT &delz) const;

  KOKKOS_INLINE_FUNCTION F_FLOAT int_pow(F_FLOAT base, int exp) const;

	  KOKKOS_INLINE_FUNCTION void eval_radial_basic_mu_group(const int itype, const int jtype,
	                                                         const F_FLOAT dist, const int mu_group,
	                                                         F_FLOAT &val, F_FLOAT &der) const;
	  KOKKOS_INLINE_FUNCTION void eval_radial_packed_basic_mu_group(
	      const int table_index, const int r_list, const int r_next, const F_FLOAT ddr,
	      const int mu_group, F_FLOAT &val, F_FLOAT &der) const;
	  KOKKOS_INLINE_FUNCTION void eval_radial_value_basic_mu_group(
	      const int itype, const int jtype, const F_FLOAT dist, const int mu_group,
	      F_FLOAT &val) const;
	  KOKKOS_INLINE_FUNCTION void eval_gate_radial_basic_mu_group(
	      const int itype, const int jtype, const F_FLOAT dist, const int mu_group,
	      F_FLOAT &val, F_FLOAT &der) const;
	  KOKKOS_INLINE_FUNCTION void eval_gate_radial_value_basic_mu_group(
	      const int itype, const int jtype, const F_FLOAT dist, const int mu_group,
	      F_FLOAT &val) const;
	  KOKKOS_INLINE_FUNCTION void eval_main_gate_radial_value_basic_mu_group(
	      const int itype, const int jtype, const F_FLOAT dist, const int mu_group,
	      const int gate_atom_index, F_FLOAT &val) const;
	  KOKKOS_INLINE_FUNCTION void eval_main_gate_radial_basic_mu_group(
	      const int itype, const int jtype, const F_FLOAT dist, const int mu_group,
	      const int gate_atom_index, F_FLOAT &val, F_FLOAT &der,
	      F_FLOAT &gate_residual_val) const;
		  KOKKOS_INLINE_FUNCTION void eval_env_gate_weighted_basis(
		      const int itype, const int jtype, const F_FLOAT dist, F_FLOAT *weighted_basis,
		      F_FLOAT *weighted_basis_der) const;
	  KOKKOS_INLINE_FUNCTION void eval_env_gate_rho(const int itype, const F_FLOAT dist,
	                                                F_FLOAT &rho, F_FLOAT &rho_der) const;

  // ---------- SUS2MTP routines (in order of execution) ----------

  // Device-side radial basis function calculation for RBChebyshev_sss

  //Kernels for initing working views
	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPInitMomentValsDers, const int &ii, const int &k) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPPackForwardComm, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPUnpackForwardComm, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPPackReverseComm, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPUnpackReverseComm, const int &ii) const;

  // Kernels for computation
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSUS2MTPComputeEnvGate, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPApplyEnvGate, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void
		  operator()(TagPairSUS2MTPComputeGateFirstLayer,
		             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeGateFirstLayer>::member_type
		                 &team) const;

		  KOKKOS_INLINE_FUNCTION
		  void operator()(TagPairSUS2MTPComputeGateFirstFinalize, const int &ii) const;

		  KOKKOS_INLINE_FUNCTION
		  void operator()(TagPairSUS2MTPComputeGateFirstDerivs, const int &ii) const;

		  KOKKOS_INLINE_FUNCTION
		  void
		  operator()(TagPairSUS2MTPComputeGateFirstDerivsTeam,
		             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeGateFirstDerivsTeam>::member_type
		                 &team) const;

		  KOKKOS_INLINE_FUNCTION
	  void
	  operator()(TagPairSUS2MTPComputeGateMainAlphaBasic,
	             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeGateMainAlphaBasic>::member_type
	                 &team) const;

	  KOKKOS_INLINE_FUNCTION
	  void
	  operator()(TagPairSUS2MTPComputeAlphaBasic,
             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeAlphaBasic>::member_type
                 &team) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSUS2MTPComputeAlphaTimes, const int &ii) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSUS2MTPSetScalarNbhDers, const int &ii, const int &k) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairSUS2MTPComputeNbhDers, const int &ii) const;

	  KOKKOS_INLINE_FUNCTION
	  void operator()(TagPairSUS2MTPComputeEnvRhoChain, const int &ii) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateMainForce<NEIGHFLAG, EVFLAG>,
	             const int &ii) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateMainForce<NEIGHFLAG, EVFLAG>, const int &ii,
	             EV_FLOAT &) const;

	  template <int NEIGHFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateMainForceTeam<NEIGHFLAG>,
	             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeGateMainForceTeam<NEIGHFLAG>>::member_type
	                 &team) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateChainForce<NEIGHFLAG, EVFLAG>,
	             const int &ii) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateChainForce<NEIGHFLAG, EVFLAG>, const int &ii,
	             EV_FLOAT &) const;

	  template <int NEIGHFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeGateChainForceTeam<NEIGHFLAG>,
	             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeGateChainForceTeam<NEIGHFLAG>>::member_type
	                 &team) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeZBLForce<NEIGHFLAG, EVFLAG>,
	             const int &ii) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeZBLForce<NEIGHFLAG, EVFLAG>, const int &ii,
	             EV_FLOAT &) const;

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeForce<NEIGHFLAG, EVFLAG>,
             const int &ii) const;    // This eventually calls the below version

	  template <int NEIGHFLAG, int EVFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeForce<NEIGHFLAG, EVFLAG>, const int &ii,
	             EV_FLOAT &) const;    // With global energy reduction as needed

	  template <int NEIGHFLAG>
	  KOKKOS_INLINE_FUNCTION void
	  operator()(TagPairSUS2MTPComputeForceTeam<NEIGHFLAG>,
	             const typename Kokkos::TeamPolicy<DeviceType, TagPairSUS2MTPComputeForceTeam<NEIGHFLAG>>::member_type
	                 &team) const;

 protected:
  int input_chunk_size, chunk_size,
      chunk_offset;    // Needed to process the computation in batches to avoid running out of VRAM.

  // Characteric flags
  int inum, max_neighs;
  int host_flag, neighflag;

  int eflag, vflag;    // Energy and virial flag

	  typename AT::t_neighbors_2d d_neighbors;    //
	  typename AT::t_int_1d_randomread d_ilist;
	  typename AT::t_int_1d_randomread d_numneigh;
	  typename AT::t_int_1d d_sendlist;
	  typename AT::t_double_1d_um v_buf;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  DAT::ttransform_kkacc_1d_9 k_cvatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;
  typename AT::t_kkacc_1d_9 d_cvatom;

  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d_randomread type;

  // ---------- Device Arrays  ----------
  // Alphas indicies
  Kokkos::View<int **, DeviceType> d_alpha_index_basic;      // For constructing of basic alphas.
  Kokkos::View<int **, DeviceType> d_alpha_index_times;      // For combining alphas
  Kokkos::View<int *, DeviceType> d_alpha_moment_mapping;    // Maps alphas to basis functions.
  Kokkos::View<int *, DeviceType> d_alpha_basic_sh_index;     // Real-SH flat index for SUS2-SH basic alphas.
  Kokkos::View<double *, DeviceType> d_alpha_times_coeff;     // Tensor-product coefficient for SH products.

  // SUS2-MLIP mapping arrays
  Kokkos::View<int *, DeviceType> d_mu_to_K;              // Mapping from radial function to scaling dimension
  Kokkos::View<int *, DeviceType> d_mu_to_sigma;          // Mapping from radial function to sigma index
  Kokkos::View<int *, DeviceType> d_k_to_mu_offsets;      // CSR offsets for mu grouped by scaling channel
  Kokkos::View<int *, DeviceType> d_grouped_mu;           // Mu indices grouped by scaling channel
  Kokkos::View<int *, DeviceType> d_mu_radial_offset;     // Precomputed offset into radial coeff block per mu
  Kokkos::View<int *, DeviceType> d_mu_to_basic_group;    // Maps mu to compact basic-mu group index, or -1
  Kokkos::View<double **, DeviceType> d_grouped_radial_coeffs;    // Radial basis coeffs grouped by scaling channel
  Kokkos::View<int *, DeviceType> d_basic_mu_offsets;     // CSR offsets for basic moments grouped by mu
  Kokkos::View<int *, DeviceType> d_basic_mu_values;      // Unique mu values appearing in alpha_index_basic
  Kokkos::View<int *, DeviceType> d_basic_grouped_indices;    // alpha_index_basic indices grouped by mu
  Kokkos::View<int *, DeviceType> d_alpha_basic_mu_group; // Compact basic-mu group index for each basic alpha
  Kokkos::View<int *, DeviceType> d_pair_to_table_index;  // Maps full species pair to sparse table index

  // The learned coefficients.
  Kokkos::View<double *, DeviceType> d_species_coeffs;     // The species-based constants
  Kokkos::View<double *, DeviceType> d_linear_coeffs;      // Basis coeffs

  // SUS2-MLIP regression coefficients (unified array)
  // Layout: [shift_coeffs][scal_coeffs][radial_basis_coeffs]
  // Use offsets to access different sections
  Kokkos::View<double *, DeviceType> d_regression_coeffs;    // Unified array
  Kokkos::View<double *, DeviceType> d_shift_coeffs;        // Shift coefficients for each species

  using radial_table_value_type = double;
  bool do_list = false;                                      // Enable/disable preinterpolation table
  double inv_dr = 0.0;                                       // 1/dr for table indexing
  Kokkos::View<radial_table_value_type***, DeviceType,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      d_radial_list;                                         // Values: [species_pairs][grid_size][basic_mu_group_count]
  Kokkos::View<radial_table_value_type***, DeviceType,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_radial_der_list;                                     // Derivatives: [species_pairs][grid_size][basic_mu_group_count]
  Kokkos::View<radial_table_value_type *, DeviceType,
               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
      d_radial_vd_list;                                      // Interleaved [value,derivative] for force kernels
	  Kokkos::View<radial_table_value_type***, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_two_layer_gate_radial_list;                           // Values: [species_pairs][grid_size][basic_mu_group_count]
	  Kokkos::View<radial_table_value_type***, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_two_layer_gate_radial_der_list;                       // Derivatives: [species_pairs][grid_size][basic_mu_group_count]
	  Kokkos::View<radial_table_value_type***, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_env_gate_radial_list;                                 // Values: [species_pairs][grid_size][6]
	  Kokkos::View<radial_table_value_type***, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_env_gate_radial_der_list;                             // Derivatives: [species_pairs][grid_size][6]
	  Kokkos::View<radial_table_value_type**, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_env_gate_rho_list;                                     // Values: [central_species][grid_size]
	  Kokkos::View<radial_table_value_type**, DeviceType,
	               Kokkos::MemoryTraits<Kokkos::RandomAccess>>
	      d_env_gate_rho_der_list;                                 // Derivatives: [central_species][grid_size]

  using moment_buffer_value_type = double;
  using nbh_der_buffer_value_type = double;

  // Global working buffers.
  Kokkos::View<moment_buffer_value_type **, DeviceType> d_moment_tensor_vals;
  Kokkos::View<nbh_der_buffer_value_type **, DeviceType> d_nbh_energy_ders_wrt_moments;
	  Kokkos::View<double *, DeviceType> d_env_gate_values;
	  Kokkos::View<double *, DeviceType> d_env_gate_der_prefactors;
	  Kokkos::View<double *, DeviceType> d_env_gate_rho_chain_values;
	  Kokkos::View<moment_buffer_value_type **, DeviceType> d_env_gate_activation_basic_vals;
	  Kokkos::View<double *, DeviceType> d_two_layer_gate_values;
	  Kokkos::View<double *, DeviceType> d_two_layer_gate_adjoints;
	  Kokkos::View<double ***, DeviceType> d_two_layer_gate_first_derivs;
	  Kokkos::View<double **, DeviceType> d_two_layer_gate_mu_multipliers;
	  Kokkos::View<double **, DeviceType> d_two_layer_gate_mu_derivs;
	  Kokkos::View<int *, DeviceType> d_two_layer_gate_weight_moment_indices;
	  Kokkos::View<double *, DeviceType> d_two_layer_gate_weights;
	  Kokkos::View<double *, DeviceType> d_two_layer_gate_additive_coeffs;
	  Kokkos::View<int *, DeviceType> d_zbl_atomic_numbers;
	  Kokkos::View<double *, DeviceType> d_zbl_pair_inner_cutoffs;
	  Kokkos::View<double *, DeviceType> d_zbl_pair_outer_cutoffs;
	  Kokkos::View<double *, DeviceType> d_zbl_pair_outer_sq;

  // Typedefs for shared memory
  typedef Kokkos::View<F_FLOAT **[3], typename DeviceType::scratch_memory_space,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      shared_double_3d;    // Used for coord powers
	  typedef Kokkos::View<F_FLOAT **, typename DeviceType::scratch_memory_space,
	                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
	      shared_double_2d;    // Used for radial basis vals, ders, and dist powers
	  typedef Kokkos::View<F_FLOAT *, typename DeviceType::scratch_memory_space,
	                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>
	      shared_double_1d;    // Used for per-center SH adjoint values

  // SUS2-MLIP parameters
	  int L_max;             // Maximum moment tensor level
	  int K_scaling;         // Scaling dimension parameter
	  int regression_coeffs_count;    // Total size of unified regression_coeffs array
	  int basic_mu_group_count;       // Number of unique mu values in alpha_index_basic
	  int two_layer_gate_product_limit = 0;
	  int first;
	  int need_dup;

  // ---------- Define the forces, per-atom energy, and virials----------
  using KKDeviceType = typename KKDevice<DeviceType>::value;

  template <typename DataType, typename Layout>
  using DupScatterView =
      KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterDuplicated>;

  template <typename DataType, typename Layout>
  using NonDupScatterView =
      KKScatterView<DataType, Layout, KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  DupScatterView<KK_ACC_FLOAT *[3], typename DAT::t_kkacc_1d_3::array_layout> dup_f;
  DupScatterView<KK_ACC_FLOAT *[6], typename DAT::t_kkacc_1d_6::array_layout> dup_vatom;
  DupScatterView<KK_ACC_FLOAT *[9], typename DAT::t_kkacc_1d_9::array_layout> dup_cvatom;

  NonDupScatterView<KK_ACC_FLOAT *[3], typename DAT::t_kkacc_1d_3::array_layout> ndup_f;
  NonDupScatterView<KK_ACC_FLOAT *[6], typename DAT::t_kkacc_1d_6::array_layout> ndup_vatom;
  NonDupScatterView<KK_ACC_FLOAT *[9], typename DAT::t_kkacc_1d_9::array_layout> ndup_cvatom;

  friend void pair_virial_fdotr_compute<PairSUS2MTPKokkos>(PairSUS2MTPKokkos *);
};

}    // namespace LAMMPS_NS

#endif
#endif

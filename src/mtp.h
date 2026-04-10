/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifndef MLIP_MTP_H
#define MLIP_MTP_H


#include "common/multidimensional_arrays.h"
#include "configuration.h"
#include "neighborhoods.h"
#include "radial_basis.h"
#include "basic_mlip.h"


// Moment Thensor Potential
class MTP : public AnyLocalMLIP
{
private:
	const int default_radial_basis_size = 8;		// will be set if not specified in .mtp file

	int alpha_moments_count = 0;					//	/=================================================================================================
	int alpha_index_basic_count = 0;				//	|                                                                                                |
	int(*alpha_index_basic)[4] = nullptr;			//	|   Internal representation of Moment Tensor Potential basis                                     |
	int alpha_index_times_count = 0;				//	|   These items is required for calculation of basis functions values and their derivatives      |
	int(*alpha_index_times)[4] = nullptr;			//	|                                                                                                |
	int *alpha_moment_mapping = nullptr;			//	\=================================================================================================

	int radial_funcs_count = 0;
	Array2D radial_coeffs;

	Array3D moment_jacobian_;
	std::vector<double> site_energy_ders_wrt_moments_;
	std::vector<double> dist_powers_;
	std::vector<Vector3> coords_powers_;

	std::string tag;

protected:
	Array1D moment_vals;							//!< Array of basis function values calculated for certain atomic neighborhood
	Array3D moment_ders;							//!< Array of basis function derivatives w.r.t. motion of neighboring atoms calculated for certain atomic neighborhood
	Array1D basis_vals;								//!< Array of the basis functions values calculated for certain neighborhood of the certain configuration
	Array3D basis_ders;								//!< Array derivatives w.r.t. each atom in neiborhood of the basis functions values calculated for certain neighborhood of the certain configuration

protected:
	int alpha_scalar_moments = 0;					//!< = alpha_count-1 (MTP-basis except constant function)

	void MemAlloc();								// 
	void MemFree();									//

	// I/O functions, parts of Save() and Load()
	void ReadRadialBasisOldFormat(std::ifstream& ifs);
	void ReadRadialCoeffs(std::ifstream& ifs);
	void ReadMTPBasis(std::ifstream& ifs);			// 
	void WriteMTPBasis(std::ofstream& ofs);			// 
	void ReadRegressCoeffs(std::ifstream& ifs);
	void WriteRegressCoeffs(std::ofstream& ofs);

public:
	int alpha_count = 0;							//!< Basis functions count 

	std::string pot_desc = "NoName";				//!< Some description of potential

	Array1D regress_coeffs;							// Array of regression coefficients

	MTP(const std::string& filename);
	MTP(const Settings& settings);
	
	~MTP();

	int CoeffCount() override;
	double* Coeff() override;

	void Load(const std::string& filename) override;				//!< Loading MTP from a file
	void Save(const std::string& filename) override;				//!< Saving MTP to a file

	void CalcBasisFuncs(const Neighborhood& nbh);					//! Calculates the basis function values (basis_vals) for given atomic Neighborhood.
	void CalcSiteEnergyGrad(const Neighborhood& nbh, std::vector<double>& out_se_grad) override;	// The same as previous
	void AddSiteEnergyGrad(const Neighborhood& nbh, std::vector<double>& out_accumul_arr) override;

	void CalcBasisFuncsDers(const Neighborhood& nbh);				//! Calculates the basis function values and their derivatives for given atomic Neighborhood.

	double SiteEnergy(const Neighborhood& nbh) override;
	void CalcSiteEnergyDers(const Neighborhood& nbh) override;		//! Calculates the basis function values and their derivatives for given atomic Neighborhood.

	void CalcEFSGrads(	const Configuration& cfg,					//! Calculates arrays of energy, forces and stresses components 
						Array1D& out_ene_grad,
						Array3D& out_frc_grad,
						Array3D& out_str_grad) override;

	void CalcE(Configuration& cfg) override;
	void CalcEFS(Configuration& cfg) override;
	void CalcEFSDebug(Configuration& cfg);							//!< original implementation of CalcEFS() with forward propagation (not optimal)

	// Implementation of the following function is provided but it shouldn't be called due to inefficiency
	void AccumulateCombinationGrad(	const Neighborhood& nbh,
									std::vector<double>& out_grad_accumulator,
									const double se_weight = 0.0,
									const Vector3* se_ders_weights = nullptr) override;
};

#endif
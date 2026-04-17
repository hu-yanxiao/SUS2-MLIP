/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_BASICMLIP_H
#define MLIP_BASICMLIP_H

#pragma once
#include "basic_potentials.h"
#include "radial_basis.h"
#include "common/multidimensional_arrays.h"


//! Defines a MLIP through sum of contribitions of each local patch of atoms
//! This virtual class requires its child to implement site energy calculations
class AnyLocalMLIP : public AnyPotential
{
private:
	bool rb_owner = false;							//!< true if RadialBasis is created by this object (not externally transfered). Requires for correct deletion of RadialBasis
	std::vector<Vector3> tmp_se_ders_weights_;		//!< Temporal array storing the last argument (se_ders_weights) for AccumulateCombinationGrad function

protected:
	Array1D tmp_grad_accumulator_;					//!< Temporal variable storing gradient of site energy w.r.t. coefficients
	
public:
	AnyRadialBasis* p_RadialBasis = nullptr;		//!< pointer to RadialBasis

public:
	double buff_site_energy_;						//!< Temporal variable storing site energy after its calculation for a certain neighborhood
	double buff_site_energy_0;
	//double cn_cf;
	std::vector<Vector3> buff_site_energy_ders_;	//!< Temporal variable storing derivatives of site energy w.r.t. positions of neghboring atoms after their calculation for a certain neighborhood

//
// Virtual functions. Some of them implemented in this class but not optimally. It is recomended to provide their optimal implementations in derived classes
//
public:
	// Functions related to coefficients
	virtual int CoeffCount() = 0;					//!< Number of coefficients
	virtual double* Coeff() = 0;					//!< Coefficients themselves (returns pointer to the first coefficient in array)

	// MLIP's I/O functions
	virtual void Save(const std::string& filename);		//!< Writes a file with MLIP's data (structure and coefficints if fitted)
	virtual void Load(const std::string& filename);		//!< Loads MLIP from file (with or without coefficients)

	// Basic functions required for EFS calculations
	virtual double SiteEnergy(const Neighborhood& nbh);												// Must calculate and return site energy for a given atomic neighborhood
	virtual void CalcSiteEnergyDers(const Neighborhood& nbh);										// Must calculate site energy and derivatives of site energy w.r.t. positions of neghboring atoms for a given neighborhood. Must update buff_site_energy_ and buff_site_energy_ders_

protected:
	//! Basic function required for training of nonlinear MLIPs.
	virtual void AccumulateCombinationGrad( const Neighborhood& nbh,								// Must calculate and accumulate for a given atomic neighborhood nbh the gradient w.r.t coefficients of a linear combination site_energy*se_weight + scalar_product(se_ders_weights, site_energy_ders). It must also update buff_site_energy_ and buff_site_energy_ders_
											Array1D& out_grad_accumulator,							// Gradient of linear combination must be accumulated in out_grad_accumulator. It should not be zeroed before accumulation in this function
											const double se_weight = 0.0,							// se_weight is a weight of site energy gradients (w.r.t. coefficients) in a linear combination
											const Vector3* se_ders_weights = nullptr) = 0;			// se_ders_weights is an array of gradients (w.r.t. coefficients) of site energy derivatives (w.r.t. atomic positions) in a linear combination. If last two arguments are default it must just update buff_site_energy_ and buff_site_energy_ders_
	virtual void PrepareEvalCaches();															// Optional hook for derived classes to refresh cached coefficient views before repeated neighborhood evaluation

	// Function accumalating gradients of site energy. Required for different purposes
	virtual void AddSiteEnergyGrad(const Neighborhood& nbh, Array1D& out_se_grad_accumulator);		// Must calculate gradient of site energy w.r.t. coefficients and add it to out_se_grad_accumulator. out_se_grad_accumulator is not zeroed before calculation

public:
	// Functions calculating gradients w.r.t. coefficients. Required by Maxvol-based selection routines
	virtual void CalcSiteEnergyGrad(const Neighborhood& nbh, Array1D& out_se_grad);					// Must calculate gradient of site energy w.r.t. coefficients and write them in the second argument.
	virtual void CalcEnergyGrad(Configuration &cfg, std::vector<double>& out_energy_grad);			// Calculate gradient of energy w.r.t. coefficients and write them in second argument
//	virtual void CalcForceGrad(	const Configuration& cfg,											// Must calculate gradient of the force on ind-th atom w.r.t. coefficients and write them in the third argument
//								const int ind, 
//								std::vector<Vector3>& out_frc_grad);			
	virtual void CalcStressesGrads(const Configuration& cfg, Array3D& out_str_grad);				// Must calculate gradient of the stress tensor w.r.t. coefficients and write them in the third argument. Order of indices in out_str_grad: (i,j,k) where i,j are the stress tensor components, k indicates derivative over k-th coefficient of MLIP
	virtual void CalcEFSGrads(	const Configuration& cfg,											// Must calculate gradients of the energy, all forces and stress tensor w.r.t. coefficients and write them in output parameters. Order of indices in out_frc_grad: (i,j,k) where i is atom index, j is force component (0,1,2), k indicates derivative over k-th coefficient of MLIP. Order of indices in out_str_grad is the same as in CalcStressesGrads(...)
								Array1D& out_ene_grad,
								Array3D& out_frc_grad,
								Array3D& out_str_grad);

	//! Function implementing soft constraints on coefficients (e.g., so that the norm of the radial function is 1) or regularization implementing as penalty to training procedure
	virtual void AddPenaltyGrad(const double coeff,													// Must calculate penalty and optionally (if out_penalty_grad_accumulator != nullptr) its gradient w.r.t. coefficients multiplied by coeff to out_penalty
								double& out_penalty_accumulator, 
								Array1D* out_penalty_grad_accumulator = nullptr);

	// CalcEFS
	virtual void CalcE(Configuration& cfg);
	virtual void CalcEFS(Configuration &cfg);
	virtual void CalcEFS(Configuration &cfg, const Neighborhoods& neighborhoods);

	//! Destructor
	virtual ~AnyLocalMLIP();

//
// Non-virtual functions
//
public:	
	// functions calculating mindist and cutoff
	inline double CutOff();
	inline double MinDist();
	
	// Constructors
	AnyLocalMLIP(double _min_dist, double _max_dist, int _radial_basis_size);
	AnyLocalMLIP(AnyRadialBasis* _p_radial_basis);
	AnyLocalMLIP() {}


	// Called by Trainer in case of nonlinear MLIP
	void AccumulateEFSCombinationGrad(	Configuration &cfg,											// Calculates gradient w.r.t. coefficients of a linear combination of energy, forces and stresses via linear combination of gradients of site energies and site energy derivatives w.r.t. positions of neighboring atoms
		                                std::vector<double>& ene_weight,
										const std::vector<Vector3>& frc_weights,
										const Matrix3& str_weights,
										Array1D& out_grads_accumulator);
	void AccumulateEFSCombinationGrad(	Configuration &cfg,
		                                std::vector<double>& ene_weight,
										const std::vector<Vector3>& frc_weights,
										const Matrix3& str_weights,
										Array1D& out_grads_accumulator,
										const Neighborhoods& neighborhoods);
};



inline double AnyLocalMLIP::CutOff()
{
#ifdef MLIP_DEBUG
	if (p_RadialBasis == nullptr)
		ERROR("Attempting to access uninitialized RadialBasis");
#endif 
	return p_RadialBasis->max_dist;
}

inline double AnyLocalMLIP::MinDist()
{
#ifdef MLIP_DEBUG
	if (p_RadialBasis == nullptr)
		ERROR("Attempting to access uninitialized RadialBasis");
#endif
	return p_RadialBasis->min_dist;
}


#endif //#ifndef MLIP_BASICMLIP_H


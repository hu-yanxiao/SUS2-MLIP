/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Ivan Novikov
 */

#ifndef MLIP_EAMMult
#define MLIP_EAMMult

#include "../src/basic_mlip.h"

//! This is a multi-component EAM potential
//! Uses the AnyLocalMLIP class, allowing to just define the site potentials
class EAMMult : public AnyLocalMLIP
{
protected:
	std::vector<double> regression_coeffs;

public:
	int atom_types_count_; //!< number of atom types in subsctance fitted
	int radial_basis_count_; //!< number of radial basis functions used
	int embedding_parameters_count_; //!< number of terms in f(rho) = c_0 + c_2 rho^2 + c rho^3 + ...
	//!< note that c_1 rho is missing intentionally

	EAMMult(int _radial_basis_count, int embedding_function_parameters_count, int atom_types_in_substance_count,
		double _min_dist = 1.6, double _max_dist = 5.0)
		: AnyLocalMLIP(_min_dist, _max_dist, _radial_basis_count)
		, radial_basis_count_(_radial_basis_count)
		, embedding_parameters_count_(embedding_function_parameters_count)
		, atom_types_count_(atom_types_in_substance_count)
		, regression_coeffs(atom_types_in_substance_count * (2 * atom_types_in_substance_count *_radial_basis_count + embedding_function_parameters_count))
	{}
	EAMMult(AnyRadialBasis* _p_radial_basis, int embedding_function_parameters_count, int atom_types_in_substance_count)
		: AnyLocalMLIP(_p_radial_basis)
		, embedding_parameters_count_(embedding_function_parameters_count)
		, atom_types_count_(atom_types_in_substance_count)
		, regression_coeffs(atom_types_in_substance_count * (2 * atom_types_in_substance_count *_p_radial_basis->rb_size + embedding_function_parameters_count))
	{}

	~EAMMult();

	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}

protected:
	void AddPenaltyGrad(const double coeff,	
						double& out_penalty_accumulator, 
						Array1D* out_penalty_grad_accumulator) override;
	void AddSiteEnergyGrad(const Neighborhood&, std::vector<double>&) override;
	void CalcSiteEnergyDers(const Neighborhood& nbh) override;
	void AccumulateCombinationGrad(const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double se_weight = 0.0,
		const Vector3* se_ders_weights = nullptr) override;
};

#endif // MLIP_EAMMult


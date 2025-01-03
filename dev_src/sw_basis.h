/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_SW_BASIS
#define MLIP_SW_BASIS

#include "../src/basic_mlip.h"
//#include "../src/common/multidimensional_arrays.h"
//#include "../src/radial_basis.h"

class StillingerWeberRadialBasis : public AnyLocalMLIP
{
	private:
		//const std::string filename;	
		int n_at_types;
		std::string rbasis_type;
		//double max_dist;
		//double min_dist;
		//int radial_basis_size;

		//AnyRadialBasis* p_RadialBasis = nullptr;

		Array1D regression_coeffs;
		Array1D rad_bas_val_j;
		Array1D rad_bas_der_j;		
		Array1D en_der_wrt_dist_j_grad;
		Array1D en_der_wrt_dist_k_grad;

	public:
		void Load(const std::string&) override;
		void Save(const std::string&) override;

		void AddPenaltyGrad(const double, double&, Array1D*);
		void AccumulateEnergy2body(Configuration&, const Neighborhood&);
		void AccumulateEFS2body(Configuration&, const Neighborhood&, const int);
		void AccumulateEnergyGrad2body(const Neighborhood&, Array1D&);
		void AccumulateEFSGrad2body(Configuration&, const Neighborhood&, const int, 
		Array1D&, Array3D&, Array3D&);
		void AccumulateEnergyGrad3body(const Neighborhood&, Array1D&);
		void AccumulateEFS3body(Configuration&, const Neighborhood&, const int);
		void AccumulateEFSGrad3body(Configuration&, const Neighborhood&, const int, 
		Array1D&, Array3D&, Array3D&); 
		void CalcE(Configuration&) override;
		void CalcEFS(Configuration&) override;
		void CalcEFSGrads(Configuration&, Array1D&, Array3D&, Array3D&);
		void CalcEnergyGrad(Configuration&, Array1D&) override;
		void AccumulateCombinationGrad(const Neighborhood& nbh, Array1D& out_grad_accumulator, 
		const double se_weight = 0.0, const Vector3* se_ders_weights = nullptr);

		StillingerWeberRadialBasis(const std::string& filename);

		~StillingerWeberRadialBasis()
		{
			delete p_RadialBasis;	
		}

		int CoeffCount() 
		{
			return (int)regression_coeffs.size();
		}
		double* Coeff() 
		{
			return &regression_coeffs[0];
		}
};

#endif

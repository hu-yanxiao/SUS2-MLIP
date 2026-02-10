/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../src/basic_mlip.h"
#include "../src/common/multidimensional_arrays.h"
#include <map>
#ifndef MLIP_MTPR_H
#define MLIP_MTPR_H

struct BasicIndices {
        std::vector<int> comp0, comp1, comp2, comp3;
    };



class MLMTPR : virtual public AnyLocalMLIP
{
protected:
	void MemAlloc();

	void ReadLinearCoeffs(std::ifstream& ifs);				//Read linear regression coeffs from MTP file
	void WriteLinearCoeffs(std::ofstream& ofs);				//Write linear regression coeffs to MTP file


	int alpha_moments_count;						//	/=================================================================================================
	int alpha_index_basic_count;					//	|                                                                                                |
	int(*alpha_index_basic_comp0);
	int(*alpha_index_basic_comp1);
	int(*alpha_index_basic_comp2);
	int(*alpha_index_basic_comp3);
	int(*alpha_index_basic)[4];


    			                        //	|   Internal representation of Moment Tensor Potential basis                                     |
	int alpha_index_times_count;					//	|   These items is required for calculation of basis functions values and their derivatives      |
	int(*alpha_index_times)[4];						//	|                                                                                                |
	int(*alpha_index_times_comp0);
	int(*alpha_index_times_comp1);
	int(*alpha_index_times_comp2);
	int(*alpha_index_times_comp3);
	int *alpha_moment_mapping;						//	\=================================================================================================
	std::string pot_desc;
	std::string rbasis_type;
	std::string scaling_map;
	
	double *moment_vals; //!< Array of basis function values calculated for certain atomic neighborhood
	Array3D moment_ders;//!< Array of basis function derivatives w.r.t. motion of neighboring atoms calculated for certain atomic neighborhood
	double *basis_vals;	//!< Array of the basis functions values calculated for certain neighbor	hood of the certain configuration
	Array3D basis_ders;	//!< Array derivatives w.r.t. each atom in neiborhood of the basis functions values calculated for certain neighborhood of the certain configuration

	Array3D moment_jacobian_;
	std::vector<double> site_energy_ders_wrt_moments_;
	std::vector<double> dist_powers_;
	std::vector<double> coords_powers_x;
	std::vector<double> coords_powers_y;
	std::vector<double> coords_powers_z;

	std::vector<Vector3> coords_powers_;



	void CalcSiteEnergyDers(const Neighborhood& nbh) override;

public:
	double scaling = 1.0; //!< how to scale moments
	std::vector<double> regression_coeffs;
	Array3D radial_coeffs;	
							//!< array of radial coefficients
	int L=0;
	int K_ = 1;
    std::vector<double> mu_;
    bool is_fixed=false;

    BasicIndices alpha_index_basic_;
    BasicIndices alpha_index_times_;
    Array3D radial_list;
    Array3D radial_der_list;
    //const Array3D& get_radial_list() const { return radial_list; }
    //const Array3D& get_radial_der_list() const { return radial_der_list; }
    double inv_dr;

	std::vector<double> linear_coeffs;
	//std::map<int, int> mu_to_K;
	std::vector<int> mu_to_K;
	std::vector<int> mu_to_sigma;
	std::vector<double> linear_mults;					//!< array of multiplers for basis functions
	std::vector<double> max_linear;						//!< maximum absolute values of basis functions
	//double cn_cf;
	std::vector<double> max_radial;                         	//!< maximum values of r.b.f on the training set

	bool inited = false;
        bool shift_ = true;
	double* energy_cmpnts;								// Energy components for SLAE matrix
	Array3D forces_cmpnts;								// Force components for SLAE matrix
	double(*stress_cmpnts)[3][3];						// Stress components for SLAE matrix

	void CalcBasisFuncs(Neighborhood& Neighborhood, double* bf_vals); //Linear basic functions calculation
	void CalcBasisFuncsDers(const Neighborhood& Neighborhood);		//Linear basic functions and their derivatives calculation
	void CalcEFSComponents(Configuration& cfg);						//Calculate the components for linear regression matrix
	void CalcEComponents(Configuration& cfg);			//Calculate the components for linear regression matrix
//	void CalcEFS(Configuration& cfg) override
//	{
//		AnyLocalMLIP::CalcEFS(cfg);
//		cfg.features["EFS_by"] = "MultiMTP" + to_string(alpha_count);
//	}

	void AccumulateCombinationGrad(	const Neighborhood& nbh,
									std::vector<double>& out_grad_accumulator,
									const double se_weight = 0.0,
									const Vector3* se_ders_weights = nullptr) override;
	MLMTPR():
		AnyLocalMLIP() {
	}
	MLMTPR(const std::string& mtpfnm) :
		AnyLocalMLIP() {
		Load(mtpfnm);
	}

	~MLMTPR();
        void CalcDescriptors(Configuration& cfg, std::ofstream &ofs);
        void CalcpartialE(Configuration& cfg, std::ofstream &ofs);
	void ReadMTPBasis(std::ifstream& ifs);		// Read MTP basis from file	
	void WriteMTPBasis(std::ofstream& ofs);		// Write MTP basis for file

	void Load(const std::string& filename) override;
	void Save(const std::string& filename) override;
        void Save_2(const std::string& filename);
	void RadialCoeffsInit(std::ifstream& ifs_rad);
	void Perform_scaling();

	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}
	std::vector<double> LinCoeff()								//returns linear coefficients
	{
		int Rsize = species_count+2*species_count*species_count* K_ +(p_RadialBasis->rb_size + species_count)*radial_func_count;

		for (int i = Rsize; i < Rsize + alpha_count + species_count - 1;i++)
			linear_coeffs[i-Rsize]=regression_coeffs[i];


		return linear_coeffs;
	}
	void DistributeCoeffs()									//Combine radial and linear coefficients in one array
	{
		int radial_size = (int)regression_coeffs.size();
		int max_comp = species_count - 1;				//maximum index of component

		regression_coeffs.resize(radial_size + alpha_count + max_comp);


		if (linear_coeffs.size() == alpha_count)
		{
			for (int i = 0; i <= max_comp; i++)
				regression_coeffs[radial_size + i] = linear_coeffs[0];				//constants for component's site energy shift

			for (int i = 1; i < alpha_count; i++)
				regression_coeffs[radial_size + i + max_comp] = linear_coeffs[i];

		}
		else
			for (int i = 0; i < (int)linear_coeffs.size(); i++)
				regression_coeffs[radial_size + i] = linear_coeffs[i];

		linear_mults.resize(alpha_scalar_moments);
		max_linear.resize(alpha_scalar_moments);

		for (int i=0;i<alpha_scalar_moments;i++)
		{
			linear_mults[i]=1;
			max_linear[i]=1e-10;

		}


	}
	int Get_RB_size()
	{
		return p_RadialBasis->rb_size;
	}

	void AddPenaltyGrad(const double coeff, 
						double& out_penalty_accumulator, 
						Array1D* out_penalty_grad_accumulator) override;


	int alpha_count;								//!< Basis functions count 
	int alpha_scalar_moments;						//!< = alpha_count-1 (MTP-basis except constant function)
	int radial_func_count;							//!< number of radial basis functions used
	int species_count;							//!< number of components present in the potential

	void Orthogonalize();						//!<Orthogonalize the basic functions
};



#endif

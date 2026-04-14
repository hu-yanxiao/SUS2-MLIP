/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_MTPR_TRAINER_H
#define MLIP_MTPR_TRAINER_H

#include "../src/common/stdafx.h"
#include "non_linear_regression.h"
#include "mtpr.h"
#include "../src/common/bfgs.h"
#include <vector>

class MTPR_trainer : public NonLinearRegression
{
public:
	struct TrainErrorSummary {
		double energy_mae_mev_atom = 0.0;
		double energy_rmse_mev_atom = 0.0;
		double force_mae_mev_a = 0.0;
		double force_rmse_mev_a = 0.0;
		double stress_mae_ev = 0.0;
		double stress_rmse_ev = 0.0;
	};

private:
	//LINEAR REGRESSION
	double* quad_opt_matr = nullptr;					// Pointer to "active" (either equals Mtrx1 or Mtrx2) SLAE matrix (which is modified and being solved) of least squres minimization problem A*A^T = b*A^T 
	double* quad_opt_vec = nullptr;						// Pointer to "active" (either equals RightPart1 or RightPart2) SLAE right part (which is modified and being solved) of least squres minimization problem A*A^T = b*A^T 
	double quad_opt_scalar;								// Scalar value in least squares minimization problem A*A^T = b*A^T (Not needed in most scenarios) 
	int quad_opt_eqn_count;								// Number of equation in overdetermined system
	int quad_opt_allocated_n = 0;
	std::vector<double> lin_force_block_;
	std::vector<double> lin_force_rhs_;
	std::vector<double> lin_stress_block_;
	std::vector<double> lin_stress_rhs_;

	Array2D inv_hessian;
	bool have_hess = false;								//is hessian currently approximated

	MLMTPR* p_mlmtpr = nullptr;
	BFGS bfgs;
	double bfgs_f;
	Array1D bfgs_g;
	TrainErrorSummary last_train_error_summary_;
	bool have_last_train_error_summary_ = false;

public:
	std::string curr_pot_name="";
	std::string bfgs_trace_file="";
	int max_step_count = 2000;
	double linstop = 1e-8;
	int random_perturb = 0;
        bool do_lin=true;
	int do_lin_step_limit = 1000;
	int do_lin_frequency = 50;
	MTPR_trainer(MLMTPR* _p_mlip,						// Constructor requires MTP basis
						double opt_en_coeff = 1.0,				//	Optional parameters are the weights coeficients of energy, forces and stresses equations in minimization problem
						double opt_fr_coeff = 1.0,
						double opt_st_coeff = 1.0,
						double _wgt_rel_forces = 0.0,
						double OptCnCoeff = 1.0e-6,
						std::string _curr_pot_name = "",
						int _norm_by_forces=0,
						int _random_perturb=0) :
		NonLinearRegression(_p_mlip, 
							opt_en_coeff, 
							opt_fr_coeff, 
							opt_st_coeff, 
							_wgt_rel_forces, 
							OptCnCoeff)
	{
		p_mlmtpr = _p_mlip;
		ClearSLAE();
		curr_pot_name = _curr_pot_name;
		norm_by_forces = _norm_by_forces;
		random_perturb = _random_perturb;
		bfgs_g.resize(p_mlmtpr->CoeffCount());
	};
	~MTPR_trainer();
	const TrainErrorSummary& LastTrainErrorSummary() const { return last_train_error_summary_; }
	bool HasLastTrainErrorSummary() const { return have_last_train_error_summary_; }

	void LoadWeights(std::ifstream& ifs);						// Load weights in Linear regression from file
        void shift(bool shift_);
	void ClearSLAE();											// Setting SLAE Matrix and right part to zero
	void SymmetrizeSLAE();										// Symmetrization of the SLAE matrix before solving (only upper right part is filled during adding or removing configuration to regression)
	void SolveSLAE();											// Find the corresponding linear coefficients
	void AddToSLAE(Configuration& cfg, double weight = 1, const Neighborhoods* neighborhoods = nullptr);	// Adds configuration to regression SLAE. If weight = -1 removes from regression

	double* ConstructLinHessian();
	void TrainLinear(int prank,
					std::vector<Configuration>& training_set,
					const std::vector<Neighborhoods>* neighborhoods = nullptr,
					const std::string& context = "");
        void random_sample(int prank, std::vector<Configuration>& training_set, int max_step, const std::vector<Neighborhoods>* neighborhoods = nullptr);
	#ifndef ALGLIB
		void Train(std::vector<Configuration>& training_set) override; //with Shapeev bfgs
	#else
		void Train2(std::vector<Configuration>& training_set);
	#endif

	
};

#endif // MLIP_MTPR_TRAINER_H

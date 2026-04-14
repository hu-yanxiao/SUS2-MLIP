/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_NON_LINEAR_REGRESSION_H
#define MLIP_NON_LINEAR_REGRESSION_H

#include "../src/basic_mlip.h"
#include "../src/basic_trainer.h"


class NonLinearRegression : public AnyTrainer//, protected LogWriting 
{
protected:
	std::vector<Vector3> dLdF;
	std::vector<double> dLdE_i_;
	std::vector<int> type_count_scratch_;
	std::vector<int> active_type_scratch_;
	std::vector<double> type_mean_scratch_;
	std::vector<double> type_joint_scratch_;
	Matrix3 dLdS;
	double metric_energy_abs_sum_ = 0.0;
	double metric_energy_sq_weighted_sum_ = 0.0;
	double metric_force_abs_component_sum_ = 0.0;
	double metric_force_sq_component_sum_ = 0.0;
	double metric_stress_abs_component_sum_ = 0.0;
	double metric_stress_sq_component_sum_ = 0.0;
	long long metric_energy_atom_count_ = 0;
	long long metric_force_component_count_ = 0;
	long long metric_stress_component_count_ = 0;
	bool collect_error_metrics_ = true;
	void AddLoss(const Configuration &orig);
	void AddLoss(const Configuration &orig, const Neighborhoods* neighborhoods);
	void AddLossGrad(const Configuration &orig);
	void AddLossGrad(const Configuration &orig, const Neighborhoods* neighborhoods);
	bool NeedStdTerms() const { return std_scaling != 0.0 || stdd_scaling != 0.0; }
	void PrepareTypeScratch(const Configuration& cfg);
	void ResetObjectiveAccumulators();
	double loss_;											// result of AddLoss and AddLossGrad
	double std_;
        double stdd_;
	double mean_1;
        double mean_2;
		double mean_3;
        std::vector<double> loss_grad_;							// result of AddLossGrad

public:
	int norm_by_forces = 0;									// whether to scale weight of E&F in configurations depending on the abs(F)
        double std_scaling = 0.2;
        double stdd_scaling = 0.00001;	

	NonLinearRegression(AnyLocalMLIP* _p_mlip,				// Constructor requires MTP basis
						double _wgt_energy = 1.0,			// Optional parameters are the weights coeficients of energy, forces and stresses equations in minimization problem
						double _wgt_forces = 1.0,
						double _wgt_stress = 1.0,
						double _wgt_relfrc = 0.0,
						double _wgt_constr = 1.0) :
		AnyTrainer(	_p_mlip, 
					_wgt_energy, 
					_wgt_forces, 
					_wgt_stress,
					_wgt_relfrc,
					_wgt_constr) {};
														
	double ObjectiveFunction(std::vector<Configuration>& train_set);			// Calculates objective function summed over train_set
	double ObjectiveFunction(std::vector<Configuration>& train_set, const std::vector<Neighborhoods>* neighborhoods);
	void CalcObjectiveFunctionGrad(std::vector<Configuration>& train_set);		// Calculates objective function summed over train_set with their gradients
	void CalcObjectiveFunctionGrad(std::vector<Configuration>& train_set, const std::vector<Neighborhoods>* neighborhoods);
	double EnergyMAE_meVPerAtom() const;
	double EnergyRMSE_meVPerAtom() const;
	double ForceMAE_meVPerA() const;
	double ForceRMSE_meVPerA() const;
	double StressMAE_eV() const;
	double StressRMSE_eV() const;
	double EFSLoss(double total_loss, double std_term, double stdd_term) const;
	void SetCollectErrorMetrics(bool enabled) { collect_error_metrics_ = enabled; }
	bool CollectErrorMetrics() const { return collect_error_metrics_; }

	virtual void Train(std::vector<Configuration>& train_set) override;

	bool CheckLossConsistency_debug(Configuration cfg,
									double displacement = 0.001,
									double control_delta = 0.01);

};

#endif // MLIP_NON_LINEAR_REGRESSION_H

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include <random>
#include <ctime>
#include <sstream>


#include <fstream>
#include <iomanip>
#include <set>
#include <iostream>
#include <algorithm>

#include "non_linear_regression.h"
#include "../src/common/bfgs.h"

using namespace std;

namespace {
inline void AccumulateStressErrorMetrics(const Matrix3& lhs,
										 const Matrix3& rhs,
										 double& abs_sum,
										 double& sq_sum)
{
	const double d0 = lhs[0][0] - rhs[0][0];
	const double d1 = lhs[1][1] - rhs[1][1];
	const double d2 = lhs[2][2] - rhs[2][2];
	const double d3 = lhs[1][2] - rhs[1][2];
	const double d4 = lhs[0][2] - rhs[0][2];
	const double d5 = lhs[0][1] - rhs[0][1];

	abs_sum += std::abs(d0) + std::abs(d1) + std::abs(d2)
			+ std::abs(d3) + std::abs(d4) + std::abs(d5);
	sq_sum += d0 * d0 + d1 * d1 + d2 * d2
		   + d3 * d3 + d4 * d4 + d5 * d5;
}
}

void NonLinearRegression::PrepareTypeScratch(const Configuration& cfg)
{
	for (int type : active_type_scratch_) {
		type_count_scratch_[type] = 0;
		type_mean_scratch_[type] = 0.0;
		type_joint_scratch_[type] = 0.0;
	}
	active_type_scratch_.clear();

	int max_type = -1;
	for (int type : cfg.types_)
		max_type = std::max(max_type, type);
	if (max_type < 0)
		return;

	const std::size_t required_size = static_cast<std::size_t>(max_type + 1);
	if (type_count_scratch_.size() < required_size) {
		type_count_scratch_.resize(required_size, 0);
		type_mean_scratch_.resize(required_size, 0.0);
		type_joint_scratch_.resize(required_size, 0.0);
	}

	for (int type : cfg.types_) {
		if (type_count_scratch_[type] == 0)
			active_type_scratch_.push_back(type);
		++type_count_scratch_[type];
	}
}

void NonLinearRegression::ResetObjectiveAccumulators()
{
	loss_ = 0.0;
	std_ = 0.0;
	stdd_ = 0.0;
	mean_1 = 0.0;
	mean_2 = 0.0;
	mean_3 = 0.0;
	metric_energy_abs_sum_ = 0.0;
	metric_energy_sq_weighted_sum_ = 0.0;
	metric_force_abs_component_sum_ = 0.0;
	metric_force_sq_component_sum_ = 0.0;
	metric_stress_abs_component_sum_ = 0.0;
	metric_stress_sq_component_sum_ = 0.0;
	metric_energy_atom_count_ = 0;
	metric_force_component_count_ = 0;
	metric_stress_component_count_ = 0;
}

void NonLinearRegression::AddLoss(const Configuration & orig)
{
	AddLoss(orig, nullptr);
}

void NonLinearRegression::AddLoss(const Configuration & orig, const Neighborhoods* neighborhoods)
{
	if (orig.size() == 0)
		return;

	const bool need_std_terms = NeedStdTerms();
	Configuration cfg = orig;
	if (neighborhoods != nullptr)
		p_mlip->CalcEFS(cfg, *neighborhoods);
	else
		p_mlip->CalcEFS(cfg);

	if (collect_error_metrics_ && orig.has_energy() && cfg.has_energy()) {
		const double energy_delta = orig.energy - cfg.energy;
		metric_energy_abs_sum_ += std::abs(energy_delta);
		metric_energy_sq_weighted_sum_ += energy_delta * energy_delta / orig.size();
		metric_energy_atom_count_ += orig.size();
	}
	if (collect_error_metrics_ && orig.has_forces() && cfg.has_forces()) {
		for (int i = 0; i < cfg.size(); i++) {
			for (int a = 0; a < 3; a++)
			{
				const double force_delta = cfg.force(i)[a] - orig.force(i)[a];
				metric_force_abs_component_sum_ += std::abs(force_delta);
				metric_force_sq_component_sum_ += force_delta * force_delta;
			}
		}
		metric_force_component_count_ += static_cast<long long>(3) * cfg.size();
	}
	if (collect_error_metrics_ && orig.has_stresses() && cfg.has_stresses()) {
		AccumulateStressErrorMetrics(cfg.stresses,
									 orig.stresses,
									 metric_stress_abs_component_sum_,
									 metric_stress_sq_component_sum_);
		metric_stress_component_count_ += 6;
	}
	int fn = norm_by_forces;
	double d = 0.1;
	double avef = 0;
	double _std_ = 0.0;
	double _stdd_ = 0.0;
	if (need_std_terms)
	{
		const double mean_ = orig.energy / orig.size();
		PrepareTypeScratch(cfg);

		std::vector<double>& type_mean = type_mean_scratch_;
		for (int i = 0; i < cfg.size(); i++)
			type_mean[cfg.type(i)] += cfg.cal_se[i] / type_count_scratch_[cfg.type(i)];

		for (int type : active_type_scratch_)
			_stdd_ += (type_mean[type] - mean_) * (type_mean[type] - mean_);

		for (int i = 0; i < cfg.size(); i++)
			_std_ += 20.0 * (cfg.cal_se[i] - type_mean[cfg.type(i)]) * (cfg.cal_se[i] - type_mean[cfg.type(i)])
				/ orig.size() / (20.0 + orig.force(i).NormSq());

		stdd_ += stdd_scaling * _stdd_;
		std_ += std_scaling * _std_;
		loss_ += std_scaling * _std_ + stdd_scaling * _stdd_;
	}
	if (orig.has_forces())
		for (int ind = 0; ind < orig.size(); ind++)
			avef += orig.force(ind).NormSq() / orig.size();

	if (wgt_eqtn_forces != 0 && orig.has_forces())
		for (int i=0; i<cfg.size(); i++)
		{
			double wgt = (wgt_rel_forces<=0.0) ?
						 wgt_eqtn_forces :
						 wgt_eqtn_forces*wgt_rel_forces / (orig.force(i).NormSq() + wgt_rel_forces);

			if (weighting == "structures")
				wgt /= cfg.size();

			loss_ += wgt * (orig.force(i) - cfg.force(i)).NormSq() * d/ (d + fn*avef);
		}

	double wgt_energy = wgt_eqtn_energy;
	double wgt_stress = wgt_eqtn_stress;

	if (weighting == "structures")
	{
		wgt_energy /= cfg.size();
		wgt_stress /= cfg.size();
	}
	else if (weighting == "molecules")
	{
		wgt_energy *= cfg.size();
		wgt_stress *= cfg.size();
	}

	if (wgt_stress!=0 && orig.has_stresses())
		loss_ += wgt_stress * (orig.stresses-cfg.stresses).NormFrobeniusSq() 
				 * (1.0/cfg.size());

	p_mlip->AddPenaltyGrad(wgt_eqtn_constr, loss_);

	// it is important to add energy latest - less round-off errors this way
	if (wgt_energy!=0 && orig.has_energy())
		loss_ += wgt_energy * (orig.energy-cfg.energy)*(orig.energy-cfg.energy) 
				 * d / ((d + fn*avef)*cfg.size());
	
}

void NonLinearRegression::AddLossGrad(const Configuration & orig)
{
	AddLossGrad(orig, nullptr);
}

void NonLinearRegression::AddLossGrad(const Configuration & orig, const Neighborhoods* neighborhoods)
{
	if (orig.size() == 0)
		return;

	if (loss_grad_.size() != p_mlip->CoeffCount())
		loss_grad_.resize(p_mlip->CoeffCount());

	const bool need_std_terms = NeedStdTerms();
	Configuration cfg = orig;
	if (neighborhoods != nullptr)
		p_mlip->CalcEFS(cfg, *neighborhoods);
	else
		p_mlip->CalcEFS(cfg);

	if (collect_error_metrics_ && orig.has_energy() && cfg.has_energy()) {
		const double energy_delta = orig.energy - cfg.energy;
		metric_energy_abs_sum_ += std::abs(energy_delta);
		metric_energy_sq_weighted_sum_ += energy_delta * energy_delta / orig.size();
		metric_energy_atom_count_ += orig.size();
	}
	if (collect_error_metrics_ && orig.has_forces() && cfg.has_forces()) {
		for (int i = 0; i < cfg.size(); i++) {
			for (int a = 0; a < 3; a++)
			{
				const double force_delta = cfg.force(i)[a] - orig.force(i)[a];
				metric_force_abs_component_sum_ += std::abs(force_delta);
				metric_force_sq_component_sum_ += force_delta * force_delta;
			}
		}
		metric_force_component_count_ += static_cast<long long>(3) * cfg.size();
	}
	if (collect_error_metrics_ && orig.has_stresses() && cfg.has_stresses()) {
		AccumulateStressErrorMetrics(cfg.stresses,
									 orig.stresses,
									 metric_stress_abs_component_sum_,
									 metric_stress_sq_component_sum_);
		metric_stress_component_count_ += 6;
	}
	//for (int i = 0; i < p_mlip->CoeffCount(); i++)
	//	cout << p_mlip->Coeff()[i] << " ";
	//cout << endl;

	// the derivatives of loss_ wrt energy, forces and stresses
	double dLdE;
	if (dLdE_i_.size() != cfg.size())
		dLdE_i_.resize(cfg.size());
	FillWithZero(dLdE_i_);
	if (dLdF.size() != cfg.size())
		dLdF.resize(cfg.size());
	int fn = norm_by_forces;
	double d = 0.1;
	double avef = 0;
	double _std_ = 0;
	double _stdd_ = 0;
	double mean_ = 0.0;
	std::vector<double>& type_mean = type_mean_scratch_;
	std::vector<double>& type_joint = type_joint_scratch_;
	if (need_std_terms)
	{
		mean_ = orig.energy / orig.size();
		PrepareTypeScratch(cfg);
		for (int i = 0; i < cfg.size(); i++)
			type_mean[cfg.type(i)] += cfg.cal_se[i] / type_count_scratch_[cfg.type(i)];

		if ((int)type_mean.size() > 0) mean_1 += type_mean[0];
		if ((int)type_mean.size() > 1) mean_2 += type_mean[1];
		if ((int)type_mean.size() > 2) mean_3 += type_mean[2];

		for (int i = 0; i < cfg.size(); i++)
			type_joint[cfg.type(i)] += (cfg.cal_se[i] - type_mean[cfg.type(i)]) * 200.0 / (200.0 + orig.force(i).NormSq())
				/ type_count_scratch_[cfg.type(i)] / orig.size();

		for (int type : active_type_scratch_)
			_stdd_ += (type_mean[type] - mean_) * (type_mean[type] - mean_);

		for (int i = 0; i < cfg.size(); i++)
			_std_ += (cfg.cal_se[i] - type_mean[cfg.type(i)]) * (cfg.cal_se[i] - type_mean[cfg.type(i)])
				* 200.0 / (200.0 + orig.force(i).NormSq()) / orig.size();

		std_ += std_scaling * _std_;
		stdd_ += stdd_scaling * _stdd_;
	}
	


	if (orig.has_forces())
		for (int ind = 0; ind < orig.size(); ind++)
			avef += orig.force(ind).NormSq() / orig.size();

	// it is important to add the energy latest - less round-off errors this way
	if (wgt_eqtn_forces != 0 && orig.has_forces())
		for (int i = 0; i < cfg.size(); i++)
		{
			double wgt = (wgt_rel_forces<=0.0) ?
						 wgt_eqtn_forces :
						 wgt_eqtn_forces*wgt_rel_forces / (orig.force(i).NormSq() + wgt_rel_forces);

			if (weighting == "structures")
				wgt /= cfg.size();

			loss_ += wgt * (cfg.force(i) - orig.force(i)).NormSq() * d / (d + fn*avef);
			dLdF[i] = 2.0 * wgt * (cfg.force(i) - orig.force(i)) * d / (d + fn*avef);
		}
	else
		FillWithZero(dLdF);

	double wgt_energy = wgt_eqtn_energy;
	double wgt_stress = wgt_eqtn_stress;

	if (weighting == "structures")
	{
		wgt_energy /= cfg.size();
		wgt_stress /= cfg.size();
	}
	else if (weighting == "molecules")
	{
		wgt_energy *= cfg.size();
		wgt_stress *= cfg.size();
	}

	if (wgt_stress!=0 && orig.has_stresses())
	{
		loss_ += wgt_stress * (cfg.stresses - orig.stresses).NormFrobeniusSq() / cfg.size();
		dLdS = -2.0 * wgt_stress * (cfg.stresses - orig.stresses) * (1.0/cfg.size());
	}
	else
		dLdS *= 0.0;

	if (wgt_energy!=0 && orig.has_energy())
	{
		loss_ += wgt_energy * (cfg.energy - orig.energy)*(cfg.energy - orig.energy) *d/ ((d + fn*avef)*cfg.size());
		dLdE = 2.0 * wgt_energy * (cfg.energy - orig.energy)*d /((d + fn*avef)*cfg.size());
	}
	else
		dLdE = 0.0;
	for (int i = 0; i < cfg.size(); i++)
	{
		dLdE_i_[i] += dLdE;
			if (need_std_terms)
			{
				dLdE_i_[i] += std_scaling * 2.0 * ((cfg.cal_se[i] - type_mean[cfg.type(i)]) * 200.0 / (200.0 + orig.force(i).NormSq()) / orig.size()
					- type_joint[cfg.type(i)])
					+ stdd_scaling * 2.0 * (type_mean[cfg.type(i)] - mean_) / type_count_scratch_[cfg.type(i)];
			}
		//dLdE_i[i] += dLdE + std_scaling * 2 * ((cfg.cal_se[i] - type_mean[cfg.type(i)])  / orig.size()- 0.0 * type_joint[cfg.type(i)]);
	}
	if (need_std_terms)
		loss_ += std_scaling * _std_ + stdd_scaling * _stdd_;
	p_mlip->AddPenaltyGrad(wgt_eqtn_constr, loss_, &loss_grad_);

	// Now we compute gradients, this adds to loss_grad_
	if (neighborhoods != nullptr)
		p_mlip->AccumulateEFSCombinationGrad(cfg, dLdE_i_, dLdF, dLdS, loss_grad_, *neighborhoods);
	else
		p_mlip->AccumulateEFSCombinationGrad(cfg, dLdE_i_, dLdF, dLdS, loss_grad_);
}


// Calculates objective function summed over train_set
double NonLinearRegression::ObjectiveFunction(vector<Configuration>& training_set)
{
	return ObjectiveFunction(training_set, nullptr);
}

double NonLinearRegression::ObjectiveFunction(vector<Configuration>& training_set, const std::vector<Neighborhoods>* neighborhoods)
{
	ResetObjectiveAccumulators();
	for (size_t i = 0; i < training_set.size(); ++i)
		AddLoss(training_set[i], neighborhoods == nullptr ? nullptr : &(*neighborhoods)[i]);
	return loss_;
}

// Calculates objective function summed over train_set with their gradients
void NonLinearRegression::CalcObjectiveFunctionGrad(vector<Configuration>& training_set)
{
	CalcObjectiveFunctionGrad(training_set, nullptr);
}

void NonLinearRegression::CalcObjectiveFunctionGrad(vector<Configuration>& training_set, const std::vector<Neighborhoods>* neighborhoods)
{
	ResetObjectiveAccumulators();
	loss_grad_.resize(p_mlip->CoeffCount());
	FillWithZero(loss_grad_);

	for (size_t i = 0; i < training_set.size(); ++i) 
		AddLossGrad(training_set[i], neighborhoods == nullptr ? nullptr : &(*neighborhoods)[i]);
}

double NonLinearRegression::EnergyMAE_meVPerAtom() const
{
	if (metric_energy_atom_count_ == 0) return 0.0;
	return 1000.0 * metric_energy_abs_sum_ / static_cast<double>(metric_energy_atom_count_);
}

double NonLinearRegression::EnergyRMSE_meVPerAtom() const
{
	if (metric_energy_atom_count_ == 0) return 0.0;
	return 1000.0 * std::sqrt(metric_energy_sq_weighted_sum_ / static_cast<double>(metric_energy_atom_count_));
}

double NonLinearRegression::ForceMAE_meVPerA() const
{
	if (metric_force_component_count_ == 0) return 0.0;
	return 1000.0 * metric_force_abs_component_sum_ / static_cast<double>(metric_force_component_count_);
}

double NonLinearRegression::ForceRMSE_meVPerA() const
{
	if (metric_force_component_count_ == 0) return 0.0;
	return 1000.0 * std::sqrt(metric_force_sq_component_sum_ / static_cast<double>(metric_force_component_count_));
}

double NonLinearRegression::StressMAE_eV() const
{
	if (metric_stress_component_count_ == 0) return 0.0;
	return metric_stress_abs_component_sum_ / static_cast<double>(metric_stress_component_count_);
}

double NonLinearRegression::StressRMSE_eV() const
{
	if (metric_stress_component_count_ == 0) return 0.0;
	return std::sqrt(metric_stress_sq_component_sum_ / static_cast<double>(metric_stress_component_count_));
}

double NonLinearRegression::EFSLoss(double total_loss, double std_term, double stdd_term) const
{
	return total_loss - std_term - stdd_term;
}

#ifdef MLIP_MPI
void NonLinearRegression::Train(std::vector<Configuration>& train_set)
{
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (mpi_rank == 0)
	{
    std::stringstream logstrm1;
		logstrm1	<< "\tTrainer(default): Training over "
			<< train_set.size() << " configurations" << endl;
    MLP_LOG("dev",logstrm1.str());
	}

	int size = p_mlip->CoeffCount();
	double *x = p_mlip->Coeff();

        int m = (int)train_set.size(); // train set size on the current core
        int K = m;                     // train set size over all cores

        MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        BFGS bfgs;
        double bfgs_f;
        Array1D bfgs_g(size);

        bfgs.Set_x(x, size);

        for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
			if (i == j)
                                bfgs.inv_hess(i,j) = 1;
                        else
                                bfgs.inv_hess(i,j) = 0;

        int max_step_count = 500;
        double linstop = 1e-6;
        int num_step = 0;
        double linf = 9e99;
        bool converge = false;
        bool linesearch = false;

        while (!converge)
        {
                if (!linesearch)
                {
					//if (mpi_rank == 0) cout << num_step << endl;
                        if (mpi_rank == 0)
                                bfgs.Set_x(x, size);

                        MPI_Bcast(&x[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                        //if (mpi_rank == 0) {
                        //        p_mlip->Save("current.mlip");
                        //}

                }

                for (int i = 0; i < size; i++)
                        x[i] = bfgs.x(i);

                MPI_Bcast(&x[0], size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                CalcObjectiveFunctionGrad(train_set);
                loss_ /= K;
                for (int i = 0; i < size; i++)
                        loss_grad_[i] /= K;

		MPI_Barrier(MPI_COMM_WORLD);
                MPI_Reduce(&loss_, &bfgs_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                MPI_Reduce(&loss_grad_[0], &bfgs_g[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

                if (mpi_rank == 0) {
                        if (!converge) {
                                bfgs.Iterate(bfgs_f,bfgs_g);
				//if (!linesearch) {
		            //            cout << bfgs_f << endl;
					//for (int i = 0; i < size; i++)
					//	cout << bfgs.x(i) << " ";
					//cout << endl;
					//for (int i = 0; i < size; i++)
					//	cout << bfgs_g[i] << " ";
					//cout << endl;
				//}

                                while (abs(bfgs.x(0) - x[0]) > 0.5) 
                                        bfgs.ReduceStep(0.25);
                        }
                }

                linesearch = bfgs.is_in_linesearch();

                if (!linesearch) {
                        if (mpi_rank == 0) {

                                num_step++;
                                //cout << num_step << endl;

                                if (num_step % 40 == 0)
                                {
                                        if ((linf - bfgs_f) / bfgs_f < linstop)
                                        {
                                                converge = true;
                                        }

                                        //cout << (linf - bfgs_f) << " " << linf << " " << bfgs_f << endl;

                                        linf = bfgs_f;
                                }

                                if (num_step >= max_step_count || bfgs_f < 1E-15)
                                {
                                        converge = true;
                                }
                        }
                }

		MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast(&converge, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                MPI_Bcast(&linesearch, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
                MPI_Bcast(&num_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }

	if (mpi_rank == 0)
	{
		if (!converge)
			Warning("Convergence was not achieved while training");
		else // Ok
    {
      std::stringstream logstrm1;
			logstrm1 << "\tTrainer(default): Training# "
					<< ++train_cntr << " complete" << endl;
      MLP_LOG("dev",logstrm1.str());
    }
	}
}
#else
void NonLinearRegression::Train(std::vector<Configuration>& train_set)
{
	int size = p_mlip->CoeffCount();
	double *x = p_mlip->Coeff();

        BFGS bfgs;
        double bfgs_f;
        Array1D bfgs_g(size);

        bfgs.Set_x(x, size);

        for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
			if (i == j)
                                bfgs.inv_hess(i,j) = 1;
                        else
                                bfgs.inv_hess(i,j) = 0;

        int max_step_count = 500;
        double linstop = 1e-6;
        int num_step = 0;
        double linf = 9e99;
        bool converge = false;
        bool linesearch = false;

        while (!converge)
        {
                if (!linesearch)
                {
                        bfgs.Set_x(x, size);

                        //p_mlip->Save("current.mlip");

                }

                for (int i = 0; i < size; i++)
                        x[i] = bfgs.x(i);

                CalcObjectiveFunctionGrad(train_set);
                loss_ /= train_set.size();
                for (int i = 0; i < size; i++)
                        loss_grad_[i] /= train_set.size();

		bfgs_f = loss_;
		memcpy(&bfgs_g[0], &loss_grad_[0], p_mlip->CoeffCount()*sizeof(double));		

                if (!converge) {
                        bfgs.Iterate(bfgs_f,bfgs_g);
			//if (!linesearch) {
	                        //cout << bfgs_f << endl;
				//for (int i = 0; i < size; i++)
				//	cout << bfgs.x(i) << " ";
				//cout << endl;
				//for (int i = 0; i < size; i++)
				//	cout << bfgs_g[i] << " ";
				//cout << endl;
			//}

                        while (abs(bfgs.x(0) - x[0]) > 0.5) 
                                bfgs.ReduceStep(0.25);
                }

                linesearch = bfgs.is_in_linesearch();

                if (!linesearch) {

                       num_step++;
                        //cout << num_step << endl;

                        if (num_step % 100 == 0)
                        {
                                if ((linf - bfgs_f) / bfgs_f < linstop)
                                {
                                        converge = true;
                                }

                                //cout << (linf - bfgs_f) << " " << linf << " " << bfgs_f << endl;

                                linf = bfgs_f;
                        }

                        if (num_step >= max_step_count || bfgs_f < 1E-15)
                        {
                                converge = true;
                        }

                }

        }

	if (!converge)
		Warning("Convergence was not achieved while training");
	else // Ok
  {
    std::stringstream logstrm1;
		logstrm1 << "\tTrainer(default): Training# "
			<< ++train_cntr << " complete" << endl;
    MLP_LOG("dev",logstrm1.str());
  }
}
#endif

bool NonLinearRegression::CheckLossConsistency_debug(Configuration cfg, double displacement, double control_delta)
{
	default_random_engine generator(777);
	uniform_real_distribution<double> distribution(0.0, 1.0);
	double delta_c = displacement;
	double dloss_actual;
	double rel_err_max = 0;
	int n = p_mlip->CoeffCount();
	vector<double> dloss_predict(n);

	vector<Configuration> train_set(1, cfg);

	//for (int i=0; i<n; i++)
		//p_mlip->Coeff()[i] = distribution(generator);

	CalcObjectiveFunctionGrad(train_set);

	//double loss0 = loss_;
	double lossp, lossm;

	for (int i=0; i<n; i++)
		dloss_predict[i] = loss_grad_[i];

	for (int i=0; i<n; i++)
	{
		p_mlip->Coeff()[i] += delta_c;
		//cout << p_mlip->Coeff()[i] << " ";
		lossp = ObjectiveFunction(train_set);
		p_mlip->Coeff()[i] -= 2*delta_c;
		//cout << p_mlip->Coeff()[i] << endl;
		lossm = ObjectiveFunction(train_set);
		p_mlip->Coeff()[i] += delta_c;
		dloss_actual = (lossp - lossm) / (2*displacement);

		//cout << i << " " << lossp << " " << lossm << endl;
		//cout << loss_grad_[i] << " " << dloss_actual << " " << fabs(dloss_actual-loss_grad_[i]) << endl; 

		if (abs((dloss_actual - dloss_predict[i]) / dloss_actual)>rel_err_max)
			rel_err_max = abs((dloss_actual - dloss_predict[i]) / dloss_actual);
	}

	return (rel_err_max < control_delta);
}

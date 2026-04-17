/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */


#include "basic_mlip.h"


void AnyLocalMLIP::Save(const std::string& filename)		//!< Saves MLIP's data to file
{
	Warning("AnyLocalMLIP: Save method is not implemented for this MLIP");
}

void AnyLocalMLIP::Load(const std::string& filename)		//!< Loads MLIP from file 
{
	Warning("AnyLocalMLIP: Load method is not implemented for this MLIP");
}

double AnyLocalMLIP::SiteEnergy(const Neighborhood & nbh)	//!<
{
	PrepareEvalCaches();
	CalcSiteEnergyDers(nbh);
	return buff_site_energy_;
}

// Must update buff_site_energy_ and buff_site_energy_ders_
void AnyLocalMLIP::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	buff_site_energy_ = 0;
	FillWithZero(buff_site_energy_ders_);
	AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0.0, nullptr);
}

//! Must calculate gradient of site energy w.r.t. coefficients and add it to second argument array
void AnyLocalMLIP::AddSiteEnergyGrad(const Neighborhood& nbh,
									 std::vector<double>& out_se_grad_accumulator)
{
	PrepareEvalCaches();
	AccumulateCombinationGrad(nbh, out_se_grad_accumulator, 1.0, nullptr);
}

// Calculate gradient of site energy w.r.t. coefficients and write them in second argument
void AnyLocalMLIP::CalcSiteEnergyGrad(const Neighborhood& nbh, std::vector<double>& out_se_grad)
{
	out_se_grad.resize(CoeffCount());
	FillWithZero(out_se_grad);
	AddSiteEnergyGrad(nbh, out_se_grad);
}

// Calculate gradient of site energy w.r.t. coefficients and write them in second argument
/*void AnyLocalMLIP::CalcForceGrad(const Configuration & cfg, const int ind, std::vector<Vector3>& out_frc_grad)
{
	out_frc_grad.resize(CoeffCount());
	FillWithZero(out_frc_grad);

	// calculating EFS components
	Neighborhoods nbhs(cfg, CutOff());

	int dir_size = 0;
	for (int i = 0; i < cfg.size(); i++)
		dir_size = std::max(dir_size, nbhs[i].count);
	std::vector<Vector3> dir(dir_size);

	for (int i = 0; i < cfg.size(); i++) {
		const Neighborhood& nbh = nbhs[i];

		for (int j = 0; j < nbh.count; j++) {
			if (i==ind)
			{
				for (int a=0; a<3; a++)
				{
					FillWithZero(dir);
					dir[ind][a] = 1;
					FillWithZero(tmp_grad_accumulator_);
					AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
					for (int k=0; k<CoeffCount(); k++)
						out_frc_grad[ind][a] += tmp_grad_accumulator_[k];
				}
			}
			else if (nbh.inds[j]==ind)
			{
				for (int a=0; a<3; a++)
				{
					FillWithZero(dir);
					dir[ind][a] = 1;
					FillWithZero(tmp_grad_accumulator_);
					AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
					for (int k=0; k<CoeffCount(); k++)
						out_frc_grad[ind][a] -= tmp_grad_accumulator_[k];
				}
			}
		}
	}
}*/

void AnyLocalMLIP::CalcStressesGrads(const Configuration & cfg, Array3D & out_str_grad)
{
	out_str_grad.resize(3, 3, CoeffCount());
	out_str_grad.set(0);

	// calculating EFS components
	Neighborhoods nbhs(cfg, CutOff());

	int dir_size = 0;
	for (int i = 0; i < cfg.size(); i++)
		dir_size = std::max(dir_size, nbhs[i].count);
	std::vector<Vector3> dir(dir_size);
	PrepareEvalCaches();

	for (int i = 0; i < cfg.size(); i++)
	{
		const Neighborhood& nbh = nbhs[i];

		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
				{
					FillWithZero(dir);
					dir[j][a] = 1;
					FillWithZero(tmp_grad_accumulator_);
					AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
					for (int k=0; k<CoeffCount(); k++)
						out_str_grad(a, b, k) -= buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
				}
	}
}

void AnyLocalMLIP::CalcEFSGrads(const Configuration & cfg, 
								Array1D & out_ene_grad, 
								Array3D & out_frc_grad, 
								Array3D & out_str_grad)
{
	out_ene_grad.resize(CoeffCount());
	FillWithZero(out_ene_grad);
	out_frc_grad.resize(cfg.size(), 3, CoeffCount());
	out_frc_grad.set(0);
	out_str_grad.resize(3, 3, CoeffCount());
	out_str_grad.set(0);

	// calculating EFS components
	Neighborhoods nbhs(cfg, CutOff());

	int dir_size = 0;
	for (int i = 0; i < cfg.size(); i++)
		dir_size = std::max(dir_size, nbhs[i].count);
	std::vector<Vector3> dir(dir_size);
	PrepareEvalCaches();

	for (int i = 0; i < cfg.size(); i++)
	{
		const Neighborhood& nbh = nbhs[i];

		for (int j=0; j<nbh.count; j++)
		{
			AccumulateCombinationGrad(nbh, out_ene_grad, 1, nullptr);

			for (int a=0; a<3; a++)
			{
				FillWithZero(dir);
				dir[j][a] = 1;
				FillWithZero(tmp_grad_accumulator_);
				AccumulateCombinationGrad(nbh, tmp_grad_accumulator_, 0, &dir[0]);
				for (int k=0; k<CoeffCount(); k++)
				{
					out_frc_grad(i, a, k) += tmp_grad_accumulator_[k];
					out_frc_grad(nbh.inds[j], a, k) -= tmp_grad_accumulator_[k];
					for (int b = 0; b < 3; b++)
						out_str_grad(a, b, k) -= buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
				}
			}
		}
	}
}

//! Function implementing soft constraints on coefficients (e.g., so that the norm of the radial function is 1) or regularization implementing as penalty to training procedure
void AnyLocalMLIP::AddPenaltyGrad(	const double coeff, 
									double & out_penalty_accumulator, 
									Array1D * out_penalty_grad_accumulator) 
{
}

AnyLocalMLIP::AnyLocalMLIP(double _min_dist, double _max_dist, int _radial_basis_size)
{
	p_RadialBasis = new RadialBasis_Shapeev(_min_dist, _max_dist, _radial_basis_size);
	//p_RadialBasis = new RadialBasis_Chebyshev(_min_dist, _max_dist, _radial_basis_size);
	rb_owner = true;
}

AnyLocalMLIP::AnyLocalMLIP(AnyRadialBasis* p_radial_basis)
{
	p_RadialBasis = p_radial_basis;
	rb_owner = false;
}

AnyLocalMLIP::~AnyLocalMLIP()
{
	if (rb_owner && p_RadialBasis != nullptr)
		delete p_RadialBasis;
}

void AnyLocalMLIP::CalcE(Configuration& cfg)
{
	cfg.energy = 0;

	// calculating EFS components
	Neighborhoods neighborhoods(cfg, CutOff());

	for (const Neighborhood& nbh : neighborhoods)
		cfg.energy += SiteEnergy(nbh);

	cfg.has_energy(true);
}

void AnyLocalMLIP::CalcEFS(Configuration& cfg)
{
	Neighborhoods neighborhoods(cfg, CutOff());
	CalcEFS(cfg, neighborhoods);
}

void AnyLocalMLIP::CalcEFS(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	ResetEFS(cfg);
	PrepareEvalCaches();

	cfg.has_energy(true);
	cfg.has_forces(true);
	cfg.has_stresses(true);
	cfg.has_site_energies(true);
	cfg.cal_se.resize(cfg.size());
	cfg.cal_se0.resize(cfg.size());
	cfg.type_mean.resize(cfg.unique_elems.size());
	FillWithZero(cfg.type_mean);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		buff_site_energy_0 = 0;
		buff_site_energy_ = 0;
		buff_site_energy_ders_.resize(nbh.count);
		CalcSiteEnergyDers(nbh);
		cfg.cal_se[ind] = buff_site_energy_;
		cfg.cal_se0[ind]= buff_site_energy_0;
		//cfg.type_mean[cfg.type(ind)] += buff_site_energy_ / cfg.type_num[cfg.type(ind)];
		cfg.energy += buff_site_energy_;

		for (int j = 0; j < nbh.count; j++) {
			cfg.force(ind) += buff_site_energy_ders_[j];
			cfg.force(nbh.inds[j]) -= buff_site_energy_ders_[j];
		}

		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					cfg.stresses[a][b] -= buff_site_energy_ders_[j][a] * nbh.vecs[j][b];
	}
}

// Calculate gradient of energy w.r.t. coefficients and write them in second argument
void AnyLocalMLIP::CalcEnergyGrad(Configuration & cfg, std::vector<double>& out_energy_grad)
{
	out_energy_grad.resize(CoeffCount());
	FillWithZero(out_energy_grad);

	Neighborhoods neighborhoods(cfg, CutOff());
	PrepareEvalCaches();
	for (const Neighborhood& nbh : neighborhoods)
		AddSiteEnergyGrad(nbh, out_energy_grad);
}

void AnyLocalMLIP::AccumulateEFSCombinationGrad(Configuration &cfg,
												std::vector<double>& ene_weight,
												const std::vector<Vector3>& frc_weights,
												const Matrix3& str_weights,
												std::vector<double>& out_grads_accumulator)
{
	out_grads_accumulator.resize(CoeffCount());

	Neighborhoods neighborhoods(cfg, CutOff());
	AccumulateEFSCombinationGrad(cfg, ene_weight, frc_weights, str_weights,
								 out_grads_accumulator, neighborhoods);
}

void AnyLocalMLIP::AccumulateEFSCombinationGrad(Configuration &cfg,
												std::vector<double>& ene_weight,
												const std::vector<Vector3>& frc_weights,
												const Matrix3& str_weights,
												std::vector<double>& out_grads_accumulator,
												const Neighborhoods& neighborhoods)
{
	out_grads_accumulator.resize(CoeffCount());
	PrepareEvalCaches();

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		tmp_se_ders_weights_.resize(nbh.count);
		FillWithZero(tmp_se_ders_weights_);
		double ene_weight_ = ene_weight[ind];
		for (int j = 0; j < nbh.count; j++) {
			tmp_se_ders_weights_[j] += frc_weights[ind];
			tmp_se_ders_weights_[j] -= frc_weights[nbh.inds[j]];
		}
		
		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					tmp_se_ders_weights_[j][a] += str_weights[a][b] * nbh.vecs[j][b];

		//AccumulateCombinationGrad(	nbh, 
		//							out_grads_accumulator, 
		//							buff_site_energy_, 
		//							buff_site_energy_ders_, 
		//							ene_weight, 
		//							&tmp_se_ders_weights_[0]);
		bool grad_zero = true;
		for (int i=0; i<3*tmp_se_ders_weights_.size(); i++)
			if (tmp_se_ders_weights_[i/3][i%3] != 0.0)
			{
				grad_zero = false;
				break;
			}

		if (grad_zero)
		{
			AccumulateCombinationGrad(  nbh,
										out_grads_accumulator,
										ene_weight_,
										nullptr);
		}
		else
		{
			AccumulateCombinationGrad(	nbh, 
										out_grads_accumulator, 
										ene_weight_, 
										&tmp_se_ders_weights_[0]);
		}
	}
}

void AnyLocalMLIP::PrepareEvalCaches()
{
}

//#pragma warning(pop)

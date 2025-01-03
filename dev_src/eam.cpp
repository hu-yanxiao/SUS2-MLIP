/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Ivan Novikov
 */

#include "eam.h"

void EAMMult::AddPenaltyGrad(	const double coeff,
								double& out_penalty_accumulator,
								Array1D* out_penalty_grad_accumulator)
{
	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	const double* density_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_ * radial_basis_count_];

	double sum_coeffs2 = 0;
	for (int i = 0; i < atom_types_count_ * atom_types_count_ * radial_basis_count_; i++) 
		sum_coeffs2 += density_coeffs[i]*density_coeffs[i];

	out_penalty_accumulator += coeff * (sum_coeffs2-1) * (sum_coeffs2-1);

	if (out_penalty_grad_accumulator != nullptr)
		for (int i = 0; i < atom_types_count_ * atom_types_count_ * radial_basis_count_; i++)
			(*out_penalty_grad_accumulator)[i + atom_types_count_ * atom_types_count_ * radial_basis_count_] += coeff * 4 * (sum_coeffs2 - 1) * density_coeffs[i];
}

void EAMMult::AddSiteEnergyGrad(const Neighborhood& nbh, std::vector<double>& site_energy_grad)
{
//	const double* pair_potentials_coeffs = &regression_coeffs[0];
	const double* density_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_ * radial_basis_count_];
	const double* embedding_coeffs = &regression_coeffs[2 * atom_types_count_ * atom_types_count_ * radial_basis_count_];

	double density = 0;

	for (int j = 0; j < nbh.count; j++) {
//		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(dist_j);

		// pair potential part
		for (int i = 0; i < radial_basis_count_; i++) {
			int pair_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			double val = p_RadialBasis->rb_vals[i];
			site_energy_grad[pair_idx] += 0.5 * val;
		}

		// density calculation
		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			const double val = p_RadialBasis->rb_vals[i];
			density += val * density_coeffs[dens_idx];
		}
	}

	// embedding function and its derivative WRT density
	double density_pow = density; // will be density^m
	double embedding = embedding_coeffs[nbh.my_type * embedding_parameters_count_];
	double embedding_der = 0;
	site_energy_grad[2*radial_basis_count_ * atom_types_count_ * atom_types_count_ + nbh.my_type * embedding_parameters_count_] += 1;
	for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
		const int embd_idx = m + nbh.my_type * embedding_parameters_count_;
		embedding_der += m * embedding_coeffs[embd_idx - 1] * density_pow;
		density_pow *= density;
		embedding += embedding_coeffs[embd_idx - 1] * density_pow;
		site_energy_grad[embd_idx - 1 + 2 * atom_types_count_ * atom_types_count_ * radial_basis_count_] += density_pow;
	}

	for (int j = 0; j < nbh.count; j++) {
//		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(dist_j);

		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_
					+ atom_types_count_ * atom_types_count_ * radial_basis_count_;
			const double val = p_RadialBasis->rb_vals[i];
			site_energy_grad[dens_idx] += embedding_der * val;
		}
	}
}

void EAMMult::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	buff_site_energy_ = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	const double* pair_potentials_coeffs = &regression_coeffs[0];
	const double* density_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_ * radial_basis_count_];
	const double* embedding_coeffs = &regression_coeffs[2 * atom_types_count_ * atom_types_count_ * radial_basis_count_];

	Array2D rb_der(nbh.count, CoeffCount());

	double density = 0;

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(nbh.dists[j]);

		// pair potential part
		for (int i = 0; i < radial_basis_count_; i++) {
			const int pair_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			const double val = p_RadialBasis->rb_vals[i];
			const double der = p_RadialBasis->rb_ders[i];
			rb_der(j, i) = der;
			buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[pair_idx];
			buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[pair_idx]);
		}

		// density calculation
		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			const double val = p_RadialBasis->rb_vals[i];
			density += val * density_coeffs[dens_idx];
		}
	}

	// embedding function and its derivative WRT density
	double density_pow = density; // will be density^m
	double embedding = embedding_coeffs[nbh.my_type * embedding_parameters_count_];
	double embedding_der = 0;
	for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
		const int embd_idx = m + nbh.my_type * embedding_parameters_count_;
		embedding_der += m * embedding_coeffs[embd_idx - 1] * density_pow;
		density_pow *= density;
		embedding += embedding_coeffs[embd_idx - 1] * density_pow;
	}

	buff_site_energy_ += embedding;

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		//p_RadialBasis->RB_Calc(nbh.dists[j]);
		// now we can compute the derivative of the embedding WRT atom positions
		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
					+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			//const double der = p_RadialBasis->rb_ders[i];
			buff_site_energy_ders_[j] += vec_j * ((rb_der(j, i) / dist_j) * density_coeffs[dens_idx]) * embedding_der;
		}
	}	
}

void EAMMult::AccumulateCombinationGrad(const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double se_weight,
		const Vector3* se_ders_weights)
{
	out_grad_accumulator.resize(CoeffCount());
	buff_site_energy_ders_.resize(CoeffCount());

	FillWithZero(buff_site_energy_ders_);
	buff_site_energy_ = 0.0;

	const double* pair_potentials_coeffs = &regression_coeffs[0];
	double* density_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_ * radial_basis_count_];
	const double* embedding_coeffs = &regression_coeffs[2 * atom_types_count_ * atom_types_count_ * radial_basis_count_];

	Array2D rb_val(nbh.count, CoeffCount());
	Array2D rb_der(nbh.count, CoeffCount());

	double density = 0;

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(nbh.dists[j]);

		// pair potential part
		for (int i = 0; i < radial_basis_count_; i++) {
			const int pair_idx = i + nbh.types[j] * radial_basis_count_
				+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			const double val = p_RadialBasis->rb_vals[i];
			const double der = p_RadialBasis->rb_ders[i];
			rb_val(j, i) = val;
			rb_der(j, i) = der;
			buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[pair_idx];
			buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[pair_idx]);
		}

		// density calculation
		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
				+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			const double val = p_RadialBasis->rb_vals[i];
			density += val * density_coeffs[dens_idx];
		}
	}

	// embedding function and its derivative WRT density
	double density_pow = density; // will be density^m
	double embedding = embedding_coeffs[nbh.my_type * embedding_parameters_count_];
	double embedding_der = 0;
	double embedding_der2 = 0;
	out_grad_accumulator[2*radial_basis_count_ * atom_types_count_ * atom_types_count_ + nbh.my_type * embedding_parameters_count_] += se_weight * 1;
	for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
		const int embd_idx = m + nbh.my_type * embedding_parameters_count_;
		embedding_der += m * embedding_coeffs[embd_idx - 1] * density_pow;
		density_pow *= density;
		embedding += embedding_coeffs[embd_idx - 1] * density_pow;
		out_grad_accumulator[embd_idx - 1 + 2 * atom_types_count_ * atom_types_count_ * radial_basis_count_] += se_weight * density_pow;
	}
	density_pow = 1; // will be density^m
	for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
		const int embd_idx = m + nbh.my_type * embedding_parameters_count_;
		embedding_der2 += m * (m - 1) * embedding_coeffs[embd_idx - 1] * density_pow;
		density_pow *= density;
	}

	buff_site_energy_ += embedding;

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];

		// calculates vals and ders for j-th atom in the neighborhood
		//p_RadialBasis->RB_Calc(nbh.dists[j]);
		// now we can compute the derivative of the embedding WRT atom positions
		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
				+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			//const double der = p_RadialBasis->rb_ders[i];
			buff_site_energy_ders_[j] += vec_j * ((rb_der(j, i) / dist_j) * density_coeffs[dens_idx]) * embedding_der;
		}
	}

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];
		double vec_j_se_ders_inner; 
		if (se_ders_weights != nullptr) 
			vec_j_se_ders_inner = vec_j.dot(se_ders_weights[j]);
		else
			vec_j_se_ders_inner = 0;

		// calculates vals and ders for j-th atom in the neighborhood
		//p_RadialBasis->RB_Calc(nbh.dists[j]);

		double total_density_coeff_der = 0;
		for (int i = 0; i < radial_basis_count_; i++) {
			const int pair_idx = i + nbh.types[j] * radial_basis_count_
				+ nbh.my_type * radial_basis_count_ * atom_types_count_;
			//const double val = p_RadialBasis->rb_vals[i];
			//const double der = p_RadialBasis->rb_ders[i];
			out_grad_accumulator[pair_idx] += 0.5 * se_weight * rb_val(j, i);
			if(se_ders_weights != nullptr)
				out_grad_accumulator[pair_idx] += 0.5 * vec_j_se_ders_inner * (rb_der(j, i) / dist_j);
			total_density_coeff_der += density_coeffs[pair_idx] * rb_der(j, i);
		}

		for (int i = 0; i < radial_basis_count_; i++) {
			const int dens_idx = i + nbh.types[j] * radial_basis_count_
				+ nbh.my_type * radial_basis_count_ * atom_types_count_
				+ atom_types_count_ * atom_types_count_ * radial_basis_count_;
			//const double val = p_RadialBasis->rb_vals[i];
			//const double der = p_RadialBasis->rb_ders[i];
			out_grad_accumulator[dens_idx] += se_weight * embedding_der * rb_val(j, i);
			if(se_ders_weights != nullptr) {
				out_grad_accumulator[dens_idx] += vec_j_se_ders_inner * embedding_der * (rb_der(j, i) / dist_j);
			}
		}

		if(se_ders_weights != nullptr) {
			for (int jj = 0; jj < nbh.count; jj++) {
				//p_RadialBasis->RB_Calc(nbh.dists[jj]);
				for (int i = 0; i < radial_basis_count_; i++) {
					const int dens_idx = i + nbh.types[jj] * radial_basis_count_
						+ nbh.my_type * radial_basis_count_ * atom_types_count_
						+ atom_types_count_ * atom_types_count_ * radial_basis_count_;
					//const double val = p_RadialBasis->rb_vals[i];
					out_grad_accumulator[dens_idx] += vec_j_se_ders_inner * embedding_der2 * rb_val(jj, i) * total_density_coeff_der / dist_j;
				}
			}
		}

		density_pow = density; // will be density^m
		for (int i = 2; i < embedding_parameters_count_ + 1; i++) {
			const int embd_idx = i + nbh.my_type * embedding_parameters_count_;
			if(se_ders_weights != nullptr)
				out_grad_accumulator[embd_idx - 1 + 2 * radial_basis_count_* atom_types_count_ * atom_types_count_] += vec_j_se_ders_inner
					* i * density_pow * total_density_coeff_der / dist_j;
			density_pow *= density;
		}
	}
}

EAMMult::~EAMMult()
{
}

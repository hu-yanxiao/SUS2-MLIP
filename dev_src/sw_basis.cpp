/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "sw_basis.h"

using namespace std;

void StillingerWeberRadialBasis::AddPenaltyGrad(const double weigh, double& loss, Array1D* loss_grad)
{
	int n_at_types2 = n_at_types*n_at_types;
	int n_at_types3 = n_at_types2*n_at_types;
	int rad_size_n_types2 = p_RadialBasis->rb_size*n_at_types2;

	const double * tripple_lambda = &regression_coeffs[2*rad_size_n_types2+n_at_types3];

	double sum_coeffs2 = 0;
	for (int i = 0; i < n_at_types3; i++) 
		sum_coeffs2 += tripple_lambda[i]*tripple_lambda[i];

	loss += weigh * (sum_coeffs2-1) * (sum_coeffs2-1);

	if (loss_grad != nullptr)
		for (int i = 0; i < n_at_types3; i++) 
			(*loss_grad)[i+2*rad_size_n_types2+n_at_types3] += weigh * 4 * (sum_coeffs2 - 1) * 
										tripple_lambda[i];
}

void StillingerWeberRadialBasis::AccumulateEnergy2body(Configuration& cfg, const Neighborhood& nbh) {
	const double * pair_potential_coeff = &regression_coeffs[0];

	for (int j = 0; j < nbh.count; j++) {

		double dist_j = nbh.dists[j];	
		p_RadialBasis->RB_Calc(dist_j);

		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			const int idx = alpha + nbh.types[j] * p_RadialBasis->rb_size
					+ nbh.my_type * p_RadialBasis->rb_size * n_at_types;
			const double val = p_RadialBasis->rb_vals[alpha];
			cfg.energy += 0.5 * val * pair_potential_coeff[idx];	
		}

	}
}

void StillingerWeberRadialBasis::AccumulateEFS2body(Configuration& cfg, const Neighborhood& nbh, const int my_ind) {
	const double* pair_potential_coeff = &regression_coeffs[0];

	for (int j = 0; j < nbh.count; j++) {

		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;
		p_RadialBasis->RB_Calc(dist_j);
		double en_der_wrt_dist_j = 0;

		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			const int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
			const double val = p_RadialBasis->rb_vals[alpha];
			const double der = p_RadialBasis->rb_ders[alpha];
			cfg.energy += 0.5 * val * pair_potential_coeff[idx_curr_j];
			en_der_wrt_dist_j += 0.5 * der * pair_potential_coeff[idx_curr_j];
		}

		double force[3];
		for (int a = 0; a < 3; a++) {
			force[a] = en_der_wrt_dist_j * vec_j[a] / dist_j;
			cfg.force(my_ind)[a] += force[a];
			cfg.force(nbh.inds[j])[a] -= force[a];
		}

		for (int a = 0; a < 3; a++) {
			for (int b = 0; b < 3; b++) {
				cfg.stresses[a][b] -= vec_j[a]*force[b];
			}
		}

	}
}

void StillingerWeberRadialBasis::AccumulateEnergyGrad2body(const Neighborhood& nbh, 
Array1D& energy_grad) {
	const double* pair_potential_coeff = &regression_coeffs[0];

	for (int j = 0; j < nbh.count; j++) {
		const double dist_j = nbh.dists[j];
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;
		p_RadialBasis->RB_Calc(dist_j);
		double en_der_wrt_dist_j = 0;

		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			const int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
			const double val = p_RadialBasis->rb_vals[alpha];
			energy_grad[idx_curr_j] += 0.5 * val;
		}
		
	}
}

void StillingerWeberRadialBasis::AccumulateEFSGrad2body(Configuration& cfg, const Neighborhood& nbh, 
const int my_ind, Array1D& energy_grad, Array3D& forces_grad, Array3D& stress_grad) {
	const double* pair_potential_coeff = &regression_coeffs[0];

	FillWithZero(en_der_wrt_dist_j_grad);

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		const double dist_j = nbh.dists[j];
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;
		p_RadialBasis->RB_Calc(dist_j);
		double en_der_wrt_dist_j = 0;

		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			const int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
			const double val = p_RadialBasis->rb_vals[alpha];
			const double der = p_RadialBasis->rb_ders[alpha];
			cfg.energy += 0.5 * val * pair_potential_coeff[idx_curr_j];
			en_der_wrt_dist_j += 0.5 * der * pair_potential_coeff[idx_curr_j];
			energy_grad[idx_curr_j] += 0.5 * val;
			en_der_wrt_dist_j_grad[idx_curr_j] = 0.5 * der;

			for (int a = 0; a < 3; a++) {
				forces_grad(my_ind, idx_curr_j, a) += en_der_wrt_dist_j_grad[idx_curr_j] * vec_j[a] / dist_j;
				forces_grad(nbh.inds[j], idx_curr_j, a) -= en_der_wrt_dist_j_grad[idx_curr_j] * vec_j[a] / dist_j;
			}

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					stress_grad(a, b, idx_curr_j) -= en_der_wrt_dist_j_grad[idx_curr_j] *vec_j[a] * vec_j[b] / dist_j;
				}
			}
		}

		double force[3];
		for (int a = 0; a < 3; a++) {
			force[a] = en_der_wrt_dist_j * vec_j[a] / dist_j;
			cfg.force(my_ind)[a] += force[a];
			cfg.force(nbh.inds[j])[a] -= force[a];
		}

		for (int a = 0; a < 3; a++) {
			for (int b = 0; b < 3; b++) {
				cfg.stresses[a][b] -= vec_j[a]*force[b];
			}
		}
		
	}
}

void StillingerWeberRadialBasis::AccumulateEFS3body(Configuration& cfg, const Neighborhood& nbh, 
const int my_ind) {

	int n_at_types2 = n_at_types*n_at_types;
	int n_at_types3 = n_at_types2*n_at_types;
	int rad_size_n_types2 = p_RadialBasis->rb_size*n_at_types2;

	const double * tripple_potential_coeff = &regression_coeffs[rad_size_n_types2];
	const double * tripple_theta = &regression_coeffs[2*rad_size_n_types2];
	const double * tripple_lambda = &regression_coeffs[2*rad_size_n_types2+n_at_types3];
	const double * free_coeff = &regression_coeffs[2*rad_size_n_types2+2*n_at_types3];

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		double dist_j = nbh.dists[j];	
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;

		p_RadialBasis->RB_Calc(dist_j);
		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			rad_bas_val_j[alpha] = p_RadialBasis->rb_vals[alpha];
			rad_bas_der_j[alpha] = p_RadialBasis->rb_ders[alpha];
		}

		for (int k = j+1; k < nbh.count; k++) {		
			const Vector3& vec_k = nbh.vecs[k];
			double dist_k = nbh.dists[k];	
			const int idx_k = nbh.types[k] + nbh.my_type * n_at_types;
			const int idx_jk = nbh.types[k] + idx_j * n_at_types;	

			p_RadialBasis->RB_Calc(dist_k);

			double inner_prod = 0;			
			for (int a = 0; a < 3; a++)
				 inner_prod += vec_j[a]*vec_k[a];
			double cos_theta = inner_prod / (dist_j * dist_k); 

			double lambda_factor = tripple_lambda[idx_jk] * (cos_theta + tripple_theta[idx_jk]);
			double lambda_factor2 = tripple_lambda[idx_jk] * (cos_theta + tripple_theta[idx_jk]) *
										(cos_theta + tripple_theta[idx_jk]);

			double inner_coeffs_vals_k = 0;
			double inner_coeffs_ders_k = 0;

			for (int gamma = 0; gamma < p_RadialBasis->rb_size; gamma++) {
				int idx_curr_k = gamma + idx_k * p_RadialBasis->rb_size;
				double coeff_times_val_k = tripple_potential_coeff[idx_curr_k] * p_RadialBasis->rb_vals[gamma];
				double coeff_times_der_k = tripple_potential_coeff[idx_curr_k] * p_RadialBasis->rb_ders[gamma];
				inner_coeffs_vals_k += coeff_times_val_k;
				inner_coeffs_ders_k += coeff_times_der_k;
			}

			double coeff_val_j_inner_coeffs_vals_k = 0;
			double coeff_der_j_inner_coeffs_vals_k = 0;
			double coeff_val_j_inner_coeffs_ders_k = 0;

			for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
				int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
				double coeff_times_val_j = tripple_potential_coeff[idx_curr_j] * rad_bas_val_j[alpha];
				double coeff_times_der_j = tripple_potential_coeff[idx_curr_j] * rad_bas_der_j[alpha];

				cfg.energy += lambda_factor2 * coeff_times_val_j * inner_coeffs_vals_k;
				coeff_val_j_inner_coeffs_vals_k += coeff_times_val_j * inner_coeffs_vals_k;
				coeff_der_j_inner_coeffs_vals_k += coeff_times_der_j * inner_coeffs_vals_k;
				coeff_val_j_inner_coeffs_ders_k += coeff_times_val_j * inner_coeffs_ders_k; 
			}

			double force_j[3];
			for (int a = 0; a < 3; a++) {
				force_j[a] = lambda_factor2 * coeff_der_j_inner_coeffs_vals_k * vec_j[a] / dist_j;
				force_j[a] += 2 * lambda_factor * (vec_k[a] / (dist_j * dist_k) - 
								cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k;
				cfg.force(my_ind)[a] += force_j[a];
				cfg.force(nbh.inds[j])[a] -= force_j[a];
			}

			double force_k[3];
			for (int a = 0; a < 3; a++) {
				force_k[a] = lambda_factor2 * coeff_val_j_inner_coeffs_ders_k * vec_k[a] / dist_k;
				force_k[a] += 2 * lambda_factor * (vec_j[a] / (dist_j * dist_k) - 
								cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k;
				cfg.force(my_ind)[a] += force_k[a];
				cfg.force(nbh.inds[k])[a] -= force_k[a];
			}

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					cfg.stresses[a][b] -= vec_j[a]*force_j[b];
					cfg.stresses[a][b] -= vec_k[a]*force_k[b];
				}
			}

		}
	}
	cfg.energy += free_coeff[nbh.my_type];
	
}

void StillingerWeberRadialBasis::AccumulateEnergyGrad3body(const Neighborhood& nbh, 
Array1D& energy_grad) {

	int n_at_types2 = n_at_types*n_at_types;
	int n_at_types3 = n_at_types2*n_at_types;
	int rad_size_n_types2 = p_RadialBasis->rb_size*n_at_types2;

	const double * tripple_potential_coeff = &regression_coeffs[rad_size_n_types2];
	const double * tripple_theta = &regression_coeffs[2*rad_size_n_types2];
	const double * tripple_lambda = &regression_coeffs[2*rad_size_n_types2+n_at_types3];

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j]; //
		double dist_j = nbh.dists[j];//	
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;//

		p_RadialBasis->RB_Calc(dist_j);//
		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			rad_bas_val_j[alpha] = p_RadialBasis->rb_vals[alpha];//
			rad_bas_der_j[alpha] = p_RadialBasis->rb_ders[alpha];//
		}

		for (int k = j+1; k < nbh.count; k++) {		
			const Vector3& vec_k = nbh.vecs[k];//
			double dist_k = nbh.dists[k];	//
			const int idx_k = nbh.types[k] + nbh.my_type * n_at_types;//
			const int idx_jk = nbh.types[k] + idx_j * n_at_types;	//

			p_RadialBasis->RB_Calc(dist_k);//

			double inner_prod = 0;//			
			for (int a = 0; a < 3; a++)//
				 inner_prod += vec_j[a]*vec_k[a];//
			double cos_theta = inner_prod / (dist_j * dist_k); //
			double cos_theta_p_coeff_theta = (cos_theta + tripple_theta[idx_jk]);//

			double lambda_factor = tripple_lambda[idx_jk] * cos_theta_p_coeff_theta; //
			double lambda_factor2 = tripple_lambda[idx_jk] * cos_theta_p_coeff_theta *
										cos_theta_p_coeff_theta; //

			double coeff_val_j_inner_coeffs_vals_k = 0;
			double inner_coeffs_vals_k = 0;

			for (int gamma = 0; gamma < p_RadialBasis->rb_size; gamma++) {
				int idx_curr_k = gamma + idx_k * p_RadialBasis->rb_size;
				double coeff_times_val_k = tripple_potential_coeff[idx_curr_k] * p_RadialBasis->rb_vals[gamma];
				inner_coeffs_vals_k += coeff_times_val_k;
			}
			

			//energies gradients w.r.t. tripple_potential_coeff
			for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
				int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
				double coeff_times_val_j = tripple_potential_coeff[idx_curr_j] * rad_bas_val_j[alpha];
				coeff_val_j_inner_coeffs_vals_k += coeff_times_val_j * inner_coeffs_vals_k;

				energy_grad[idx_curr_j+rad_size_n_types2] += lambda_factor2 * rad_bas_val_j[alpha] * 
																inner_coeffs_vals_k;
 
				for (int gamma = 0; gamma < p_RadialBasis->rb_size; gamma++) {
					int idx_curr_k = gamma + idx_k * p_RadialBasis->rb_size;
					energy_grad[idx_curr_k+rad_size_n_types2] += lambda_factor2 * coeff_times_val_j * 
																p_RadialBasis->rb_vals[gamma];
				}

			}

			//energies gradients w.r.t. tripple_theta
			energy_grad[idx_jk+2*rad_size_n_types2] += 2 * lambda_factor * coeff_val_j_inner_coeffs_vals_k;

			//energies gradients w.r.t. tripple_lambda
			energy_grad[idx_jk+2*rad_size_n_types2+n_at_types3] += cos_theta_p_coeff_theta * 
								cos_theta_p_coeff_theta * coeff_val_j_inner_coeffs_vals_k;
			
		}
	}

	energy_grad[2*rad_size_n_types2+2*n_at_types3+nbh.my_type] += 1;
}

void StillingerWeberRadialBasis::AccumulateEFSGrad3body(Configuration& cfg, const Neighborhood& nbh, 
const int my_ind, Array1D& energy_grad, Array3D& forces_grad, Array3D& stress_grad) {

	int n_at_types2 = n_at_types*n_at_types;
	int n_at_types3 = n_at_types2*n_at_types;
	int rad_size_n_types2 = p_RadialBasis->rb_size*n_at_types2;

	const double * tripple_potential_coeff = &regression_coeffs[rad_size_n_types2];
	const double * tripple_theta = &regression_coeffs[2*rad_size_n_types2];
	const double * tripple_lambda = &regression_coeffs[2*rad_size_n_types2+n_at_types3];
	const double * free_coeff = &regression_coeffs[2*rad_size_n_types2+2*n_at_types3];

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& vec_j = nbh.vecs[j];
		double dist_j = nbh.dists[j];	
		const int idx_j = nbh.types[j] + nbh.my_type * n_at_types;

		p_RadialBasis->RB_Calc(dist_j);
		for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
			rad_bas_val_j[alpha] = p_RadialBasis->rb_vals[alpha];
			rad_bas_der_j[alpha] = p_RadialBasis->rb_ders[alpha];
		}

		for (int k = j+1; k < nbh.count; k++) {		
			const Vector3& vec_k = nbh.vecs[k];
			double dist_k = nbh.dists[k];	
			const int idx_k = nbh.types[k] + nbh.my_type * n_at_types;
			const int idx_jk = nbh.types[k] + idx_j * n_at_types;	

			p_RadialBasis->RB_Calc(dist_k);

			double inner_prod = 0;			
			for (int a = 0; a < 3; a++)
				 inner_prod += vec_j[a]*vec_k[a];
			double cos_theta = inner_prod / (dist_j * dist_k); 
			double cos_theta_p_coeff_theta = (cos_theta + tripple_theta[idx_jk]);

			double lambda_factor = tripple_lambda[idx_jk] * cos_theta_p_coeff_theta;
			double lambda_factor2 = tripple_lambda[idx_jk] * cos_theta_p_coeff_theta *
										cos_theta_p_coeff_theta;

			FillWithZero(en_der_wrt_dist_j_grad);
			FillWithZero(en_der_wrt_dist_k_grad);	

			double inner_coeffs_vals_k = 0;
			double inner_coeffs_ders_k = 0;

			for (int gamma = 0; gamma < p_RadialBasis->rb_size; gamma++) {
				int idx_curr_k = gamma + idx_k * p_RadialBasis->rb_size;
				double coeff_times_val_k = tripple_potential_coeff[idx_curr_k] * p_RadialBasis->rb_vals[gamma];
				double coeff_times_der_k = tripple_potential_coeff[idx_curr_k] * p_RadialBasis->rb_ders[gamma];
				inner_coeffs_vals_k += coeff_times_val_k;
				inner_coeffs_ders_k += coeff_times_der_k;
			}

			double coeff_val_j_inner_coeffs_vals_k = 0;
			double coeff_der_j_inner_coeffs_vals_k = 0;
			double coeff_val_j_inner_coeffs_ders_k = 0;

			//energies, forces and stresses gradients w.r.t. tripple_potential_coeff
			for (int alpha = 0; alpha < p_RadialBasis->rb_size; alpha++) {
				int idx_curr_j = alpha + idx_j * p_RadialBasis->rb_size;
				double coeff_times_val_j = tripple_potential_coeff[idx_curr_j] * rad_bas_val_j[alpha];
				double coeff_times_der_j = tripple_potential_coeff[idx_curr_j] * rad_bas_der_j[alpha];

				cfg.energy += lambda_factor2 * coeff_times_val_j * inner_coeffs_vals_k;
				coeff_val_j_inner_coeffs_vals_k += coeff_times_val_j * inner_coeffs_vals_k;
				coeff_der_j_inner_coeffs_vals_k += coeff_times_der_j * inner_coeffs_vals_k;
				coeff_val_j_inner_coeffs_ders_k += coeff_times_val_j * inner_coeffs_ders_k; 

				//grad_k part
				for (int gamma = 0; gamma < p_RadialBasis->rb_size; gamma++) {
					int idx_curr_k = gamma + idx_k * p_RadialBasis->rb_size;				
					energy_grad[idx_curr_k+rad_size_n_types2] += lambda_factor2 * coeff_times_val_j * 
																p_RadialBasis->rb_vals[gamma];

					double coeff_val_j_inner_coeffs_vals_k_grad_k = coeff_times_val_j * p_RadialBasis->rb_vals[gamma];
					double coeff_der_j_inner_coeffs_vals_k_grad_k = coeff_times_der_j * p_RadialBasis->rb_vals[gamma];
					double coeff_val_j_inner_coeffs_ders_k_grad_k = coeff_times_val_j * p_RadialBasis->rb_ders[gamma];

					double force_grad_k_idx_j[3];
					for (int a = 0; a < 3; a++) {
						force_grad_k_idx_j[a] = lambda_factor2 * coeff_der_j_inner_coeffs_vals_k_grad_k * vec_j[a] / dist_j;
						force_grad_k_idx_j[a] += 2 * lambda_factor * (vec_k[a] / (dist_j * dist_k) - 
									cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k_grad_k;
						forces_grad(my_ind, idx_curr_k+rad_size_n_types2, a) += force_grad_k_idx_j[a];
						forces_grad(nbh.inds[j], idx_curr_k+rad_size_n_types2, a) -= force_grad_k_idx_j[a];
					}	

					double force_grad_k_idx_k[3];
					for (int a = 0; a < 3; a++) {
						force_grad_k_idx_k[a] = lambda_factor2 * coeff_val_j_inner_coeffs_ders_k_grad_k * vec_k[a] / dist_k;
						force_grad_k_idx_k[a] += 2 * lambda_factor * (vec_j[a] / (dist_j * dist_k) - 
										cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k_grad_k;
						forces_grad(my_ind, idx_curr_k+rad_size_n_types2, a) += force_grad_k_idx_k[a];
						forces_grad(nbh.inds[k], idx_curr_k+rad_size_n_types2, a) -= force_grad_k_idx_k[a];
					}

					for (int a = 0; a < 3; a++) {
						for (int b = 0; b < 3; b++) {
							stress_grad(a, b, idx_curr_k+rad_size_n_types2) -= vec_j[a]*force_grad_k_idx_j[b];
							stress_grad(a, b, idx_curr_k+rad_size_n_types2) -= vec_k[a]*force_grad_k_idx_k[b];
						}
					}
				}

				//grad_j part
				double coeff_val_j_inner_coeffs_vals_k_grad_j = rad_bas_val_j[alpha] * inner_coeffs_vals_k;
				double coeff_der_j_inner_coeffs_vals_k_grad_j = rad_bas_der_j[alpha] * inner_coeffs_vals_k;
				double coeff_val_j_inner_coeffs_ders_k_grad_j = rad_bas_val_j[alpha] * inner_coeffs_ders_k;

				energy_grad[idx_curr_j+rad_size_n_types2] += lambda_factor2 * rad_bas_val_j[alpha] * 
																inner_coeffs_vals_k;

				double force_grad_j_idx_j[3];
				for (int a = 0; a < 3; a++) {
					force_grad_j_idx_j[a] = lambda_factor2 * coeff_der_j_inner_coeffs_vals_k_grad_j
											 * vec_j[a] / dist_j; 
					force_grad_j_idx_j[a] += 2 * lambda_factor * (vec_k[a] / (dist_j * dist_k) - 
								cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k_grad_j; 

					forces_grad(my_ind, idx_curr_j+rad_size_n_types2, a) += force_grad_j_idx_j[a];
					forces_grad(nbh.inds[j], idx_curr_j+rad_size_n_types2, a) -= force_grad_j_idx_j[a];
				}	

				double force_grad_j_idx_k[3];
				for (int a = 0; a < 3; a++) {
					force_grad_j_idx_k[a] = lambda_factor2 * coeff_val_j_inner_coeffs_ders_k_grad_j
											 * vec_k[a] / dist_k; 
					force_grad_j_idx_k[a] += 2 * lambda_factor * (vec_j[a] / (dist_j * dist_k) - 
								cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k_grad_j; 
					forces_grad(my_ind, idx_curr_j+rad_size_n_types2, a) += force_grad_j_idx_k[a];
					forces_grad(nbh.inds[k], idx_curr_j+rad_size_n_types2, a) -= force_grad_j_idx_k[a];
				}

				for (int a = 0; a < 3; a++) {
					for (int b = 0; b < 3; b++) {
						stress_grad(a, b, idx_curr_j+rad_size_n_types2) -= vec_j[a]*force_grad_j_idx_j[b];
						stress_grad(a, b, idx_curr_j+rad_size_n_types2) -= vec_k[a]*force_grad_j_idx_k[b];
					}
				}
			}

			//forces and stresses
			double force_j[3];
			for (int a = 0; a < 3; a++) {
				force_j[a] = lambda_factor2 * coeff_der_j_inner_coeffs_vals_k * vec_j[a] / dist_j;
				force_j[a] += 2 * lambda_factor * (vec_k[a] / (dist_j * dist_k) - 
								cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k;
				cfg.force(my_ind)[a] += force_j[a];
				cfg.force(nbh.inds[j])[a] -= force_j[a];
			}

			double force_k[3];
			for (int a = 0; a < 3; a++) {
				force_k[a] = lambda_factor2 * coeff_val_j_inner_coeffs_ders_k * vec_k[a] / dist_k;
				force_k[a] += 2 * lambda_factor * (vec_j[a] / (dist_j * dist_k) - 
								cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k;
				cfg.force(my_ind)[a] += force_k[a];
				cfg.force(nbh.inds[k])[a] -= force_k[a];
			}

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					cfg.stresses[a][b] -= vec_j[a]*force_j[b];
					cfg.stresses[a][b] -= vec_k[a]*force_k[b];
				}
			}

			//energies, forces and stresses gradients w.r.t. tripple_theta
			energy_grad[idx_jk+2*rad_size_n_types2] += 2 * lambda_factor * coeff_val_j_inner_coeffs_vals_k;

			double force_grad_j[3];
			for (int a = 0; a < 3; a++) {
				force_grad_j[a] = 2 * lambda_factor * coeff_der_j_inner_coeffs_vals_k * vec_j[a] / dist_j;
				force_grad_j[a] += 2 * tripple_lambda[idx_jk] * (vec_k[a] / (dist_j * dist_k) - 
								cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k;
				forces_grad(my_ind, idx_jk+2*rad_size_n_types2, a) += force_grad_j[a];
				forces_grad(nbh.inds[j], idx_jk+2*rad_size_n_types2, a) -= force_grad_j[a];
			}

			double force_grad_k[3];
			for (int a = 0; a < 3; a++) {
				force_grad_k[a] = 2 * lambda_factor * coeff_val_j_inner_coeffs_ders_k * vec_k[a] / dist_k;
				force_grad_k[a] += 2 * tripple_lambda[idx_jk] * (vec_j[a] / (dist_j * dist_k) - 
								cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k;
				forces_grad(my_ind, idx_jk+2*rad_size_n_types2, a) += force_grad_k[a];
				forces_grad(nbh.inds[k], idx_jk+2*rad_size_n_types2, a) -= force_grad_k[a];
			}

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					stress_grad(a, b, idx_jk+2*rad_size_n_types2) -= vec_j[a]*force_grad_j[b];
					stress_grad(a, b, idx_jk+2*rad_size_n_types2) -= vec_k[a]*force_grad_k[b];
				}
			}

			//energies, forces and stresses gradients w.r.t. tripple_lambda
			energy_grad[idx_jk+2*rad_size_n_types2+n_at_types3] += cos_theta_p_coeff_theta * 
								cos_theta_p_coeff_theta * coeff_val_j_inner_coeffs_vals_k;

			for (int a = 0; a < 3; a++) {
				force_grad_j[a] = cos_theta_p_coeff_theta * cos_theta_p_coeff_theta * 
									coeff_der_j_inner_coeffs_vals_k * vec_j[a] / dist_j;
				force_grad_j[a] += 2 * cos_theta_p_coeff_theta * (vec_k[a] / (dist_j * dist_k) - 
								cos_theta * vec_j[a] / (dist_j * dist_j)) * coeff_val_j_inner_coeffs_vals_k;
				forces_grad(my_ind, idx_jk+2*rad_size_n_types2+n_at_types3, a) += force_grad_j[a];
				forces_grad(nbh.inds[j], idx_jk+2*rad_size_n_types2+n_at_types3, a) -= force_grad_j[a];
			}

			for (int a = 0; a < 3; a++) {
				force_grad_k[a] = cos_theta_p_coeff_theta * cos_theta_p_coeff_theta * 
									coeff_val_j_inner_coeffs_ders_k * vec_k[a] / dist_k;
				force_grad_k[a] += 2 * cos_theta_p_coeff_theta * (vec_j[a] / (dist_j * dist_k) - 
								cos_theta * vec_k[a] / (dist_k * dist_k)) * coeff_val_j_inner_coeffs_vals_k;
				forces_grad(my_ind, idx_jk+2*rad_size_n_types2+n_at_types3, a) += force_grad_k[a];
				forces_grad(nbh.inds[k], idx_jk+2*rad_size_n_types2+n_at_types3, a) -= force_grad_k[a];
			}

			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++) {
					stress_grad(a, b, idx_jk+2*rad_size_n_types2+n_at_types3) -= vec_j[a]*force_grad_j[b];
					stress_grad(a, b, idx_jk+2*rad_size_n_types2+n_at_types3) -= vec_k[a]*force_grad_k[b];
				}
			}
			
		}
	}
	cfg.energy += free_coeff[nbh.my_type];
	energy_grad[2*rad_size_n_types2+2*n_at_types3+nbh.my_type] += 1;
}

void StillingerWeberRadialBasis::CalcE(Configuration& cfg) {

	ResetEFS(cfg);

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		//AccumulateEnergy2body(cfg, nbh);
	}

}

void StillingerWeberRadialBasis::CalcEFS(Configuration& cfg) {

	ResetEFS(cfg);

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		//cout << "ind = " << ind << endl;
		AccumulateEFS2body(cfg, nbh, ind);
		AccumulateEFS3body(cfg, nbh, ind);
	}

	//cout << cfg.energy << endl;
}

void StillingerWeberRadialBasis::CalcEnergyGrad(Configuration& cfg, Array1D& energy_grad) {

	energy_grad.resize(CoeffCount());
	FillWithZero(energy_grad);

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		AccumulateEnergyGrad2body(nbh, energy_grad);
		AccumulateEnergyGrad3body(nbh, energy_grad);
	}
}

void StillingerWeberRadialBasis::CalcEFSGrads(Configuration& cfg, Array1D& energy_grad, 
Array3D& forces_grad, Array3D& stress_grad) {

	ResetEFS(cfg);

	int natom = cfg.size();

	energy_grad.resize(CoeffCount());
	FillWithZero(energy_grad);
	forces_grad.resize(natom, CoeffCount(), 3);
	forces_grad.set(0);
	stress_grad.resize(3, 3, CoeffCount());
	stress_grad.set(0);

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];

		//cout << "ind = " << ind << endl;
		AccumulateEFSGrad2body(cfg, nbh, ind, energy_grad, forces_grad, stress_grad);
		AccumulateEFSGrad3body(cfg, nbh, ind, energy_grad, forces_grad, stress_grad);
	}
}

void StillingerWeberRadialBasis::AccumulateCombinationGrad(const Neighborhood& nbh, 
Array1D& out_grad_accumulator, const double se_weight, const Vector3* se_ders_weights) {

}

StillingerWeberRadialBasis::StillingerWeberRadialBasis(const std::string& filename) 
{
	Load(filename);
	en_der_wrt_dist_j_grad.resize(regression_coeffs.size());
	en_der_wrt_dist_k_grad.resize(regression_coeffs.size());
	rad_bas_val_j.resize(p_RadialBasis->rb_size);
	rad_bas_der_j.resize(p_RadialBasis->rb_size);
}

void StillingerWeberRadialBasis::Load(const string& filename)
{
	ifstream ifs(filename);

	try {
		if (!ifs.is_open()) throw 1;
	}
	catch(...)
	{
		ERROR((string)"Can't open file \"" + filename + "\" for reading");
	}

	string line;

	getline(ifs, line);

	string foo_str;
	char foo_char;
	ifs >> foo_str >> foo_char >> n_at_types;
	getline(ifs, line);
	ifs >> foo_str >> foo_char >> rbasis_type;
	getline(ifs, line);
	if (rbasis_type == "RBChebyshev")
		p_RadialBasis = new RadialBasis_Chebyshev(ifs);
	else if (rbasis_type == "RBChebyshev_repuls")
		p_RadialBasis = new RadialBasis_Chebyshev_repuls(ifs);
	else if (rbasis_type == "RBShapeev")
		p_RadialBasis = new RadialBasis_Shapeev(ifs);
	else if (rbasis_type == "RBTaylor")
		p_RadialBasis = new RadialBasis_Taylor(ifs);
	else
		ERROR("Wrong radial basis type");
	//ifs >> foo_str >> foo_char >> min_dist;
	//getline(ifs, line);
	//ifs >> foo_str >> foo_char >> max_dist;
	//getline(ifs, line);
	//ifs >> foo_str >> foo_char >> radial_basis_size;
	//getline(ifs, line);

	int n_coeffs = 2*n_at_types*n_at_types*(p_RadialBasis->rb_size+n_at_types)+n_at_types;

	regression_coeffs.resize(n_coeffs);

	ifs >> foo_str >> foo_char >> foo_char;

	if (foo_str == "sw_coefficients") {
		for (int i = 0; i < CoeffCount(); i++) {
			ifs >> regression_coeffs[i] >> foo_char;
		}
	}
	else {
		FillWithZero(regression_coeffs);
	}
}

void StillingerWeberRadialBasis::Save(const string& filename)
{
	ofstream ofs(filename);

	ofs << "Stillinger-Weber" << endl;
	
	ofs << "species_count = " << n_at_types << endl;
	ofs << "radial_basis_type = " << rbasis_type << endl;
	ofs << "    scaling = " << p_RadialBasis->scaling << endl;
	ofs << "    min_dist = " << p_RadialBasis->min_dist << endl;
	ofs << "    max_dist = " << p_RadialBasis->max_dist << endl;
	ofs << "    radial_basis_size = " << p_RadialBasis->rb_size << endl;

	ofs.setf(ios::scientific);
	ofs.precision(15);

	ofs << "sw_coefficients = {";

	for (int i = 0; i < CoeffCount()-1; i++) {
		ofs << regression_coeffs[i] << ", ";
	}

	ofs << regression_coeffs[CoeffCount()-1] << "}" << endl;
}


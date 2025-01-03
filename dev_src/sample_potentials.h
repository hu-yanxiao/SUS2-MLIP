/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Ivan Novikov
 */

#ifndef MLIP_SamplePotentials
#define MLIP_SamplePotentials

#include "../src/basic_mlip.h"

//! This is a sample class for a pair potential
//! It is a linear potential, but it pretends it is not
//! (and therefore is fitted with nonlinear optimizer like bfgs)
//! Uses the AnyLocalMLIP class, allowing to just define the site potentials
class PairPotentialNonlinear : public AnyLocalMLIP
{
protected:
	std::vector<double> regression_coeffs;

protected:

	void AddSiteEnergyGrad(	const Neighborhood& neighborhood, 
							std::vector<double>& site_energy_grad) override {}

	void CalcSiteEnergyDers(const Neighborhood& nbh) override
	{
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);
			for (int i = 0; i < p_RadialBasis->rb_size; i++) {
				double val = p_RadialBasis->rb_vals[i];
				double der = p_RadialBasis->rb_ders[i];

				// pair potential part
				buff_site_energy_ += val * regression_coeffs[i];
				buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * regression_coeffs[i]);
			}
		}
	}

	void AccumulateCombinationGrad(const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double se_weight = 0.0,
		const Vector3* se_ders_weights = nullptr) override
	{
		out_grad_accumulator.resize(CoeffCount());
		buff_site_energy_ders_.resize(nbh.count);

		FillWithZero(buff_site_energy_ders_);
		buff_site_energy_ = 0.0;

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);
			for (int i = 0; i < p_RadialBasis->rb_size; i++) {
				double val = p_RadialBasis->rb_vals[i];
				double der = p_RadialBasis->rb_ders[i];

				// pair potential part
				buff_site_energy_ += val * regression_coeffs[i];
				out_grad_accumulator[i] += se_weight * val;
				buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * regression_coeffs[i]);
				if (se_ders_weights != nullptr)
					out_grad_accumulator[i] += vec_j.dot(se_ders_weights[j]) * (der / dist_j);
			}
		}
	}

public:
	PairPotentialNonlinear(int _radial_basis_count, double _min_dist = 1.6, double _max_dist = 5.0)
		: AnyLocalMLIP(_min_dist, _max_dist, _radial_basis_count)
//		, radial_basis_count(_radial_basis_count)
		, regression_coeffs(_radial_basis_count)
	{
	}

	// this may disapear in future:
	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}
};


//! This is a single-component EAM potential
//! Uses the AnyLocalMLIP class, allowing to just define the site potentials
class EAMSimple : public AnyLocalMLIP
{
protected:
	std::vector<double> regression_coeffs;

public:
	int radial_basis_count_; //!< number of radial basis functions used
	int embedding_parameters_count_; //!< number of terms in f(rho) = c_0 + c_2 rho^2 + c rho^3 + ...
	//!< note that c_1 rho is missing intentionally

	EAMSimple(int _radial_basis_count, int embedding_function_parameters_count,
		double _min_dist = 1.6, double _max_dist = 5.0)
		: AnyLocalMLIP(_min_dist, _max_dist, _radial_basis_count)
		, radial_basis_count_(_radial_basis_count)
		, embedding_parameters_count_(embedding_function_parameters_count)
		, regression_coeffs(2 * _radial_basis_count + embedding_function_parameters_count)
	{}

	// this may disapear in future:
	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}

protected:
	void AddPenaltyGrad(const double coeff, double& out_penalty, Array1D* out_penalty_grad) override
	{
		const double* density_coeffs = &regression_coeffs[radial_basis_count_];

		double sum_coeffs2 = 0;
		for (int i = 0; i < radial_basis_count_; i++) 
			sum_coeffs2 += density_coeffs[i]*density_coeffs[i];

		out_penalty += coeff * (sum_coeffs2-1) * (sum_coeffs2-1);

		if (out_penalty_grad != nullptr)
			for (int i = radial_basis_count_; i < 2 * radial_basis_count_; i++) 
				(*out_penalty_grad)[i] += coeff * 4 * (sum_coeffs2 - 1) * density_coeffs[i - radial_basis_count_];
	}

	void AddSiteEnergyGrad(const Neighborhood& neighborhood, 
		std::vector<double>& site_energy_grad) override
	{
		//const double* pair_potentials_coeffs = &regression_coeffs[0];
		const double* density_coeffs = &regression_coeffs[radial_basis_count_];
		const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

		//std::cout << radial_basis_count_ << std::endl;	

		double density = 0;
		//site_energy_grad.resize(2 * radial_basis_count_ + embedding_parameters_count_);
		//FillWithZero(site_energy_grad);

		for (int j = 0; j < neighborhood.count; j++) {
			const double dist_j = neighborhood.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(dist_j);

			// density calculation
			for (int i = 0; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				density += val * density_coeffs[i];
				site_energy_grad[i] += 0.5 * val;
			}
		}

		// embedding function and its derivative WRT density
		double density_pow = density; // will be density^m
		//double embedding = embedding_coeffs[0];
		double embedding_der = 0;
		site_energy_grad[2*radial_basis_count_] += 1;
		for (int m = 2; m < embedding_parameters_count_+1; m++) {
			embedding_der += m * embedding_coeffs[m-1] * density_pow;
			density_pow *= density;
			site_energy_grad[m-1+2*radial_basis_count_] += density_pow;
		}

		for (int j = 0; j < neighborhood.count; j++) {
			const double dist_j = neighborhood.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(dist_j);

			for (int i = radial_basis_count_; i < 2 * radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i - radial_basis_count_];
				site_energy_grad[i] += embedding_der * val;
			}
		}
	}

	void CalcSiteEnergyDers(const Neighborhood& nbh) override
	{
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* pair_potentials_coeffs = &regression_coeffs[0];
		const double* density_coeffs = &regression_coeffs[radial_basis_count_];
		const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

		//std::cout << radial_basis_count_ << std::endl;	

		double density = 0;

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);

			// pair potential part
			for (int i = 0; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				const double der = p_RadialBasis->rb_ders[i];
				buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[i];
				buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[i]);
			}

			// density calculation
			for (int i = 0; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				density += val * density_coeffs[i];
			}
		}

		// embedding function and its derivative WRT density
		double density_pow = density; // will be density^m
		double embedding = embedding_coeffs[0];
		double embedding_der = 0;
		for (int m = 2; m < embedding_parameters_count_ + 1; m++) {
			embedding_der += m * embedding_coeffs[m-1] * density_pow;
			density_pow *= density;
			embedding += embedding_coeffs[m-1] * density_pow;
		}

		buff_site_energy_ += embedding;

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);
			// now we can compute the derivative of the embedding WRT atom positions
			for (int i = 0; i < radial_basis_count_; i++) {
				const double der = p_RadialBasis->rb_ders[i];
				buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * density_coeffs[i]) * embedding_der;
			}
		}
	}

	void AccumulateCombinationGrad(const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator,
		const double se_weight = 0.0,
		const Vector3* se_ders_weights = nullptr) override
	{
		out_grad_accumulator.resize(CoeffCount());
		buff_site_energy_ders_.resize(nbh.count);

		FillWithZero(buff_site_energy_ders_);
		buff_site_energy_ = 0.0;

		const double* pair_potentials_coeffs = &regression_coeffs[0];
		const double* density_coeffs = &regression_coeffs[radial_basis_count_];
		const double* embedding_coeffs = &regression_coeffs[2 * radial_basis_count_];

		double density = 0;
		std::vector<double> density_grad(radial_basis_count_);
		FillWithZero(density_grad);

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);

			// pair potential part
			for (int i = 0; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				const double der = p_RadialBasis->rb_ders[i];
				buff_site_energy_ += 0.5 * val * pair_potentials_coeffs[i];
				buff_site_energy_ders_[j] += 0.5 * vec_j * ((der / dist_j) * pair_potentials_coeffs[i]);
			}

			// density calculation
			for (int i = 0; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				density += val * density_coeffs[i];
				density_grad[i] += val;
			}
		}

		// embedding function and its derivative WRT density
		double density_pow = density; // will be density^m
		double embedding = embedding_coeffs[0];
		double embedding_der = 0;
		double embedding_der2 = 0;
		out_grad_accumulator[2*radial_basis_count_] += se_weight * 1;
		for (int m = 2; m < embedding_parameters_count_+1; m++) {
			embedding_der += m * embedding_coeffs[m-1] * density_pow;
			density_pow *= density;
			embedding += embedding_coeffs[m-1] * density_pow;
			out_grad_accumulator[m-1+2*radial_basis_count_] += se_weight * density_pow;
		}
		density_pow = 1; // will be density^m
		for (int m = 2; m < embedding_parameters_count_+1; m++) {
			embedding_der2 += m * (m - 1) * embedding_coeffs[m-1] * density_pow;
			density_pow *= density;
		}

		buff_site_energy_ += embedding;

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);
			// now we can compute the derivative of the embedding WRT atom positions
			for (int i = 0; i < radial_basis_count_; i++) {
				const double der = p_RadialBasis->rb_ders[i];
				buff_site_energy_ders_[j] += vec_j * ((der / dist_j) * density_coeffs[i]) * embedding_der;
			}
		}

		// Computing the functional
		//double functional = out_site_energy * se_weight;
		//for (int j = 0; j < nbh.count; j++)
		//functional += out_site_energy_ders[j].dot(se_ders_weights[j]);

		//int coeff_count = CoeffCount();

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double vec_j_se_ders_inner = vec_j.dot(se_ders_weights[j]);

			// calculates vals and ders for j-th atom in the neighborhood
			p_RadialBasis->RB_Calc(nbh.dists[j]);

			double total_density_coeff_der = 0;
			int i = 0;
			for (; i < radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i];
				const double der = p_RadialBasis->rb_ders[i];
				out_grad_accumulator[i] += 0.5 * se_weight * val;
				out_grad_accumulator[i] += 0.5 * vec_j_se_ders_inner * (der / dist_j);
				total_density_coeff_der += density_coeffs[i] * der;
			}
			for (; i < 2 * radial_basis_count_; i++) {
				const double val = p_RadialBasis->rb_vals[i - radial_basis_count_];
				const double der = p_RadialBasis->rb_ders[i - radial_basis_count_];
				out_grad_accumulator[i] += se_weight * embedding_der * val;
				out_grad_accumulator[i] += vec_j_se_ders_inner * embedding_der2
					* density_grad[i - radial_basis_count_] * total_density_coeff_der / dist_j;
				out_grad_accumulator[i] += vec_j_se_ders_inner * embedding_der * (der / dist_j);
			}
			double density_pou = density; // will be density^m
			for (i = 2 * radial_basis_count_ + 2; i <= CoeffCount(); i++) {
				int idx = i - 2 * radial_basis_count_;
				out_grad_accumulator[i-1] += vec_j_se_ders_inner * idx * density_pou * total_density_coeff_der / dist_j;
				density_pou *= density;
			}
		}
	}
};


class LennardJones : public AnyLocalMLIP
{
protected:
	std::vector<double> regression_coeffs;

public:
	const std::string filename;
	int atom_types_count_;

	LennardJones(const std::string filename_, double _min_dist = 1.6, double _max_dist = 5.0, int _radial_basis_count = 0)
		: AnyLocalMLIP(_min_dist, _max_dist, _radial_basis_count)
		, filename(filename_)
	{
		Load(filename);
	}

	// this may disapear in future:
	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}

	void Load(const std::string& filename)
	{
		std::ifstream ifs(filename);

		try {
			if (!ifs.is_open()) throw MlipException("wrong filename!");
		}
		catch(...)
		{
			ERROR("Can't open file \"" + filename + "\" for reading");
		}

		std::string line;

		getline(ifs, line);

		std::string foo_str;
		char foo_char;
		ifs >> foo_str >> foo_char >> atom_types_count_;
		getline(ifs, line);

		regression_coeffs.resize(2*atom_types_count_*atom_types_count_);

		ifs >> foo_str >> foo_char >> foo_char;

		for (int i = 0; i < CoeffCount(); i++) {
			ifs >> regression_coeffs[i] >> foo_char;
		}
	}

	void Save(const std::string& filename)
	{
		std::ofstream ofs(filename);

		ofs << "LJLinear" << std::endl;
	
		ofs << "species_count = " << atom_types_count_ << std::endl;

		ofs.setf(std::ios::scientific);
		ofs.precision(15);

		ofs << "lj_coefficients = {";

		for (int i = 0; i < CoeffCount()-1; i++) {
			ofs << regression_coeffs[i] << ", ";
		}

		ofs << regression_coeffs[CoeffCount()-1] << "}" << std::endl;
	}

protected:
	void CalcSiteEnergyDers(const Neighborhood& nbh) override
	{
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int idx = nbh.types[j]  + nbh.my_type * atom_types_count_;

			buff_site_energy_ += a_coeffs[idx] / dist12 - b_coeffs[idx] / dist6;
			buff_site_energy_ders_[j] += - 12 * vec_j * a_coeffs[idx] / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 6 * vec_j * b_coeffs[idx] / (dist6 * dist_j * dist_j);
		}
	}

	void AccumulateCombinationGrad(const Neighborhood& nbh,
									std::vector<double>& out_grad_accumulator,
									const double se_weight = 0.0,
									const Vector3* se_ders_weights = nullptr) override
	{
		out_grad_accumulator.resize(CoeffCount());

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int a_idx = nbh.types[j]  + nbh.my_type * atom_types_count_;
			const int b_idx = nbh.types[j]  + nbh.my_type * atom_types_count_ +
				atom_types_count_ * atom_types_count_;

			buff_site_energy_ += a_coeffs[a_idx] / dist12 - b_coeffs[a_idx] / dist6;
			buff_site_energy_ders_[j] -= 12 * vec_j * a_coeffs[a_idx] / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 6 * vec_j * b_coeffs[a_idx] / (dist6 * dist_j * dist_j);

			out_grad_accumulator[a_idx] += se_weight / dist12;
			out_grad_accumulator[b_idx] -= se_weight / dist6;
			if (se_ders_weights != nullptr) {
				out_grad_accumulator[a_idx] -= 12 * vec_j.dot(se_ders_weights[j]) / (dist12 * dist_j * dist_j);
				out_grad_accumulator[b_idx] += 6 * vec_j.dot(se_ders_weights[j]) / (dist6 * dist_j * dist_j);
			}
		}
	}
};

class NonlinearLennardJones : public AnyLocalMLIP
{
protected:
	std::vector<double> regression_coeffs;

public:
	const std::string filename;
	int atom_types_count_;

	NonlinearLennardJones(const std::string filename_, double _min_dist = 1.6, double _max_dist = 5.0, int _radial_basis_count = 0)
		: AnyLocalMLIP(_min_dist, _max_dist, _radial_basis_count)
		, filename(filename_)
	{
		Load(filename);
	}


	// this may disapear in future:
	int CoeffCount() //!< number of coefficients
	{
		return (int)regression_coeffs.size();
	}
	double* Coeff() //!< coefficients themselves
	{
		return &regression_coeffs[0];
	}

	void Load(const std::string& filename)
	{
		std::ifstream ifs(filename);

		try {
			if (!ifs.is_open()) throw MlipException("wrong filename!");
		}
		catch(...)
		{
			ERROR("Can't open file \"" + filename + "\" for reading");
		}

		std::string line;

		getline(ifs, line);

		std::string foo_str;
		char foo_char;
		ifs >> foo_str >> foo_char >> atom_types_count_;
		getline(ifs, line);

		regression_coeffs.resize(2*atom_types_count_);

		ifs >> foo_str >> foo_char >> foo_char;

		for (int i = 0; i < CoeffCount(); i++) {
			ifs >> regression_coeffs[i] >> foo_char;
		}
	}

	void Save(const std::string& filename)
	{
		std::ofstream ofs(filename);

		ofs << "LJNonlinear" << std::endl;
	
		ofs << "species_count = " << atom_types_count_ << std::endl;

		ofs.setf(std::ios::scientific);
		ofs.precision(15);

		ofs << "lj_coefficients = {";

		for (int i = 0; i < CoeffCount()-1; i++) {
			ofs << regression_coeffs[i] << ", ";
		}

		ofs << regression_coeffs[CoeffCount()-1] << "}" << std::endl;
	}

protected:

	void AddSiteEnergyGrad_old(const Neighborhood& neighborhood, 
		std::vector<double>& site_energy_grad)
	{
		for (int j = 0; j < neighborhood.count; j++) {
			//const Vector3& vec_j = neighborhood.vecs[j];
			const double dist_j = neighborhood.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int a_idx = neighborhood.types[j]  + neighborhood.my_type * atom_types_count_;
			const int b_idx = neighborhood.types[j]  + neighborhood.my_type * atom_types_count_ + 
								atom_types_count_ * atom_types_count_;

			site_energy_grad[a_idx] += 0.5 * 1 / dist12;
			site_energy_grad[b_idx] += 0.5 * -1 / dist6;
		}	
	}

	void CalcSiteEnergyDers_old(const Neighborhood& nbh)
	{
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int idx = nbh.types[j]  + nbh.my_type * atom_types_count_;

			buff_site_energy_ += 0.5 * a_coeffs[idx] / dist12 - 0.5 * b_coeffs[idx] / dist6;
			buff_site_energy_ders_[j] += - 0.5 * 12 * vec_j * a_coeffs[idx] / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b_coeffs[idx] / (dist6 * dist_j * dist_j);
		}
	}

	void SiteEnergyDersGrad_old(const Neighborhood& nbh,
		std::vector<double>& out_grad_accumulator, 
		double se_weight, 
		const std::vector<Vector3>& se_ders_weights)
	{
		out_grad_accumulator.resize(CoeffCount());
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_ * atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int a_idx = nbh.types[j] + nbh.my_type * atom_types_count_;
			const int b_idx = nbh.types[j] + nbh.my_type * atom_types_count_ +
									atom_types_count_ * atom_types_count_;

			buff_site_energy_ += 0.5 * a_coeffs[a_idx] / dist12 - 0.5 * b_coeffs[a_idx] / dist6;
			buff_site_energy_ders_[j] -= 0.5 * 12 * vec_j * a_coeffs[a_idx] / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b_coeffs[a_idx] / (dist6 * dist_j * dist_j);
			
			out_grad_accumulator[a_idx] += 0.5 * se_weight / dist12;
			out_grad_accumulator[b_idx] -= 0.5 * se_weight / dist6;
			//if (se_ders_weights != nullptr) {
				out_grad_accumulator[a_idx] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) / (dist12 * dist_j * dist_j);
				out_grad_accumulator[b_idx] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) / (dist6 * dist_j * dist_j);
			//}
		}
	}

	void AddSiteEnergyGrad(const Neighborhood& neighborhood, 
		std::vector<double>& site_energy_grad)
	{
		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_];

		for (int j = 0; j < neighborhood.count; j++) {
			//const Vector3& vec_j = neighborhood.vecs[j];
			const double dist_j = neighborhood.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int a_idx1 = neighborhood.my_type;
			const int a_idx2 = neighborhood.types[j];
			const int b_idx1 = neighborhood.my_type + atom_types_count_;
			const int b_idx2 = neighborhood.types[j] + atom_types_count_;

			/*double a1a2_der_idx1 = 0.5 * sqrt(a_coeffs[a_idx2] / a_coeffs[a_idx1]);
			double a1a2_der_idx2 = 0.5 * sqrt(a_coeffs[a_idx1] / a_coeffs[a_idx2]);
			double b1b2_der_idx1 = 0.5 * sqrt(b_coeffs[a_idx2] / b_coeffs[a_idx1]);
			double b1b2_der_idx2 = 0.5 * sqrt(b_coeffs[a_idx1] / b_coeffs[a_idx2]);*/

			double a1a2_der_idx1 = 2 * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
			double a1a2_der_idx2 = 2 * a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1];
			double b1b2_der_idx1 = 2 * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];
			double b1b2_der_idx2 = 2 * b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1];

			site_energy_grad[a_idx1] += 0.5 * 1 * a1a2_der_idx1/ dist12;
			site_energy_grad[a_idx2] += 0.5 * 1 * a1a2_der_idx2 / dist12;
			site_energy_grad[b_idx1] += 0.5 * -1 * b1b2_der_idx1 / dist6;
			site_energy_grad[b_idx2] += 0.5 * -1 * b1b2_der_idx2 / dist6;
		}	
	}

	void CalcSiteEnergyDers(const Neighborhood& nbh) override
	{
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int idx1 = nbh.my_type;
			const int idx2 = nbh.types[j];

			//double a1a2 = sqrt(a_coeffs[idx1] * a_coeffs[idx2]);
			//double b1b2 = sqrt(b_coeffs[idx1] * b_coeffs[idx2]);

			double a1a2 = a_coeffs[idx1] * a_coeffs[idx2] * a_coeffs[idx1] * a_coeffs[idx2];
			double b1b2 = b_coeffs[idx1] * b_coeffs[idx2] * b_coeffs[idx1] * b_coeffs[idx2];

			buff_site_energy_ += 0.5 * a1a2 / dist12 - 0.5 * b1b2 / dist6;
			buff_site_energy_ders_[j] += - 0.5 * 12 * vec_j * a1a2 / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b1b2 / (dist6 * dist_j * dist_j);
		}
	}

	void AccumulateCombinationGrad(	const Neighborhood& nbh,
									std::vector<double>& out_grad_accumulator,
									const double se_weight = 0.0,
									const Vector3* se_ders_weights = nullptr) override
	{
		out_grad_accumulator.resize(CoeffCount());
		buff_site_energy_ = 0.0;
		buff_site_energy_ders_.resize(nbh.count);
		FillWithZero(buff_site_energy_ders_);

		const double* a_coeffs = &regression_coeffs[0];
		const double* b_coeffs = &regression_coeffs[atom_types_count_];

		for (int j = 0; j < nbh.count; j++) {
			const Vector3& vec_j = nbh.vecs[j];
			const double dist_j = nbh.dists[j];
			const double dist6 = dist_j * dist_j * dist_j * dist_j * dist_j * dist_j;
			const double dist12 = dist6 * dist6;
			const int a_idx1 = nbh.my_type;
			const int a_idx2 = nbh.types[j];
			const int b_idx1 = nbh.my_type + atom_types_count_;
			const int b_idx2 = nbh.types[j] + atom_types_count_;

			/*double a1a2 = sqrt(a_coeffs[a_idx1] * a_coeffs[a_idx2]);
			double b1b2 = sqrt(b_coeffs[a_idx1] * b_coeffs[a_idx2]);

			double a1a2_der_idx1 = 0.5 * sqrt(a_coeffs[a_idx2] / a_coeffs[a_idx1]);
			double a1a2_der_idx2 = 0.5 * sqrt(a_coeffs[a_idx1] / a_coeffs[a_idx2]);
			double b1b2_der_idx1 = 0.5 * sqrt(b_coeffs[a_idx2] / b_coeffs[a_idx1]);
			double b1b2_der_idx2 = 0.5 * sqrt(b_coeffs[a_idx1] / b_coeffs[a_idx2]);*/

			double a1a2 = a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
			double b1b2 = b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];

			double a1a2_der_idx1 = 2 * a_coeffs[a_idx2] * a_coeffs[a_idx1] * a_coeffs[a_idx2];
			double a1a2_der_idx2 = 2 * a_coeffs[a_idx1] * a_coeffs[a_idx2] * a_coeffs[a_idx1];
			double b1b2_der_idx1 = 2 * b_coeffs[a_idx2] * b_coeffs[a_idx1] * b_coeffs[a_idx2];
			double b1b2_der_idx2 = 2 * b_coeffs[a_idx1] * b_coeffs[a_idx2] * b_coeffs[a_idx1];

			buff_site_energy_ += 0.5 * a1a2 / dist12 - 0.5 * b1b2 / dist6;
			buff_site_energy_ders_[j] -= 0.5 * 12 * vec_j * a1a2 / (dist12 * dist_j * dist_j);
			buff_site_energy_ders_[j] += 0.5 * 6 * vec_j * b1b2 / (dist6 * dist_j * dist_j);
			
			out_grad_accumulator[a_idx1] += 0.5 * se_weight * a1a2_der_idx1 / dist12;
			out_grad_accumulator[b_idx1] -= 0.5 * se_weight * b1b2_der_idx1 / dist6;
			out_grad_accumulator[a_idx2] += 0.5 * se_weight * a1a2_der_idx2 / dist12;
			out_grad_accumulator[b_idx2] -= 0.5 * se_weight * b1b2_der_idx2 / dist6;
			if (se_ders_weights != nullptr) {
				out_grad_accumulator[a_idx1] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) *
							a1a2_der_idx1 / (dist12 * dist_j * dist_j);
				out_grad_accumulator[b_idx1] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) *
							b1b2_der_idx1 / (dist6 * dist_j * dist_j);
				out_grad_accumulator[a_idx2] -= 0.5 * 12 * vec_j.dot(se_ders_weights[j]) *
							a1a2_der_idx2 / (dist12 * dist_j * dist_j);
				out_grad_accumulator[b_idx2] += 0.5 * 6 * vec_j.dot(se_ders_weights[j]) *
							b1b2_der_idx2 / (dist6 * dist_j * dist_j);
			}
		}
	}

};

#endif // MLIP_SamplePotentials

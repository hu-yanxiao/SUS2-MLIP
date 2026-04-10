/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#include <algorithm>

#include "mtp.h"


using namespace std;


void MTP::MemAlloc()
{
	moment_vals.resize(alpha_moments_count);
	basis_vals.resize(alpha_count);
	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
}

void MTP::MemFree()
{
	if (alpha_moment_mapping != nullptr) delete[] alpha_moment_mapping;
	if (alpha_index_times != nullptr) delete[] alpha_index_times;
	if (alpha_index_basic != nullptr) delete[] alpha_index_basic;

	alpha_moment_mapping = nullptr;
	alpha_index_times = nullptr;
	alpha_index_basic = nullptr;

	if (p_RadialBasis != nullptr)
		delete p_RadialBasis;
}

void MTP::ReadRadialBasisOldFormat(ifstream& ifs)
{
	if ((!ifs.is_open()) || (ifs.eof()))
		ERROR("MTP_basis::read_params_file: Can't load radial basis");

	string tmpstr;

	// reading min_dist, max_dist and initializing radial basis
	double min_dist=0, max_dist=0;
	ifs >> tmpstr;
	if (tmpstr != "min_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> min_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "max_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> max_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	p_RadialBasis = new RadialBasis_Shapeev(min_dist, max_dist, default_radial_basis_size);

	// initializing radial coefficients
	radial_funcs_count = default_radial_basis_size;
	radial_coeffs.resize(radial_funcs_count, radial_funcs_count);
	radial_coeffs.set(0);
	for (int i=0; i<radial_funcs_count; i++)
		radial_coeffs(i, i) = 1.0;
}

void MTP::ReadRadialCoeffs(std::ifstream & ifs)
{
	if ((!ifs.is_open()) || (ifs.eof()))
		ERROR("MTP_basis::read_params_file: Can't load MTP basis functions");

	string tmpstr;

	// Radial functions count reading block
	ifs >> tmpstr;
	if (tmpstr != "radial_funcs_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> radial_funcs_count;
	if (radial_funcs_count <= 0 || radial_funcs_count > p_RadialBasis->rb_size)
		ERROR("Error reading .mtp file");
	radial_coeffs.resize(radial_funcs_count, p_RadialBasis->rb_size);

	// Radial coefficients reading block
	ifs >> tmpstr;
	if (tmpstr == "radial_coeffs")
	{
		ifs >> tmpstr;
		if (tmpstr != "0-0")
			ERROR("Error reading .mtp file");
		if (ifs.fail())
			ERROR("Error reading .mtp file");
		for (int i=0; i<radial_funcs_count; i++)
		{
			ifs.ignore(1000, '{');
			char foo = ' ';
			for (int j=0; j<p_RadialBasis->rb_size; j++)
				ifs >> radial_coeffs(i, j) >> foo;
		}

	}
	else
	{
		Warning("MTP: No radial coefficients loaded. Default one's will be set");
		radial_coeffs.resize(radial_funcs_count, p_RadialBasis->rb_size);
		radial_coeffs.set(0);
		for (int i=0; i<radial_funcs_count; i++)
			radial_coeffs(i, i) = 1.0;
	}
}

void MTP::ReadMTPBasis(ifstream& ifs)
{
	if ((!ifs.is_open()) || (ifs.eof()))
		ERROR("MTP_basis::read_params_file: Can't load MTP basis functions");

	string tmpstr;

	// ignore everything until alpha_moments_count
	while (tmpstr != "alpha_moments_count")
	{
		ifs >> tmpstr;
		if (ifs.eof() || ifs.fail())
			ERROR("Error reading .mtp file");
	}

	if (tmpstr != "alpha_moments_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_moments_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_basic_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	alpha_index_basic = new int[alpha_index_basic_count][4];	// !!!
	if (alpha_index_basic == nullptr)
		ERROR("malloc error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_basic[i][0] >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}
	ifs.ignore(1000, '\n');

	// Cheking consistency (radial_funcs_count-1 < max(alpha_index_basic[i][0]))
	int max_ind = 0;
	for (int i=0; i<alpha_index_basic_count; i++)
		max_ind = __max(max_ind, alpha_index_basic[i][0]);
	if (max_ind+1 > radial_funcs_count)
		ERROR(".mtp file is inconsistent. Wrong radial_funcs_count?");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_times_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	alpha_index_times = new int[alpha_index_times_count][4];	///!!!
	if (alpha_index_times == nullptr)
		ERROR("malloc error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
	for (int i = 0; i < alpha_index_times_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_scalar_moments")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_scalar_moments;
	if (alpha_index_times_count < 0)
		ERROR("Error reading .mtp file");

	alpha_moment_mapping = new int[alpha_scalar_moments];
	if (alpha_moment_mapping == nullptr)
		ERROR("malloc error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_moment_mapping")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
	for (int i = 0; i < alpha_scalar_moments; i++)
	{
		char tmpch = ' ';
		ifs >> alpha_moment_mapping[i] >> tmpch;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}
	ifs.ignore(1000, '\n');

	alpha_count = alpha_scalar_moments + 1;
}

void MTP::WriteMTPBasis(ofstream& ofs)
{
	ofs << "MTP\n"
		<< "version = 1.1.0\n";
	ofs << "potential_name = " << pot_desc << endl;
	ofs << "species_count = 1" << endl;

	p_RadialBasis->WriteRadialBasis(ofs);

	// radial_funcs_count writing
	ofs << "\tradial_funcs_count = " << radial_funcs_count << '\n';

	// radial_coeffs writing block
	ofs.setf(ios::scientific);
	ofs.precision(15);
	ofs << "\tradial_coeffs\n"
		<< "\t\t0-0\n";
	for (int i=0; i<radial_funcs_count; i++)
	{
		ofs << "\t\t\t{";
		for (int j=0; j<p_RadialBasis->rb_size-1; j++)
			ofs << radial_coeffs(i, j) << ", ";
		ofs << radial_coeffs(i, p_RadialBasis->rb_size-1) << "}\n";
	}

	// MTP writing block
	ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
	ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';

	ofs << "alpha_index_basic = {";
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		ofs << '{'
			<< alpha_index_basic[i][0] << ", "
			<< alpha_index_basic[i][1] << ", "
			<< alpha_index_basic[i][2] << ", "
			<< alpha_index_basic[i][3] << "}";
		if (i < alpha_index_basic_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';

	ofs << "alpha_index_times = {";
	for (int i = 0; i < alpha_index_times_count; i++)
	{
		ofs << '{'
			<< alpha_index_times[i][0] << ", "
			<< alpha_index_times[i][1] << ", "
			<< alpha_index_times[i][2] << ", "
			<< alpha_index_times[i][3] << "}";
		if (i < alpha_index_times_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

	ofs << "alpha_moment_mapping = {";
	for (int i = 0; i < alpha_scalar_moments-1; i++)
		ofs << alpha_moment_mapping[i] << ", ";
	ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";
}	  

void MTP::CalcBasisFuncs(const Neighborhood& nbh)
{
	FillWithZero(moment_vals);

	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(nbh.dists[j]);

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}

		for (int i = 0; i < alpha_index_basic_count; i++)
		{
			double* p_rb_vals = &p_RadialBasis->rb_vals[0];
			double val = 0.0;
			for (int rb_ind=0; rb_ind<p_RadialBasis->rb_size; rb_ind++)
				val += radial_coeffs(alpha_index_basic[i][0], rb_ind) * p_rb_vals[rb_ind];
			//double val = p_RadialBasis->rb_vals[alpha_index_basic[i][0]];

			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
	}

	basis_vals[0] = 1.0;
	for (int i = 0; i < alpha_scalar_moments; i++)
		basis_vals[i+1] = moment_vals[alpha_moment_mapping[i]];
}

void MTP::CalcBasisFuncsDers(const Neighborhood& nbh)
{
	if (nbh.count != moment_ders.size2)
		moment_ders.resize(alpha_moments_count, nbh.count, 3);

	FillWithZero(moment_vals);
	moment_ders.set(0);

	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the nbh
		p_RadialBasis->RB_Calc(nbh.dists[j]);

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}

		for (int i = 0; i < alpha_index_basic_count; i++)
		{
			double* p_rb_vals = &p_RadialBasis->rb_vals[0];
			double* p_rb_ders = &p_RadialBasis->rb_ders[0];
			double val = 0.0;
			double der = 0.0;
			for (int rb_ind=0; rb_ind<p_RadialBasis->rb_size; rb_ind++)
			{
				val += radial_coeffs(alpha_index_basic[i][0], rb_ind) * p_rb_vals[rb_ind];
				der += radial_coeffs(alpha_index_basic[i][0], rb_ind) * p_rb_ders[rb_ind];
			}
			//double val = p_RadialBasis->rb_vals[alpha_index_basic[i][0]];
			//double der = p_RadialBasis->rb_ders[alpha_index_basic[i][0]];

			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_ders(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_ders(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_ders(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_ders(i, j, 0) += val * alpha_index_basic[i][1]
					* coords_powers_[alpha_index_basic[i][1] - 1][0]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_ders(i, j, 1) += val * alpha_index_basic[i][2]
					* pow0
					* coords_powers_[alpha_index_basic[i][2] - 1][1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_ders(i, j, 2) += val * alpha_index_basic[i][3]
					* pow0
					* pow1
					* coords_powers_[alpha_index_basic[i][3] - 1][2];
			}
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
		for (int j = 0; j < nbh.count; j++) {
			for (int a = 0; a < 3; a++) {
				moment_ders(alpha_index_times[i][3], j, a) += val2 * (moment_ders(alpha_index_times[i][0], j, a) * val1 + val0 * moment_ders(alpha_index_times[i][1], j, a));
			}
		}
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	basis_vals[0] = 1.0; // setting the constant basis function
	if (basis_ders.size2 != nbh.count) // TODO: remove this check?
		basis_ders.resize(alpha_count, nbh.count, 3);
	memset(&basis_ders(0, 0, 0), 0, 3*nbh.count*sizeof(double));
	for (int i = 0; i < alpha_scalar_moments; i++) {
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
		memcpy(&basis_ders(1+i, 0, 0), &moment_ders(alpha_moment_mapping[i], 0, 0), 3*nbh.count*sizeof(double));
	}
}

void MTP::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	if (nbh.count != buff_site_energy_ders_.size())
		buff_site_energy_ders_.resize(nbh.count);

	if (nbh.count != moment_jacobian_.size2)
		moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

	FillWithZero(moment_vals);
	moment_jacobian_.set(0);
	FillWithZero(buff_site_energy_ders_);

	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
		p_RadialBasis->RB_Calc(nbh.dists[j]);

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}

		for (int i = 0; i < alpha_index_basic_count; i++)
		{
			double* p_rb_vals = &p_RadialBasis->rb_vals[0];
			double* p_rb_ders = &p_RadialBasis->rb_ders[0];
			double val = 0.0;
			double der = 0.0;
			for (int rb_ind=0; rb_ind<p_RadialBasis->rb_size; rb_ind++)
			{
				val += radial_coeffs(alpha_index_basic[i][0], rb_ind) * p_rb_vals[rb_ind];
				der += radial_coeffs(alpha_index_basic[i][0], rb_ind) * p_rb_ders[rb_ind];
			}
			//double val = p_RadialBasis->rb_vals[alpha_index_basic[i][0]];
			//double der = p_RadialBasis->rb_ders[alpha_index_basic[i][0]];

			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
					* coords_powers_[alpha_index_basic[i][1]-1][0]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
					* pow0
					* coords_powers_[alpha_index_basic[i][2]-1][1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
					* pow0
					* pow1
					* coords_powers_[alpha_index_basic[i][3]-1][2];
			}
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
	}

	// convolving with coefficients
	buff_site_energy_ = regress_coeffs[0];
	for (int i = 0; i < alpha_scalar_moments; i++)
		buff_site_energy_ += regress_coeffs[1 + i] * moment_vals[alpha_moment_mapping[i]];

	// Backpropagation starts

	// Backpropagation step 1: site energy derivative is the corresponding linear combination
	memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

	for (int i = 0; i < alpha_scalar_moments; i++)
		site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = regress_coeffs[1 + i];
	// SAME BUT UNSAFE:
	// memcpy(&site_energy_ders_wrt_moments_[0], &regress_coeffs[1],
	//		alpha_scalar_moments * sizeof(site_energy_ders_wrt_moments_[0]));

	// Backpropagation step 2: expressing through basic moments:
	for (int i = alpha_index_times_count - 1; i >= 0; i--) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];

		site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
			site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
			* val2 * val0;
		site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
			site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
			* val2 * val1;
	}

	// Backpropagation step 3: multiply by the Jacobian:
	for (int i = 0; i < alpha_index_basic_count; i++)
		for (int j = 0; j < nbh.count; j++)
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a);
}

//! Calculates the basis function values and their derivatives for given atomic Neighborhood.

void MTP::AddSiteEnergyGrad(const Neighborhood & nbh, std::vector<double>& out_accumul_arr)
{
	out_accumul_arr.resize(CoeffCount());
	CalcBasisFuncs(nbh);
	for (int i=0; i<out_accumul_arr.size(); i++)
		out_accumul_arr[i] += basis_vals[i];
}

void MTP::CalcSiteEnergyGrad(const Neighborhood & nbh, std::vector<double>& out_se_grad)
{
	out_se_grad.resize(CoeffCount());
	CalcBasisFuncs(nbh);
	memcpy(&out_se_grad[0], &basis_vals[0], alpha_count*sizeof(double));
}

void MTP::AccumulateCombinationGrad(const Neighborhood & nbh, 
									std::vector<double>& out_grad_accumulator, 
									const double se_weight, 
									const Vector3 * se_ders_weights)
{
	Warning("MTP: Function \"MTP::AccumulateSiteEnergyGrads()\" \
			 should not be called for linear potential. It is not optimal!");

	out_grad_accumulator.resize(CoeffCount());
	buff_site_energy_ders_.resize(nbh.count);

	FillWithZero(buff_site_energy_ders_);
	buff_site_energy_ = 0.0;

	CalcBasisFuncsDers(nbh);
	for (int i=0; i<alpha_count; i++)
	{
		out_grad_accumulator[i] += se_weight*basis_vals[i];
		buff_site_energy_ += basis_vals[i];
		for (int j=0; j<nbh.count; j++)
			for (int a=0; a<3; a++)
			{
				buff_site_energy_ders_[j][a] += basis_ders(i, j, a);
				if (se_ders_weights != nullptr)
					out_grad_accumulator[i] += basis_ders(i, j, a) * se_ders_weights[i][a];
			}
	}
}

void MTP::ReadRegressCoeffs(ifstream& ifs)
{
	regress_coeffs.resize(alpha_count);

	std::string tmpstr;
	ifs >> tmpstr;
	if (tmpstr == "species_coeffs")
	{
		ifs.ignore(1000, '{');
		ifs >> regress_coeffs[0];
		ifs.ignore(1000, '\n');

		ifs >> tmpstr;
		if (tmpstr != "moment_coeffs")
			ERROR("Error reading .mtp file");

		ifs.ignore(1000, '{');
		char foo = ' ';
		for (int i = 1; i < alpha_count; i++)
			ifs >> regress_coeffs[i] >> foo;
	}
	else
		Warning("MTP: parameters have not been loaded");
}

void MTP::WriteRegressCoeffs(ofstream& ofs)
{
	ofs.setf(ios::scientific);
	ofs.precision(15);

	ofs << "species_coeffs = {" << regress_coeffs[0] << "}\n";
	ofs << "moment_coeffs = {";
	for (int i = 1; i < alpha_count; i++)
	{
		if (i != 1)
			ofs << ", ";
		ofs << regress_coeffs[i];
	}
	ofs << '}';
}

void MTP::Load(const std::string& filename)
{
	MemFree();

	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"MTP::load(): Cannot open " + filename);
	
	char tmpline[1000];
	string tmpstr;

	ifs.getline(tmpline, 1000);
	int len = (int)((string)tmpline).length();
	if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
		tmpline[len - 1] = '\0';
	if ((string)tmpline == "MTP")
	{
		// version reading block
		ifs.getline(tmpline, 1000);
		len = (int)((string)tmpline).length();
		if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
			tmpline[len - 1] = '\0';

		if ((string)tmpline != "version = 1.1.0")
			ERROR("MTP file must have version \"1.1.0\"");

		// name/description reading block
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		// species count reading block	
		if (tmpstr != "species_count")
			ERROR("Error reading .mtp file");
		ifs.ignore(2);
		int spec_cnt=-1;
		ifs >> spec_cnt;
		if (spec_cnt != 1)
			Warning("MTP: MTP file contains data for multicomponent MTP. \
					 MTP will be constructed for the first component");

		// tag reading block
		ifs >> tmpstr;

		if (tmpstr == "potential_tag") // optional 
		{
			ifs.ignore(2);
			ifs.getline(tmpline, 1000);
			tag = tmpline;
			ifs >> tmpstr;
		}

		// RadialBasis reading and initialization block
		if (tmpstr != "radial_basis_type")
			ERROR("Error reading .mtp file");
		ifs.ignore(2);
		ifs >> tmpstr;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
		if (tmpstr == "RBShapeev")
			p_RadialBasis = new RadialBasis_Shapeev(ifs);
		else if (tmpstr == "RBChebyshev")
			p_RadialBasis = new RadialBasis_Chebyshev(ifs);
		else if (tmpstr == "RBTaylor")
			p_RadialBasis = new RadialBasis_Taylor(ifs);

		// reading radial coeffs
		ReadRadialCoeffs(ifs);

		// reading MTP basis
		ReadMTPBasis(ifs);

		// reading regresion coeffs
		ReadRegressCoeffs(ifs);
	}
	else if ((string)tmpline == "version = 1.0.2")
	{
		ReadRadialBasisOldFormat(ifs);

		ReadMTPBasis(ifs);
		Message("MTP basis loaded from \"" + filename + "\"");

		// optional potential_name reading
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		regress_coeffs.resize(alpha_count);

		// regress_coeffs reading		
		if (tmpstr == "coeffs")
		{

			ifs.ignore(10, '{');
			char foo = ' ';
			for (int i = 0; i < alpha_count; i++)
				ifs >> regress_coeffs[i] >> foo;
		}
		else
			Warning("MTP: parameters have not been loaded");
	}
	else
		ERROR("MTP file must have version \"1.0.2\" or \"1.0.3\" or \"1.1.0\"");

	ifs.close();

	MemAlloc();
}

void MTP::Save(const std::string& filename)
{
	ofstream ofs(filename);

	if (!ofs.is_open())
		ERROR((string)"MTP::save(): Cannot open file \"" + filename + "\" for saving MTP");

	WriteMTPBasis(ofs);
	WriteRegressCoeffs(ofs);

	ofs.close();

	Message("MTP is saved to \"" + filename + "\"");
}

MTP::MTP(const std::string& filename)
{
	Load(filename);
}

MTP::MTP(const Settings& settings)
{
	Load(settings["filename"]);
}

MTP::~MTP()
{
	MemFree();
}

void MTP::CalcEFSGrads( const Configuration& cfg,
						Array1D& out_ene_cmpnts,
						Array3D& out_frc_cmpnts,
						Array3D& out_str_cmpnts)
{
	out_ene_cmpnts.resize(alpha_count);
	out_frc_cmpnts.resize(cfg.size(), 3, alpha_count);
	out_str_cmpnts.resize(3, 3, alpha_count);

	FillWithZero(out_ene_cmpnts);
	out_frc_cmpnts.set(0);
	out_str_cmpnts.set(0);

	Neighborhoods neighborhoods(cfg, CutOff());
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];

		CalcBasisFuncsDers(nbh);

		for (int i = 0; i < alpha_count; i++)
			out_ene_cmpnts[i] += basis_vals[i];

		for (int i = 0; i < alpha_count; i++)
			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++) {
					out_frc_cmpnts(ind, a, i) += basis_ders(i,j,a);
					out_frc_cmpnts(nbh.inds[j], a ,i) -= basis_ders(i, j, a);
				}

		for (int i = 0; i < alpha_count; i++)
			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						out_str_cmpnts(a,b,i) -= basis_ders(i, j, a) * nbh.vecs[j][b];
	}
}

void MTP::CalcEFSDebug(Configuration& cfg)
{
	ResetEFS(cfg);

	Array1D energy_cmpnts;
	Array3D forces_cmpnts;
	Array3D stress_cmpnts;

	CalcEFSGrads(cfg, energy_cmpnts, forces_cmpnts, stress_cmpnts);

	cfg.energy += regress_coeffs[0] * energy_cmpnts[0];
	for (int i = 1; i < alpha_count; i++)
		cfg.energy += regress_coeffs[i] * energy_cmpnts[i];

	for (int ind = 0; ind < cfg.size(); ind++)
		for (int i = 1; i < alpha_count; i++)
			for (int a = 0; a < 3; a++)
				cfg.force(ind, a) += regress_coeffs[i] * forces_cmpnts(ind, a, i);

	for (int i = 1; i < alpha_count; i++)
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				cfg.stresses[a][b] += regress_coeffs[i] * stress_cmpnts(a, b, i);

	cfg.has_energy(true);
	cfg.has_forces(true);
	cfg.has_stresses(true);
}

void MTP::CalcE(Configuration& cfg)
{
	CalcEnergyGrad(cfg, tmp_grad_accumulator_);
	cfg.energy = 0.0;
	for (int i=0; i<alpha_count; i++)
		cfg.energy += regress_coeffs[i] * tmp_grad_accumulator_[i];

	cfg.has_energy(true);
}

void MTP::CalcEFS(Configuration & cfg)
{
	AnyLocalMLIP::CalcEFS(cfg);
	cfg.features["EFS_by"] = "MTP" + to_string(alpha_count);
}

int MTP::CoeffCount()
{
	return alpha_count;
}

double * MTP::Coeff()
{
	return regress_coeffs.data();
}

double MTP::SiteEnergy(const Neighborhood & nbh)
{
	CalcBasisFuncs(nbh);
	double summ = 0.0;
	for (int i=0; i<alpha_count; i++)
		summ += basis_vals[i]*regress_coeffs[i];
	return summ;
}

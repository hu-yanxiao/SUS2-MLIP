/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev
 */

#include "radial_basis.h"


#include <array>
#include <cmath>

using namespace std;


void AnyRadialBasis::ReadRadialBasis(std::ifstream & ifs)
{
	if ((!ifs.is_open()) || (ifs.eof()))
		ERROR("RadialBasis::ReadRadialBasis: Can't load radial basis");

	string tmpstr;

	// reading min_dist / scaling
	ifs >> tmpstr;
	if (tmpstr == "scaling") {
		ifs.ignore(2);
		ifs >> scaling;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
		ifs >> tmpstr;
	}

	if (tmpstr != "min_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> min_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	// reading max_dist 
	ifs >> tmpstr;
	if (tmpstr != "max_dist")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> max_dist;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	// reading rb_size 
	ifs >> tmpstr;
	if (tmpstr != "radial_basis_size")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> rb_size;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	rb_vals.resize(rb_size);
	rb_ders.resize(rb_size*5);
        
}

void AnyRadialBasis::WriteRadialBasis(std::ofstream & ofs)
{
	if (!ofs.is_open())
		ERROR("RadialBasis::WriteRadialBasis: Output stream isn't open");

	ofs << "radial_basis_type = " << GetRBTypeString() << '\n';
	if (scaling != 1.0)
		ofs << "\tscaling = " << scaling << '\n';
	ofs << "\tmin_dist = " << min_dist << '\n'
		<< "\tmax_dist = " << max_dist << '\n'
		<< "\tradial_basis_size = " << rb_size << '\n';
}

AnyRadialBasis::AnyRadialBasis(double _min_dist, double _max_dist, int _size)
	: rb_size(_size), min_dist(_min_dist), max_dist(_max_dist)
{
	rb_vals.resize(rb_size);
	rb_ders.resize(rb_size*5);
	FillWithZero(rb_vals);
	FillWithZero(rb_ders);


}

AnyRadialBasis::AnyRadialBasis(std::ifstream & ifs)
{
	ReadRadialBasis(ifs);
}

namespace {

constexpr double kLaguerreMinRho = 1.0e-8;
constexpr double kLaguerrePositiveParamFloor = 1.0e-6;
constexpr double kJacobiEndpointEps = 1.0e-12;
constexpr int kJacobiMaxIndexedBlock = 5;

struct JacobiBlockSpec {
	int alpha;
	int beta;
	double linear_const;
	double linear_x;
};

struct JacobiRecurrenceCache {
	std::vector<double> coeff_const;
	std::vector<double> coeff_x;
	std::vector<double> prev_coeff;
};

const JacobiBlockSpec& JacobiBlockSpecForIndex(int k)
{
	static const std::array<JacobiBlockSpec, kJacobiMaxIndexedBlock + 1> kSpecTable = {{
		{0, 0,  0.0, 1.0},
		{1, 0,  0.5, 1.5},
		{1, 1,  0.0, 2.0},
		{2, 0,  1.0, 2.0},
		{2, 1,  0.5, 2.5},
		{2, 2,  0.0, 3.0},
	}};

	if (k < 0 || k > kJacobiMaxIndexedBlock) {
		ERROR("RBJacobi indexed basis supports only six indexed blocks: "
		      "k=0..5 -> (0,0),(1,0),(1,1),(2,0),(2,1),(2,2)");
	}
	return kSpecTable[static_cast<size_t>(k)];
}

const JacobiRecurrenceCache& JacobiRecurrenceCacheForBlock(int k, int rb_size)
{
	static thread_local std::array<JacobiRecurrenceCache, kJacobiMaxIndexedBlock + 1> kCaches;
	JacobiRecurrenceCache& cache = kCaches[static_cast<size_t>(k)];
	const size_t required_size = static_cast<size_t>(std::max(0, rb_size));
	if (cache.coeff_const.size() >= required_size)
		return cache;

	const size_t old_size = cache.coeff_const.size();
	cache.coeff_const.resize(required_size, 0.0);
	cache.coeff_x.resize(required_size, 0.0);
	cache.prev_coeff.resize(required_size, 0.0);

	const JacobiBlockSpec& spec = JacobiBlockSpecForIndex(k);
	const double alpha = static_cast<double>(spec.alpha);
	const double beta = static_cast<double>(spec.beta);
	const size_t start_order = std::max<size_t>(2, old_size);
	for (size_t order = start_order; order < required_size; ++order) {
		const double n = static_cast<double>(order);
		const double denom = 2.0 * n * (n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
		const double b = 2.0 * n + alpha + beta - 1.0;
		const double c = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta - 2.0);
		const double d = alpha * alpha - beta * beta;
		const double e = 2.0 * (n + alpha - 1.0) * (n + beta - 1.0) * (2.0 * n + alpha + beta);

		cache.coeff_const[order] = b * d / denom;
		cache.coeff_x[order] = b * c / denom;
		cache.prev_coeff[order] = e / denom;
	}

	return cache;
}

double StableSoftplus(double x)
{
	if (x > 40.0)
		return x;
	if (x < -40.0)
		return std::exp(x);
	return std::log1p(std::exp(x));
}

double StableSigmoid(double x)
{
	if (x >= 0.0) {
		const double z = std::exp(-x);
		return 1.0 / (1.0 + z);
	}
	const double z = std::exp(x);
	return z / (1.0 + z);
}

void JacobiWeightTerms(const JacobiBlockSpec& spec,
				      double x,
			      bool apply_sqrt_weight,
			      double& sqrt_weight,
			      double& log_weight_x,
			      double& log_weight_xx)
{
	sqrt_weight = 1.0;
	log_weight_x = 0.0;
	log_weight_xx = 0.0;
	if (!apply_sqrt_weight)
		return;

	const double one_minus_x = std::max(kJacobiEndpointEps, 1.0 - x);
	const double one_plus_x = std::max(kJacobiEndpointEps, 1.0 + x);

	switch (spec.alpha) {
	case 1:
		sqrt_weight *= std::sqrt(one_minus_x);
		break;
	case 2:
		sqrt_weight *= one_minus_x;
		break;
	default:
		break;
	}

	switch (spec.beta) {
	case 1:
		sqrt_weight *= std::sqrt(one_plus_x);
		break;
	case 2:
		sqrt_weight *= one_plus_x;
		break;
	default:
		break;
	}

	if (spec.alpha != 0) {
		const double inv_one_minus_x = 1.0 / one_minus_x;
		log_weight_x -= 0.5 * static_cast<double>(spec.alpha) * inv_one_minus_x;
		log_weight_xx -= 0.5 * static_cast<double>(spec.alpha) * inv_one_minus_x * inv_one_minus_x;
	}
	if (spec.beta != 0) {
		const double inv_one_plus_x = 1.0 / one_plus_x;
		log_weight_x += 0.5 * static_cast<double>(spec.beta) * inv_one_plus_x;
		log_weight_xx -= 0.5 * static_cast<double>(spec.beta) * inv_one_plus_x * inv_one_plus_x;
	}
}

void JacobiSSSCalc(AnyRadialBasis& basis,
			   double r,
			   double scal,
			   double s,
			   int k,
			   bool apply_sqrt_weight,
			   bool include_param_derivatives)
{
#ifdef MLIP_DEBUG
	if (r < basis.min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(basis.min_dist) + '\n');
	}
	if (r > basis.max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(basis.min_dist) + '\n');
	}
#endif

	const JacobiBlockSpec& spec = JacobiBlockSpecForIndex(k);
	const JacobiRecurrenceCache& recurrence_cache = JacobiRecurrenceCacheForBlock(k, basis.rb_size);

	const double z = 0.5 * scal * (r - s);
	double x = tanh(z);
	x = std::max(-1.0 + kJacobiEndpointEps, std::min(1.0 - kJacobiEndpointEps, x));

	const double sech_sq = 1.0 - x * x;
	const double dsech_sq_dz = -2.0 * x * sech_sq;
	const double x_r = 0.5 * scal * sech_sq;
	const double x_scal = 0.5 * (r - s) * sech_sq;
	const double x_s = -x_r;
	const double x_scal_r = 0.5 * sech_sq + 0.25 * scal * (r - s) * dsech_sq_dz;
	const double x_s_r = -0.25 * scal * scal * dsech_sq_dz;

	double sqrt_weight = 1.0;
	double log_weight_x = 0.0;
	double log_weight_xx = 0.0;
	JacobiWeightTerms(spec, x, apply_sqrt_weight, sqrt_weight, log_weight_x, log_weight_xx);

	double y_prev = 0.0;
	double y_prev_x = 0.0;
	double y_prev_xx = 0.0;

	double y_curr = sqrt_weight;
	double y_curr_x = sqrt_weight * log_weight_x;
	double y_curr_xx = sqrt_weight * (log_weight_x * log_weight_x + log_weight_xx);

	const double dr = r - basis.max_dist;
	const double cutoff_f = dr * dr;
	const double cutoff_der = 2.0 * dr;
	const double scaled_cutoff_f = basis.scaling * cutoff_f;
	const double scaled_cutoff_der = basis.scaling * cutoff_der;

	auto store_basis = [&](int index, double y, double y_x, double y_xx) {
		basis.rb_vals[index] = scaled_cutoff_f * y;
		basis.rb_ders[index] = scaled_cutoff_der * y + scaled_cutoff_f * y_x * x_r;
		if (!include_param_derivatives)
			return;

		basis.rb_ders[index + basis.rb_size] = scaled_cutoff_f * y_x * x_scal;
		basis.rb_ders[index + 2 * basis.rb_size] =
			scaled_cutoff_der * y_x * x_scal
			+ scaled_cutoff_f * (y_xx * x_r * x_scal + y_x * x_scal_r);
		basis.rb_ders[index + 3 * basis.rb_size] = scaled_cutoff_f * y_x * x_s;
		basis.rb_ders[index + 4 * basis.rb_size] =
			scaled_cutoff_der * y_x * x_s
			+ scaled_cutoff_f * (y_xx * x_r * x_s + y_x * x_s_r);
	};

	store_basis(0, y_curr, y_curr_x, y_curr_xx);
	if (basis.rb_size == 1)
		return;

	const double linear = spec.linear_const + spec.linear_x * x;
	const double linear_x = spec.linear_x;

	double y_next = linear * y_curr;
	double y_next_x = linear_x * y_curr + linear * y_curr_x;
	double y_next_xx = 2.0 * linear_x * y_curr_x + linear * y_curr_xx;

	store_basis(1, y_next, y_next_x, y_next_xx);

	y_prev = y_curr;
	y_prev_x = y_curr_x;
	y_prev_xx = y_curr_xx;
	y_curr = y_next;
	y_curr_x = y_next_x;
	y_curr_xx = y_next_xx;

	for (int order = 2; order < basis.rb_size; ++order) {
		const double coeff = recurrence_cache.coeff_const[order] + recurrence_cache.coeff_x[order] * x;
		const double coeff_x = recurrence_cache.coeff_x[order];
		const double prev_coeff = recurrence_cache.prev_coeff[order];

		y_next = coeff * y_curr - prev_coeff * y_prev;
		y_next_x = coeff_x * y_curr + coeff * y_curr_x - prev_coeff * y_prev_x;
		y_next_xx = 2.0 * coeff_x * y_curr_x + coeff * y_curr_xx - prev_coeff * y_prev_xx;

		store_basis(order, y_next, y_next_x, y_next_xx);

		y_prev = y_curr;
		y_prev_x = y_curr_x;
		y_prev_xx = y_curr_xx;
		y_curr = y_next;
		y_curr_x = y_next_x;
		y_curr_xx = y_next_xx;
	}
}

void LaguerreLog1pCalc(AnyRadialBasis& basis,
			    double r,
			    double scal,
			    double s,
			    bool apply_exponential_envelope,
			    bool include_param_derivatives)
{
#ifdef MLIP_DEBUG
	if (r < basis.min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(basis.min_dist) + '\n');
	}
	if (r > basis.max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(basis.min_dist) + '\n');
	}
#endif

	const bool rho_is_active = s > kLaguerreMinRho;
	const double rho = rho_is_active ? s : kLaguerreMinRho;
	const double log_term = log1p(r / rho);
	const double u = scal * log_term;
	const double u_r = scal / (rho + r);
	const double u_scal = log_term;
	const double u_scal_r = 1.0 / (rho + r);
	const double u_rho = rho_is_active ? (-scal * r / (rho * (rho + r))) : 0.0;
	const double u_rho_r = rho_is_active ? (-scal / ((rho + r) * (rho + r))) : 0.0;

	const double dr = r - basis.max_dist;
	const double cutoff_f = dr * dr;
	const double cutoff_der = 2.0 * dr;
	const double exp_factor = apply_exponential_envelope ? exp(-0.5 * u) : 1.0;

	double phi_prev = 0.0;
	double dphi_prev = 0.0;
	double dphi_scal_prev = 0.0;
	double dphi_scal_r_prev = 0.0;
	double dphi_rho_prev = 0.0;
	double dphi_rho_r_prev = 0.0;

	double phi_curr = basis.scaling * cutoff_f * exp_factor;
	double dphi_curr = basis.scaling * cutoff_der * exp_factor;
	double dphi_scal_curr = 0.0;
	double dphi_scal_r_curr = 0.0;
	double dphi_rho_curr = 0.0;
	double dphi_rho_r_curr = 0.0;

	if (apply_exponential_envelope) {
		dphi_curr -= 0.5 * u_r * phi_curr;
		if (include_param_derivatives) {
			dphi_scal_curr = -0.5 * u_scal * phi_curr;
			dphi_scal_r_curr = -0.5 * u_scal_r * phi_curr - 0.5 * u_scal * dphi_curr;
			dphi_rho_curr = -0.5 * u_rho * phi_curr;
			dphi_rho_r_curr = -0.5 * u_rho_r * phi_curr - 0.5 * u_rho * dphi_curr;
		}
	}

	basis.rb_vals[0] = phi_curr;
	basis.rb_ders[0] = dphi_curr;
	if (include_param_derivatives) {
		basis.rb_ders[basis.rb_size] = dphi_scal_curr;
		basis.rb_ders[2 * basis.rb_size] = dphi_scal_r_curr;
		basis.rb_ders[3 * basis.rb_size] = dphi_rho_curr;
		basis.rb_ders[4 * basis.rb_size] = dphi_rho_r_curr;
	}

	for (int n = 0; n < basis.rb_size - 1; ++n) {
		const double inv_np1 = 1.0 / (n + 1.0);
		const double coeff = (2.0 * n + 1.0 - u) * inv_np1;
		const double prev_coeff = n * inv_np1;

		const double phi_next = coeff * phi_curr - prev_coeff * phi_prev;
		const double dphi_next = -u_r * inv_np1 * phi_curr + coeff * dphi_curr - prev_coeff * dphi_prev;

		basis.rb_vals[n + 1] = phi_next;
		basis.rb_ders[n + 1] = dphi_next;

		double dphi_scal_next = 0.0;
		double dphi_scal_r_next = 0.0;
		double dphi_rho_next = 0.0;
		double dphi_rho_r_next = 0.0;

		if (include_param_derivatives) {
			dphi_scal_next = -u_scal * inv_np1 * phi_curr + coeff * dphi_scal_curr - prev_coeff * dphi_scal_prev;
			dphi_scal_r_next =
				(-u_scal_r * phi_curr - u_scal * dphi_curr - u_r * dphi_scal_curr) * inv_np1
				+ coeff * dphi_scal_r_curr - prev_coeff * dphi_scal_r_prev;
			dphi_rho_next = -u_rho * inv_np1 * phi_curr + coeff * dphi_rho_curr - prev_coeff * dphi_rho_prev;
			dphi_rho_r_next =
				(-u_rho_r * phi_curr - u_rho * dphi_curr - u_r * dphi_rho_curr) * inv_np1
				+ coeff * dphi_rho_r_curr - prev_coeff * dphi_rho_r_prev;

			basis.rb_ders[n + 1 + basis.rb_size] = dphi_scal_next;
			basis.rb_ders[n + 1 + 2 * basis.rb_size] = dphi_scal_r_next;
			basis.rb_ders[n + 1 + 3 * basis.rb_size] = dphi_rho_next;
			basis.rb_ders[n + 1 + 4 * basis.rb_size] = dphi_rho_r_next;
		}

		phi_prev = phi_curr;
		dphi_prev = dphi_curr;
		dphi_scal_prev = dphi_scal_curr;
		dphi_scal_r_prev = dphi_scal_r_curr;
		dphi_rho_prev = dphi_rho_curr;
		dphi_rho_r_prev = dphi_rho_r_curr;

		phi_curr = phi_next;
		dphi_curr = dphi_next;
		dphi_scal_curr = dphi_scal_next;
		dphi_scal_r_curr = dphi_scal_r_next;
		dphi_rho_curr = dphi_rho_next;
		dphi_rho_r_curr = dphi_rho_r_next;
	}
}

void LaguerreLog1pPositiveCalc(AnyRadialBasis& basis,
			       double r,
			       double scal_raw,
			       double s_raw,
			       bool apply_exponential_envelope,
			       bool include_param_derivatives)
{
	const double scal = kLaguerrePositiveParamFloor + StableSoftplus(scal_raw);
	const double s = kLaguerrePositiveParamFloor + StableSoftplus(s_raw);
	LaguerreLog1pCalc(basis, r, scal, s, apply_exponential_envelope, include_param_derivatives);

	if (include_param_derivatives) {
		const double dscal_draw = StableSigmoid(scal_raw);
		const double ds_draw = StableSigmoid(s_raw);
		for (int xi = 0; xi < basis.rb_size; ++xi) {
			basis.rb_ders[xi + basis.rb_size] *= dscal_draw;
			basis.rb_ders[xi + 2 * basis.rb_size] *= dscal_draw;
			basis.rb_ders[xi + 3 * basis.rb_size] *= ds_draw;
			basis.rb_ders[xi + 4 * basis.rb_size] *= ds_draw;
		}
	}
}

}  // namespace

void RadialBasis_Shapeev::InitShapeevRB()
{
#ifdef MLIP_DEBUG
	if (rb_size > ALLOCATED_DEGREE) {
		ERROR("error: RadialBasis::MAX_DEGREE > RadialBasis::ALLOCATED_DEGREE");
	}
#endif

	rb_coeffs[0][0] = (sqrt(21)*pow(max_dist, 2)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][1] = -((sqrt(21)*max_dist*pow(min_dist, 2)) / pow(max_dist - min_dist, 2));
	rb_coeffs[0][2] = (sqrt(21)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 2));
	rb_coeffs[0][3] = 0;
	rb_coeffs[0][4] = 0;
	rb_coeffs[0][5] = 0;
	rb_coeffs[0][6] = 0;
	rb_coeffs[0][7] = 0;
	rb_coeffs[0][8] = 0;
	rb_coeffs[0][9] = 0;
	rb_coeffs[0][10] = 0;
	rb_coeffs[0][11] = 0;
	rb_coeffs[0][12] = 0;
	rb_coeffs[0][13] = 0;
	rb_coeffs[1][0] = (-3 * sqrt(7)*pow(max_dist, 2)*pow(min_dist, 2)*(max_dist + 3 * min_dist)) / (4.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][1] = (9 * sqrt(7)*max_dist*pow(min_dist, 2)*(max_dist + min_dist)) / (2.*pow(max_dist - min_dist, 3));
	rb_coeffs[1][2] = (9 * sqrt(7)*pow(min_dist, 2)*(3 * max_dist + min_dist)) / (4.*pow(-max_dist + min_dist, 3));
	rb_coeffs[1][3] = (3 * sqrt(7)*pow(min_dist, 2)) / pow(max_dist - min_dist, 3);
	rb_coeffs[1][4] = 0;
	rb_coeffs[1][5] = 0;
	rb_coeffs[1][6] = 0;
	rb_coeffs[1][7] = 0;
	rb_coeffs[1][8] = 0;
	rb_coeffs[1][9] = 0;
	rb_coeffs[1][10] = 0;
	rb_coeffs[1][11] = 0;
	rb_coeffs[1][12] = 0;
	rb_coeffs[1][13] = 0;
	rb_coeffs[2][0] = (sqrt(3.6666666666666665)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 2) + 7 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][1] = -((sqrt(3.6666666666666665)*max_dist*pow(min_dist, 2)*(11 * pow(max_dist, 2) + 35 * max_dist*min_dist + 14 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][2] = (sqrt(3.6666666666666665)*pow(min_dist, 2)*(34 * pow(max_dist, 2) + 49 * max_dist*min_dist + 7 * pow(min_dist, 2))) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][3] = -((sqrt(33)*pow(min_dist, 2)*(13 * max_dist + 7 * min_dist)) / pow(max_dist - min_dist, 4));
	rb_coeffs[2][4] = (5 * sqrt(33)*pow(min_dist, 2)) / pow(max_dist - min_dist, 4);
	rb_coeffs[2][5] = 0;
	rb_coeffs[2][6] = 0;
	rb_coeffs[2][7] = 0;
	rb_coeffs[2][8] = 0;
	rb_coeffs[2][9] = 0;
	rb_coeffs[2][10] = 0;
	rb_coeffs[2][11] = 0;
	rb_coeffs[2][12] = 0;
	rb_coeffs[2][13] = 0;
	rb_coeffs[3][0] = (-3 * sqrt(6.5)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 3) + 12 * pow(max_dist, 2)*min_dist + 28 * max_dist*pow(min_dist, 2) + 14 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][1] = (3 * sqrt(6.5)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 3) + 104 * pow(max_dist, 2)*min_dist + 126 * max_dist*pow(min_dist, 2) + 28 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][2] = (3 * sqrt(6.5)*pow(min_dist, 2)*(43 * pow(max_dist, 3) + 141 * pow(max_dist, 2)*min_dist + 84 * max_dist*pow(min_dist, 2) + 7 * pow(min_dist, 3))) / (2.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][3] = (15 * sqrt(6.5)*pow(min_dist, 2)*(18 * pow(max_dist, 2) + 30 * max_dist*min_dist + 7 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][4] = (165 * sqrt(6.5)*pow(min_dist, 2)*(3 * max_dist + 2 * min_dist)) / (4.*pow(-max_dist + min_dist, 5));
	rb_coeffs[3][5] = (165 * sqrt(6.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 5));
	rb_coeffs[3][6] = 0;
	rb_coeffs[3][7] = 0;
	rb_coeffs[3][8] = 0;
	rb_coeffs[3][9] = 0;
	rb_coeffs[3][10] = 0;
	rb_coeffs[3][11] = 0;
	rb_coeffs[3][12] = 0;
	rb_coeffs[3][13] = 0;
	rb_coeffs[4][0] = (sqrt(0.6)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 4) + 90 * pow(max_dist, 3)*min_dist + 360 * pow(max_dist, 2)*pow(min_dist, 2) + 420 * max_dist*pow(min_dist, 3) + 126 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][1] = (-3 * sqrt(0.6)*max_dist*pow(min_dist, 2)*(20 * pow(max_dist, 4) + 195 * pow(max_dist, 3)*min_dist + 450 * pow(max_dist, 2)*pow(min_dist, 2) + 294 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][2] = (3 * sqrt(0.6)*pow(min_dist, 2)*(295 * pow(max_dist, 4) + 1680 * pow(max_dist, 3)*min_dist + 2232 * pow(max_dist, 2)*pow(min_dist, 2) + 756 * max_dist*pow(min_dist, 3) + 42 * pow(min_dist, 4))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][3] = (-22 * sqrt(0.6)*pow(min_dist, 2)*(65 * pow(max_dist, 3) + 216 * pow(max_dist, 2)*min_dist + 153 * max_dist*pow(min_dist, 2) + 21 * pow(min_dist, 3))) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][4] = (33 * sqrt(0.6)*pow(min_dist, 2)*(137 * pow(max_dist, 2) + 246 * max_dist*min_dist + 72 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][5] = (-429 * sqrt(0.6)*pow(min_dist, 2)*(4 * max_dist + 3 * min_dist)) / pow(max_dist - min_dist, 6);
	rb_coeffs[4][6] = (1001 * sqrt(0.6)*pow(min_dist, 2)) / (2.*pow(max_dist - min_dist, 6));
	rb_coeffs[4][7] = 0;
	rb_coeffs[4][8] = 0;
	rb_coeffs[4][9] = 0;
	rb_coeffs[4][10] = 0;
	rb_coeffs[4][11] = 0;
	rb_coeffs[4][12] = 0;
	rb_coeffs[4][13] = 0;
	rb_coeffs[5][0] = -(sqrt(62.333333333333336)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 5) + 25 * pow(max_dist, 4)*min_dist + 150 * pow(max_dist, 3)*pow(min_dist, 2) + 300 * pow(max_dist, 2)*pow(min_dist, 3) + 210 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][1] = (sqrt(62.333333333333336)*max_dist*pow(min_dist, 2)*(16 * pow(max_dist, 5) + 225 * pow(max_dist, 4)*min_dist + 825 * pow(max_dist, 3)*pow(min_dist, 2) + 1020 * pow(max_dist, 2)*pow(min_dist, 3) + 420 * max_dist*pow(min_dist, 4) + 42 * pow(min_dist, 5))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][2] = -(sqrt(561)*pow(min_dist, 2)*(107 * pow(max_dist, 5) + 925 * pow(max_dist, 4)*min_dist + 2120 * pow(max_dist, 3)*pow(min_dist, 2) + 1580 * pow(max_dist, 2)*pow(min_dist, 3) + 350 * max_dist*pow(min_dist, 4) + 14 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][3] = (5 * sqrt(62.333333333333336)*pow(min_dist, 2)*(73 * pow(max_dist, 4) + 397 * pow(max_dist, 3)*min_dist + 555 * pow(max_dist, 2)*pow(min_dist, 2) + 228 * max_dist*pow(min_dist, 3) + 21 * pow(min_dist, 4))) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][4] = (-65 * sqrt(62.333333333333336)*pow(min_dist, 2)*(53 * pow(max_dist, 3) + 177 * pow(max_dist, 2)*min_dist + 138 * max_dist*pow(min_dist, 2) + 24 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][5] = (91 * sqrt(561)*pow(min_dist, 2)*(8 * pow(max_dist, 2) + 15 * max_dist*min_dist + 5 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][6] = (-91 * sqrt(62.333333333333336)*pow(min_dist, 2)*(31 * max_dist + 25 * min_dist)) / (4.*pow(max_dist - min_dist, 7));
	rb_coeffs[5][7] = (182 * sqrt(62.333333333333336)*pow(min_dist, 2)) / pow(max_dist - min_dist, 7);
	rb_coeffs[5][8] = 0;
	rb_coeffs[5][9] = 0;
	rb_coeffs[5][10] = 0;
	rb_coeffs[5][11] = 0;
	rb_coeffs[5][12] = 0;
	rb_coeffs[5][13] = 0;
	rb_coeffs[6][0] = (sqrt(4.071428571428571)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 6) + 33 * pow(max_dist, 5)*min_dist + 275 * pow(max_dist, 4)*pow(min_dist, 2) + 825 * pow(max_dist, 3)*pow(min_dist, 3) + 990 * pow(max_dist, 2)*pow(min_dist, 4) + 462 * max_dist*pow(min_dist, 5) + 66 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][1] = -((sqrt(4.071428571428571)*max_dist*pow(min_dist, 2)*(41 * pow(max_dist, 6) + 781 * pow(max_dist, 5)*min_dist + 4125 * pow(max_dist, 4)*pow(min_dist, 2) + 8085 * pow(max_dist, 3)*pow(min_dist, 3) + 6270 * pow(max_dist, 2)*pow(min_dist, 4) + 1782 * max_dist*pow(min_dist, 5) + 132 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8));
	rb_coeffs[6][2] = (3 * sqrt(16.285714285714285)*pow(min_dist, 2)*(89 * pow(max_dist, 6) + 1078 * pow(max_dist, 5)*min_dist + 3740 * pow(max_dist, 4)*pow(min_dist, 2) + 4785 * pow(max_dist, 3)*pow(min_dist, 3) + 2310 * pow(max_dist, 2)*pow(min_dist, 4) + 363 * max_dist*pow(min_dist, 5) + 11 * pow(min_dist, 6))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][3] = (-13 * sqrt(16.285714285714285)*pow(min_dist, 2)*(124 * pow(max_dist, 5) + 990 * pow(max_dist, 4)*min_dist + 2255 * pow(max_dist, 3)*pow(min_dist, 2) + 1815 * pow(max_dist, 2)*pow(min_dist, 3) + 495 * max_dist*pow(min_dist, 4) + 33 * pow(min_dist, 5))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][4] = (65 * sqrt(4.071428571428571)*pow(min_dist, 2)*(161 * pow(max_dist, 4) + 847 * pow(max_dist, 3)*min_dist + 1221 * pow(max_dist, 2)*pow(min_dist, 2) + 561 * max_dist*pow(min_dist, 3) + 66 * pow(min_dist, 4))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][5] = (-39 * sqrt(4.071428571428571)*pow(min_dist, 2)*(497 * pow(max_dist, 3) + 1661 * pow(max_dist, 2)*min_dist + 1375 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][6] = (26 * sqrt(16.285714285714285)*pow(min_dist, 2)*(394 * pow(max_dist, 2) + 759 * max_dist*min_dist + 275 * pow(min_dist, 2))) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][7] = (-442 * sqrt(16.285714285714285)*pow(min_dist, 2)*(13 * max_dist + 11 * min_dist)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][8] = (1326 * sqrt(16.285714285714285)*pow(min_dist, 2)) / pow(max_dist - min_dist, 8);
	rb_coeffs[6][9] = 0;
	rb_coeffs[6][10] = 0;
	rb_coeffs[6][11] = 0;
	rb_coeffs[6][12] = 0;
	rb_coeffs[6][13] = 0;
	rb_coeffs[7][0] = -(sqrt(273)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 7) + 42 * pow(max_dist, 6)*min_dist + 462 * pow(max_dist, 5)*pow(min_dist, 2) + 1925 * pow(max_dist, 4)*pow(min_dist, 3) + 3465 * pow(max_dist, 3)*pow(min_dist, 4) + 2772 * pow(max_dist, 2)*pow(min_dist, 5) + 924 * max_dist*pow(min_dist, 6) + 99 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][1] = (3 * sqrt(273)*max_dist*pow(min_dist, 2)*(17 * pow(max_dist, 7) + 420 * pow(max_dist, 6)*min_dist + 3003 * pow(max_dist, 5)*pow(min_dist, 2) + 8470 * pow(max_dist, 4)*pow(min_dist, 3) + 10395 * pow(max_dist, 3)*pow(min_dist, 4) + 5544 * pow(max_dist, 2)*pow(min_dist, 5) + 1155 * max_dist*pow(min_dist, 6) + 66 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][2] = (-3 * sqrt(273)*pow(min_dist, 2)*(278 * pow(max_dist, 7) + 4473 * pow(max_dist, 6)*min_dist + 21714 * pow(max_dist, 5)*pow(min_dist, 2) + 41965 * pow(max_dist, 4)*pow(min_dist, 3) + 34650 * pow(max_dist, 3)*pow(min_dist, 4) + 11781 * pow(max_dist, 2)*pow(min_dist, 5) + 1386 * max_dist*pow(min_dist, 6) + 33 * pow(min_dist, 7))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][3] = (7 * sqrt(273)*pow(min_dist, 2)*(917 * pow(max_dist, 6) + 10038 * pow(max_dist, 5)*min_dist + 33495 * pow(max_dist, 4)*pow(min_dist, 2) + 43780 * pow(max_dist, 3)*pow(min_dist, 3) + 23265 * pow(max_dist, 2)*pow(min_dist, 4) + 4554 * max_dist*pow(min_dist, 5) + 231 * pow(min_dist, 6))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][4] = (-105 * sqrt(273)*pow(min_dist, 2)*(259 * pow(max_dist, 5) + 1953 * pow(max_dist, 4)*min_dist + 4422 * pow(max_dist, 3)*pow(min_dist, 2) + 3740 * pow(max_dist, 2)*pow(min_dist, 3) + 1155 * max_dist*pow(min_dist, 4) + 99 * pow(min_dist, 5))) / (8.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][5] = (21 * sqrt(273)*pow(min_dist, 2)*(1624 * pow(max_dist, 4) + 8328 * pow(max_dist, 3)*min_dist + 12243 * pow(max_dist, 2)*pow(min_dist, 2) + 6050 * max_dist*pow(min_dist, 3) + 825 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][6] = (-119 * sqrt(273)*pow(min_dist, 2)*(436 * pow(max_dist, 3) + 1455 * pow(max_dist, 2)*min_dist + 1254 * max_dist*pow(min_dist, 2) + 275 * pow(min_dist, 3))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][7] = (153 * sqrt(273)*pow(min_dist, 2)*(307 * pow(max_dist, 2) + 602 * max_dist*min_dist + 231 * pow(min_dist, 2))) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][8] = (-2907 * sqrt(273)*pow(min_dist, 2)*(8 * max_dist + 7 * min_dist)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][9] = (4845 * sqrt(273)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 9));
	rb_coeffs[7][10] = 0;
	rb_coeffs[7][11] = 0;
	rb_coeffs[7][12] = 0;
	rb_coeffs[7][13] = 0;
	rb_coeffs[8][0] = (sqrt(161)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 8) + 52 * pow(max_dist, 7)*min_dist + 728 * pow(max_dist, 6)*pow(min_dist, 2) + 4004 * pow(max_dist, 5)*pow(min_dist, 3) + 10010 * pow(max_dist, 4)*pow(min_dist, 4) + 12012 * pow(max_dist, 3)*pow(min_dist, 5) + 6864 * pow(max_dist, 2)*pow(min_dist, 6) + 1716 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][1] = -(sqrt(161)*max_dist*pow(min_dist, 2)*(31 * pow(max_dist, 8) + 962 * pow(max_dist, 7)*min_dist + 8918 * pow(max_dist, 6)*pow(min_dist, 2) + 34034 * pow(max_dist, 5)*pow(min_dist, 3) + 60060 * pow(max_dist, 4)*pow(min_dist, 4) + 50622 * pow(max_dist, 3)*pow(min_dist, 5) + 19734 * pow(max_dist, 2)*pow(min_dist, 6) + 3146 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][2] = (sqrt(161)*pow(min_dist, 2)*(1241 * pow(max_dist, 8) + 25532 * pow(max_dist, 7)*min_dist + 164528 * pow(max_dist, 6)*pow(min_dist, 2) + 444444 * pow(max_dist, 5)*pow(min_dist, 3) + 553410 * pow(max_dist, 4)*pow(min_dist, 4) + 320892 * pow(max_dist, 3)*pow(min_dist, 5) + 81224 * pow(max_dist, 2)*pow(min_dist, 6) + 7436 * max_dist*pow(min_dist, 7) + 143 * pow(min_dist, 8))) / (6.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][3] = (-10 * sqrt(161)*pow(min_dist, 2)*(591 * pow(max_dist, 7) + 8463 * pow(max_dist, 6)*min_dist + 38675 * pow(max_dist, 5)*pow(min_dist, 2) + 73931 * pow(max_dist, 4)*pow(min_dist, 3) + 63635 * pow(max_dist, 3)*pow(min_dist, 4) + 24167 * pow(max_dist, 2)*pow(min_dist, 5) + 3575 * max_dist*pow(min_dist, 6) + 143 * pow(min_dist, 7))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][4] = (10 * sqrt(161)*pow(min_dist, 2)*(3150 * pow(max_dist, 6) + 32032 * pow(max_dist, 5)*min_dist + 103792 * pow(max_dist, 4)*pow(min_dist, 2) + 137566 * pow(max_dist, 3)*pow(min_dist, 3) + 77935 * pow(max_dist, 2)*pow(min_dist, 4) + 17446 * max_dist*pow(min_dist, 5) + 1144 * pow(min_dist, 6))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][5] = (-68 * sqrt(161)*pow(min_dist, 2)*(1498 * pow(max_dist, 5) + 10816 * pow(max_dist, 4)*min_dist + 24349 * pow(max_dist, 3)*pow(min_dist, 2) + 21307 * pow(max_dist, 2)*pow(min_dist, 3) + 7150 * max_dist*pow(min_dist, 4) + 715 * pow(min_dist, 5))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][6] = (34 * sqrt(161)*pow(min_dist, 2)*(6102 * pow(max_dist, 4) + 30654 * pow(max_dist, 3)*min_dist + 45656 * pow(max_dist, 2)*pow(min_dist, 2) + 23738 * max_dist*pow(min_dist, 3) + 3575 * pow(min_dist, 4))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][7] = (-1292 * sqrt(161)*pow(min_dist, 2)*(207 * pow(max_dist, 3) + 689 * pow(max_dist, 2)*min_dist + 611 * max_dist*pow(min_dist, 2) + 143 * pow(min_dist, 3))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][8] = (1615 * sqrt(161)*pow(min_dist, 2)*(131 * pow(max_dist, 2) + 260 * max_dist*min_dist + 104 * pow(min_dist, 2))) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][9] = (-3230 * sqrt(161)*pow(min_dist, 2)*(29 * max_dist + 26 * min_dist)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][10] = (17765 * sqrt(161)*pow(min_dist, 2)) / (3.*pow(max_dist - min_dist, 10));
	rb_coeffs[8][11] = 0;
	rb_coeffs[8][12] = 0;
	rb_coeffs[8][13] = 0;
	rb_coeffs[9][0] = -(sqrt(3)*pow(max_dist, 2)*pow(min_dist, 2)*(5 * pow(max_dist, 9) + 315 * pow(max_dist, 8)*min_dist + 5460 * pow(max_dist, 7)*pow(min_dist, 2) + 38220 * pow(max_dist, 6)*pow(min_dist, 3) + 126126 * pow(max_dist, 5)*pow(min_dist, 4) + 210210 * pow(max_dist, 4)*pow(min_dist, 5) + 180180 * pow(max_dist, 3)*pow(min_dist, 6) + 77220 * pow(max_dist, 2)*pow(min_dist, 7) + 15015 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][1] = (sqrt(3)*max_dist*pow(min_dist, 2)*(185 * pow(max_dist, 9) + 7035 * pow(max_dist, 8)*min_dist + 81900 * pow(max_dist, 7)*pow(min_dist, 2) + 405132 * pow(max_dist, 6)*pow(min_dist, 3) + 966966 * pow(max_dist, 5)*pow(min_dist, 4) + 1171170 * pow(max_dist, 4)*pow(min_dist, 5) + 720720 * pow(max_dist, 3)*pow(min_dist, 6) + 214500 * pow(max_dist, 2)*pow(min_dist, 7) + 27027 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][2] = -(sqrt(3)*pow(min_dist, 2)*(8885 * pow(max_dist, 9) + 227115 * pow(max_dist, 8)*min_dist + 1870596 * pow(max_dist, 7)*pow(min_dist, 2) + 6703788 * pow(max_dist, 6)*pow(min_dist, 3) + 11657646 * pow(max_dist, 5)*pow(min_dist, 4) + 10180170 * pow(max_dist, 4)*pow(min_dist, 5) + 4384380 * pow(max_dist, 3)*pow(min_dist, 6) + 859716 * pow(max_dist, 2)*pow(min_dist, 7) + 63063 * max_dist*pow(min_dist, 8) + 1001 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][3] = (6 * sqrt(3)*pow(min_dist, 2)*(4265 * pow(max_dist, 8) + 77196 * pow(max_dist, 7)*min_dist + 461188 * pow(max_dist, 6)*pow(min_dist, 2) + 1206296 * pow(max_dist, 5)*pow(min_dist, 3) + 1516515 * pow(max_dist, 4)*pow(min_dist, 4) + 930930 * pow(max_dist, 3)*pow(min_dist, 5) + 266266 * pow(max_dist, 2)*pow(min_dist, 6) + 30888 * max_dist*pow(min_dist, 7) + 1001 * pow(min_dist, 8))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][4] = (-51 * sqrt(3)*pow(min_dist, 2)*(3274 * pow(max_dist, 7) + 43022 * pow(max_dist, 6)*min_dist + 187824 * pow(max_dist, 5)*pow(min_dist, 2) + 355810 * pow(max_dist, 4)*pow(min_dist, 3) + 315315 * pow(max_dist, 3)*pow(min_dist, 4) + 129129 * pow(max_dist, 2)*pow(min_dist, 5) + 22022 * max_dist*pow(min_dist, 6) + 1144 * pow(min_dist, 7))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][5] = (714 * sqrt(3)*pow(min_dist, 2)*(942 * pow(max_dist, 6) + 9054 * pow(max_dist, 5)*min_dist + 28665 * pow(max_dist, 4)*pow(min_dist, 2) + 38350 * pow(max_dist, 3)*pow(min_dist, 3) + 22737 * pow(max_dist, 2)*pow(min_dist, 4) + 5577 * max_dist*pow(min_dist, 5) + 429 * pow(min_dist, 6))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][6] = (-6783 * sqrt(3)*pow(min_dist, 2)*(258 * pow(max_dist, 5) + 1800 * pow(max_dist, 4)*min_dist + 4030 * pow(max_dist, 3)*pow(min_dist, 2) + 3614 * pow(max_dist, 2)*pow(min_dist, 3) + 1287 * max_dist*pow(min_dist, 4) + 143 * pow(min_dist, 5))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][7] = (1938 * sqrt(3)*pow(min_dist, 2)*(1545 * pow(max_dist, 4) + 7630 * pow(max_dist, 3)*min_dist + 11466 * pow(max_dist, 2)*pow(min_dist, 2) + 6188 * max_dist*pow(min_dist, 3) + 1001 * pow(min_dist, 4))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][8] = (-969 * sqrt(3)*pow(min_dist, 2)*(6905 * pow(max_dist, 3) + 22911 * pow(max_dist, 2)*min_dist + 20748 * max_dist*pow(min_dist, 2) + 5096 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][9] = (3553 * sqrt(3)*pow(min_dist, 2)*(661 * pow(max_dist, 2) + 1323 * max_dist*min_dist + 546 * pow(min_dist, 2))) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][10] = (-81719 * sqrt(3)*pow(min_dist, 2)*(23 * max_dist + 21 * min_dist)) / (2.*pow(max_dist - min_dist, 11));
	rb_coeffs[9][11] = (163438 * sqrt(3)*pow(min_dist, 2)) / pow(max_dist - min_dist, 11);
	rb_coeffs[9][12] = 0;
	rb_coeffs[9][13] = 0;
	rb_coeffs[10][0] = (3 * sqrt(0.5454545454545454)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 10) + 75 * pow(max_dist, 9)*min_dist + 1575 * pow(max_dist, 8)*pow(min_dist, 2) + 13650 * pow(max_dist, 7)*pow(min_dist, 3) + 57330 * pow(max_dist, 6)*pow(min_dist, 4) + 126126 * pow(max_dist, 5)*pow(min_dist, 5) + 150150 * pow(max_dist, 4)*pow(min_dist, 6) + 96525 * pow(max_dist, 3)*pow(min_dist, 7) + 32175 * pow(max_dist, 2)*pow(min_dist, 8) + 5005 * max_dist*pow(min_dist, 9) + 273 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][1] = (-9 * sqrt(0.5454545454545454)*max_dist*pow(min_dist, 2)*(29 * pow(max_dist, 10) + 1325 * pow(max_dist, 9)*min_dist + 18900 * pow(max_dist, 8)*pow(min_dist, 2) + 117390 * pow(max_dist, 7)*pow(min_dist, 3) + 363090 * pow(max_dist, 6)*pow(min_dist, 4) + 594594 * pow(max_dist, 5)*pow(min_dist, 5) + 525525 * pow(max_dist, 4)*pow(min_dist, 6) + 246675 * pow(max_dist, 3)*pow(min_dist, 7) + 57915 * pow(max_dist, 2)*pow(min_dist, 8) + 5915 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][2] = (9 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(822 * pow(max_dist, 10) + 25525 * pow(max_dist, 9)*min_dist + 261135 * pow(max_dist, 8)*pow(min_dist, 2) + 1195740 * pow(max_dist, 7)*pow(min_dist, 3) + 2757300 * pow(max_dist, 6)*pow(min_dist, 4) + 3360357 * pow(max_dist, 5)*pow(min_dist, 5) + 2177175 * pow(max_dist, 4)*pow(min_dist, 6) + 725010 * pow(max_dist, 3)*pow(min_dist, 7) + 113490 * pow(max_dist, 2)*pow(min_dist, 8) + 6825 * max_dist*pow(min_dist, 9) + 91 * pow(min_dist, 10))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][3] = (-255 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(397 * pow(max_dist, 9) + 8847 * pow(max_dist, 8)*min_dist + 66780 * pow(max_dist, 7)*pow(min_dist, 2) + 228228 * pow(max_dist, 6)*pow(min_dist, 3) + 392301 * pow(max_dist, 5)*pow(min_dist, 4) + 351351 * pow(max_dist, 4)*pow(min_dist, 5) + 162162 * pow(max_dist, 3)*pow(min_dist, 6) + 36270 * pow(max_dist, 2)*pow(min_dist, 7) + 3393 * max_dist*pow(min_dist, 8) + 91 * pow(min_dist, 9))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][4] = (2295 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(345 * pow(max_dist, 8) + 5676 * pow(max_dist, 7)*min_dist + 32004 * pow(max_dist, 6)*pow(min_dist, 2) + 81627 * pow(max_dist, 5)*pow(min_dist, 3) + 103285 * pow(max_dist, 4)*pow(min_dist, 4) + 66066 * pow(max_dist, 3)*pow(min_dist, 5) + 20566 * pow(max_dist, 2)*pow(min_dist, 6) + 2769 * max_dist*pow(min_dist, 7) + 117 * pow(min_dist, 8))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][5] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(444 * pow(max_dist, 7) + 5460 * pow(max_dist, 6)*min_dist + 22995 * pow(max_dist, 5)*pow(min_dist, 2) + 43225 * pow(max_dist, 4)*pow(min_dist, 3) + 39130 * pow(max_dist, 3)*pow(min_dist, 4) + 16926 * pow(max_dist, 2)*pow(min_dist, 5) + 3185 * max_dist*pow(min_dist, 6) + 195 * pow(min_dist, 7))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][6] = (20349 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(612 * pow(max_dist, 6) + 5625 * pow(max_dist, 5)*min_dist + 17475 * pow(max_dist, 4)*pow(min_dist, 2) + 23530 * pow(max_dist, 3)*pow(min_dist, 3) + 14430 * pow(max_dist, 2)*pow(min_dist, 4) + 3783 * max_dist*pow(min_dist, 5) + 325 * pow(min_dist, 6))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][7] = (-8721 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(3099 * pow(max_dist, 5) + 21025 * pow(max_dist, 4)*min_dist + 46830 * pow(max_dist, 3)*pow(min_dist, 2) + 42770 * pow(max_dist, 2)*pow(min_dist, 3) + 15925 * max_dist*pow(min_dist, 4) + 1911 * pow(min_dist, 5))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][8] = (43605 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(913 * pow(max_dist, 4) + 4444 * pow(max_dist, 3)*min_dist + 6720 * pow(max_dist, 2)*pow(min_dist, 2) + 3731 * max_dist*pow(min_dist, 3) + 637 * pow(min_dist, 4))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][9] = (-111435 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(352 * pow(max_dist, 3) + 1164 * pow(max_dist, 2)*min_dist + 1071 * max_dist*pow(min_dist, 2) + 273 * pow(min_dist, 3))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][10] = (334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(74 * pow(max_dist, 2) + 149 * max_dist*min_dist + 63 * pow(min_dist, 2))) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][11] = (-334305 * sqrt(0.5454545454545454)*pow(min_dist, 2)*(27 * max_dist + 25 * min_dist)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][12] = (1448655 * sqrt(0.5454545454545454)*pow(min_dist, 2)) / pow(max_dist - min_dist, 12);
	rb_coeffs[10][13] = 0;
	rb_coeffs[11][0] = -(sqrt(82.16666666666667)*pow(max_dist, 2)*pow(min_dist, 2)*(pow(max_dist, 11) + 88 * pow(max_dist, 10)*min_dist + 2200 * pow(max_dist, 9)*pow(min_dist, 2) + 23100 * pow(max_dist, 8)*pow(min_dist, 3) + 120120 * pow(max_dist, 7)*pow(min_dist, 4) + 336336 * pow(max_dist, 6)*pow(min_dist, 5) + 528528 * pow(max_dist, 5)*pow(min_dist, 6) + 471900 * pow(max_dist, 4)*pow(min_dist, 7) + 235950 * pow(max_dist, 3)*pow(min_dist, 8) + 62920 * pow(max_dist, 2)*pow(min_dist, 9) + 8008 * max_dist*pow(min_dist, 10) + 364 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][1] = (sqrt(82.16666666666667)*max_dist*pow(min_dist, 2)*(101 * pow(max_dist, 11) + 5456 * pow(max_dist, 10)*min_dist + 93500 * pow(max_dist, 9)*pow(min_dist, 2) + 711480 * pow(max_dist, 8)*pow(min_dist, 3) + 2762760 * pow(max_dist, 7)*pow(min_dist, 4) + 5861856 * pow(max_dist, 6)*pow(min_dist, 5) + 7002996 * pow(max_dist, 5)*pow(min_dist, 6) + 4719000 * pow(max_dist, 4)*pow(min_dist, 7) + 1746030 * pow(max_dist, 3)*pow(min_dist, 8) + 331760 * pow(max_dist, 2)*pow(min_dist, 9) + 28028 * max_dist*pow(min_dist, 10) + 728 * pow(min_dist, 11))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][2] = -(sqrt(82.16666666666667)*pow(min_dist, 2)*(1667 * pow(max_dist, 11) + 61754 * pow(max_dist, 10)*min_dist + 767360 * pow(max_dist, 9)*pow(min_dist, 2) + 4363590 * pow(max_dist, 8)*pow(min_dist, 3) + 12852840 * pow(max_dist, 7)*pow(min_dist, 4) + 20762742 * pow(max_dist, 6)*pow(min_dist, 5) + 18762744 * pow(max_dist, 5)*pow(min_dist, 6) + 9390810 * pow(max_dist, 4)*pow(min_dist, 7) + 2492490 * pow(max_dist, 3)*pow(min_dist, 8) + 318890 * pow(max_dist, 2)*pow(min_dist, 9) + 16016 * max_dist*pow(min_dist, 10) + 182 * pow(min_dist, 11))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][3] = (11 * sqrt(739.5)*pow(min_dist, 2)*(809 * pow(max_dist, 10) + 21740 * pow(max_dist, 9)*min_dist + 201990 * pow(max_dist, 8)*pow(min_dist, 2) + 871920 * pow(max_dist, 7)*pow(min_dist, 3) + 1957410 * pow(max_dist, 6)*pow(min_dist, 4) + 2395484 * pow(max_dist, 5)*pow(min_dist, 5) + 1611610 * pow(max_dist, 4)*pow(min_dist, 6) + 580840 * pow(max_dist, 3)*pow(min_dist, 7) + 104520 * pow(max_dist, 2)*pow(min_dist, 8) + 8060 * max_dist*pow(min_dist, 9) + 182 * pow(min_dist, 10))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][4] = (-1045 * sqrt(739.5)*pow(min_dist, 2)*(157 * pow(max_dist, 9) + 3156 * pow(max_dist, 8)*min_dist + 22272 * pow(max_dist, 7)*pow(min_dist, 2) + 73332 * pow(max_dist, 6)*pow(min_dist, 3) + 124852 * pow(max_dist, 5)*pow(min_dist, 4) + 113932 * pow(max_dist, 4)*pow(min_dist, 5) + 55328 * pow(max_dist, 3)*pow(min_dist, 6) + 13572 * pow(max_dist, 2)*pow(min_dist, 7) + 1482 * max_dist*pow(min_dist, 8) + 52 * pow(min_dist, 9))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][5] = (209 * sqrt(739.5)*pow(min_dist, 2)*(4569 * pow(max_dist, 8) + 69792 * pow(max_dist, 7)*min_dist + 375900 * pow(max_dist, 6)*pow(min_dist, 2) + 939400 * pow(max_dist, 5)*pow(min_dist, 3) + 1193920 * pow(max_dist, 4)*pow(min_dist, 4) + 787696 * pow(max_dist, 3)*pow(min_dist, 5) + 260988 * pow(max_dist, 2)*pow(min_dist, 6) + 39000 * max_dist*pow(min_dist, 7) + 1950 * pow(min_dist, 8))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][6] = (-1463 * sqrt(739.5)*pow(min_dist, 2)*(633 * pow(max_dist, 7) + 7383 * pow(max_dist, 6)*min_dist + 30200 * pow(max_dist, 5)*pow(min_dist, 2) + 56385 * pow(max_dist, 4)*pow(min_dist, 3) + 51870 * pow(max_dist, 3)*pow(min_dist, 4) + 23387 * pow(max_dist, 2)*pow(min_dist, 5) + 4732 * max_dist*pow(min_dist, 6) + 325 * pow(min_dist, 7))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][7] = (209 * sqrt(739.5)*pow(min_dist, 2)*(11814 * pow(max_dist, 6) + 104698 * pow(max_dist, 5)*min_dist + 320155 * pow(max_dist, 4)*pow(min_dist, 2) + 433020 * pow(max_dist, 3)*pow(min_dist, 3) + 272545 * pow(max_dist, 2)*pow(min_dist, 4) + 75166 * max_dist*pow(min_dist, 5) + 7007 * pow(min_dist, 6))) / pow(max_dist - min_dist, 13);
	rb_coeffs[11][8] = (-4807 * sqrt(739.5)*pow(min_dist, 2)*(3817 * pow(max_dist, 5) + 25300 * pow(max_dist, 4)*min_dist + 56080 * pow(max_dist, 3)*pow(min_dist, 2) + 51940 * pow(max_dist, 2)*pow(min_dist, 3) + 20020 * max_dist*pow(min_dist, 4) + 2548 * pow(min_dist, 5))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][9] = (24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(2959 * pow(max_dist, 4) + 14224 * pow(max_dist, 3)*min_dist + 21604 * pow(max_dist, 2)*pow(min_dist, 2) + 12264 * max_dist*pow(min_dist, 3) + 2184 * pow(min_dist, 4))) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][10] = (-24035 * sqrt(82.16666666666667)*pow(min_dist, 2)*(1303 * pow(max_dist, 3) + 4294 * pow(max_dist, 2)*min_dist + 4000 * max_dist*pow(min_dist, 2) + 1050 * pow(min_dist, 3))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][11] = (28405 * sqrt(82.16666666666667)*pow(min_dist, 2)*(631 * pow(max_dist, 2) + 1276 * max_dist*min_dist + 550 * pow(min_dist, 2))) / (2.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][12] = (-85215 * sqrt(739.5)*pow(min_dist, 2)*(47 * max_dist + 44 * min_dist)) / (4.*pow(max_dist - min_dist, 13));
	rb_coeffs[11][13] = (596505 * sqrt(739.5)*pow(min_dist, 2)) / (4.*pow(max_dist - min_dist, 13));
}

RadialBasis_Shapeev::RadialBasis_Shapeev(double _min_dist, double _max_dist, int _size)
	: AnyRadialBasis(_min_dist, _max_dist, _size)
{
	if (rb_size > ALLOCATED_DEGREE + 1)
		ERROR("RadialBasis error: allocated degree ecceded.");
	InitShapeevRB();
}

RadialBasis_Shapeev::RadialBasis_Shapeev(std::ifstream & ifs)
	: AnyRadialBasis(ifs)
{
	if (rb_size > ALLOCATED_DEGREE + 1)
		ERROR("RadialBasis error: allocated degree ecceded.");
	InitShapeevRB();
}

void RadialBasis_Shapeev::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	for (int xi = 0; xi < rb_size; xi++) {
		rb_vals[xi] = 0;
		rb_ders[xi] = 0;
		for (int i = -2; i <= xi; i++) {
			rb_vals[xi] += scaling * rb_coeffs[xi][i + 2] * pow(r, i);
			rb_ders[xi] += scaling * rb_coeffs[xi][i + 2] * i * pow(r, i - 1);
		}
	}
}

void RadialBasis_Chebyshev_ssss::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)*0.5;
double expm= exp(-x);
double expp= exp(x);
double x_plus = 1+x;
double ksi = -2*(exp((x_plus)*expm-1))+1;
double der = 2*exp((-1+expm)*(x_plus))*x;
double dder = -2*exp(-1-2*x+expm*(x_plus))*(expp*(-1+x)+x*x);
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;


double denom=x*x+1 ;
double sq=sqrt(denom);
double _ksi = x/sq;
double _der = 1/(denom*sq);
double _dder = -3*_ksi*_der/sq;
double _mult = _der*scal/2;
double _mult_s_r=-_dder*scal*scal/4;
double _mult_scal_r = _der/2+_dder*x/2;
double _mult_scal = _der *(r-s)/2;
double _mult_s=-_mult;
int shift=rb_size/2;



double Dr=r-max_dist;
double cutoff_f=Dr * Dr;
rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;


rb_vals[shift] = scaling * cutoff_f;
rb_ders[shift] = 0;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[shift + rb_size * 1] = 0;//scaling * 0;
rb_ders[shift + rb_size * 2] = 0;//scaling * 0;
rb_ders[shift + rb_size * 3] = 0;//scaling * 0;
rb_ders[shift + rb_size * 4] = 0;//scaling * 0;




rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);


rb_vals[1+shift] = scaling * (_ksi * cutoff_f);
rb_ders[1+shift] = scaling * (_mult * cutoff_f + 2 * _ksi * Dr);
rb_ders[1 +shift+ rb_size * 1] = scaling * _mult_scal * cutoff_f;
rb_ders[1 + shift+rb_size * 2] = scaling * (_mult_scal_r * cutoff_f + 2 * _mult_scal * Dr);
rb_ders[1 +shift +rb_size * 3] = scaling * _mult_s * cutoff_f;
rb_ders[1 +shift +rb_size * 4] = scaling * (_mult_s_r * cutoff_f + 2 * _mult_s * Dr);




         for (int i = 2; i < shift; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];

                rb_vals[i+shift] = 2 * _ksi*rb_vals[shift+i - 1] - rb_vals[shift+i - 2];
                rb_ders[i+shift] = 2 * (_mult * rb_vals[i - 1+shift] + _ksi * rb_ders[i - 1+shift]) - rb_ders[i - 2+shift];
                                rb_ders[i + rb_size+shift] = 2 * (_mult_scal * rb_vals[i - 1+shift] + _ksi * rb_ders[i - 1+ rb_size+shift]) - rb_ders[i - 2+ rb_size+shift];
                                rb_ders[i + 2 * rb_size+shift] = 2 * (_mult_scal_r * rb_vals[i - 1+shift] + _mult * rb_ders[i - 1 + rb_size+shift] + _ksi * rb_ders[i - 1 + rb_size * 2+shift] + _mult_scal * rb_ders[i - 1+shift]) - rb_ders[i - 2 + rb_size * 2+shift];
                                rb_ders[i + 3 * rb_size+shift] = 2 * (_mult_s * rb_vals[i - 1+shift] + _ksi * rb_ders[i + 3 * rb_size-1+shift])- rb_ders[i + 3 * rb_size-2+shift];
                                rb_ders[i + 4 * rb_size+shift] = 2 * (_mult_s_r * rb_vals[i - 1+shift] +_mult* rb_ders[i + 3 * rb_size - 1+shift] +_ksi * rb_ders[i + 4 * rb_size - 1+shift] + _mult_s* rb_ders[i - 1+shift]) - rb_ders[i + 4 * rb_size - 2+shift];

        }
	}




void RadialBasis_Chebyshev_sssss::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)*0.5 ;
double denom=x*x+1 ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = 1/(denom*sq);
double dder = -3*ksi*der/sq;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;



double _ksi = 2*ksi*ksi-1;
double _der = 4*ksi*der;
double _dder = 4*(der*der+ksi*dder);
double _mult = _der*scal/2;
double _mult_s_r=-_dder*scal*scal/4;
double _mult_scal_r = _der/2+_dder*x/2;
double _mult_scal = _der *(r-s)/2;
double _mult_s=-_mult;


double __ksi = ksi*ksi*ksi;
double __der = 0.75*ksi*_der;
double __dder = 6*ksi*der*der+3*ksi*ksi*dder;
double __mult = __der*scal/2;
double __mult_s_r=-__dder*scal*scal/4;
double __mult_scal_r = __der/2+__dder*x/2;
double __mult_scal = __der *(r-s)/2;
double __mult_s=-__mult;




int shift=rb_size/3;
int shift_ = 2*shift;


double Dr=r-max_dist;
double cutoff_f=Dr * Dr;
rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;


rb_vals[shift] = scaling * cutoff_f;
rb_ders[shift] = 0;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[shift + rb_size * 1] = 0;//scaling * 0;
rb_ders[shift + rb_size * 2] = 0;//scaling * 0;
rb_ders[shift + rb_size * 3] = 0;//scaling * 0;
rb_ders[shift + rb_size * 4] = 0;//scaling * 0;


rb_vals[shift_] = scaling * cutoff_f;
rb_ders[shift_] = 0;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[shift_ + rb_size * 1] = 0;//scaling * 0;
rb_ders[shift_ + rb_size * 2] = 0;//scaling * 0;
rb_ders[shift_ + rb_size * 3] = 0;//scaling * 0;
rb_ders[shift_ + rb_size * 4] = 0;//scaling * 0;



rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);


rb_vals[1+shift] = scaling * (_ksi * cutoff_f);
rb_ders[1+shift] = scaling * (_mult * cutoff_f + 2 * _ksi * Dr);
rb_ders[1 +shift+ rb_size * 1] = scaling * _mult_scal * cutoff_f;
rb_ders[1 + shift+rb_size * 2] = scaling * (_mult_scal_r * cutoff_f + 2 * _mult_scal * Dr);
rb_ders[1 +shift +rb_size * 3] = scaling * _mult_s * cutoff_f;
rb_ders[1 +shift +rb_size * 4] = scaling * (_mult_s_r * cutoff_f + 2 * _mult_s * Dr);


rb_vals[1+shift_] = scaling * (__ksi * cutoff_f);
rb_ders[1+shift_] = scaling * (__mult * cutoff_f + 2 * __ksi * Dr);
rb_ders[1 +shift_+ rb_size * 1] = scaling * __mult_scal * cutoff_f;
rb_ders[1 + shift_+rb_size * 2] = scaling * (__mult_scal_r * cutoff_f + 2 * __mult_scal * Dr);
rb_ders[1 +shift_ +rb_size * 3] = scaling * __mult_s * cutoff_f;
rb_ders[1 +shift_ +rb_size * 4] = scaling * (__mult_s_r * cutoff_f + 2 * __mult_s * Dr);




         for (int i = 2; i < shift; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];

                rb_vals[i+shift] = 2 * _ksi*rb_vals[shift+i - 1] - rb_vals[shift+i - 2];
                rb_ders[i+shift] = 2 * (_mult * rb_vals[i - 1+shift] + _ksi * rb_ders[i - 1+shift]) - rb_ders[i - 2+shift];
                                rb_ders[i + rb_size+shift] = 2 * (_mult_scal * rb_vals[i - 1+shift] + _ksi * rb_ders[i - 1+ rb_size+shift]) - rb_ders[i - 2+ rb_size+shift];
                                rb_ders[i + 2 * rb_size+shift] = 2 * (_mult_scal_r * rb_vals[i - 1+shift] + _mult * rb_ders[i - 1 + rb_size+shift] + _ksi * rb_ders[i - 1 + rb_size * 2+shift] + _mult_scal * rb_ders[i - 1+shift]) - rb_ders[i - 2 + rb_size * 2+shift];
                                rb_ders[i + 3 * rb_size+shift] = 2 * (_mult_s * rb_vals[i - 1+shift] + _ksi * rb_ders[i + 3 * rb_size-1+shift])- rb_ders[i + 3 * rb_size-2+shift];
                                rb_ders[i + 4 * rb_size+shift] = 2 * (_mult_s_r * rb_vals[i - 1+shift] +_mult* rb_ders[i + 3 * rb_size - 1+shift] +_ksi * rb_ders[i + 4 * rb_size - 1+shift] + _mult_s* rb_ders[i - 1+shift]) - rb_ders[i + 4 * rb_size - 2+shift];

                rb_vals[i+shift_] = 2 * __ksi*rb_vals[shift_+i - 1] - rb_vals[shift_+i - 2];
                rb_ders[i+shift_] = 2 * (__mult * rb_vals[i - 1+shift_] + __ksi * rb_ders[i - 1+shift_]) - rb_ders[i - 2+shift_];
                                rb_ders[i + rb_size+shift_] = 2 * (__mult_scal * rb_vals[i - 1+shift_] + __ksi * rb_ders[i - 1+ rb_size+shift_]) - rb_ders[i - 2+ rb_size+shift_];
                                rb_ders[i + 2 * rb_size+shift_] = 2 * (__mult_scal_r * rb_vals[i - 1+shift_] + __mult * rb_ders[i - 1 + rb_size+shift_] + __ksi * rb_ders[i - 1 + rb_size * 2+shift_] + __mult_scal * rb_ders[i - 1+shift_]) - rb_ders[i - 2 + rb_size * 2+shift_];
                                rb_ders[i + 3 * rb_size+shift_] = 2 * (__mult_s * rb_vals[i - 1+shift_] + __ksi * rb_ders[i + 3 * rb_size-1+shift_])- rb_ders[i + 3 * rb_size-2+shift_];
                                rb_ders[i + 4 * rb_size+shift_] = 2 * (__mult_s_r * rb_vals[i - 1+shift_] +__mult* rb_ders[i + 3 * rb_size - 1+shift_] +__ksi * rb_ders[i + 4 * rb_size - 1+shift_] + __mult_s* rb_ders[i - 1+shift_]) - rb_ders[i + 4 * rb_size - 2+shift_];
        }
	}





void RadialBasis_Bessel_sss::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');}
#endif
//int sigma = k%4;
double x=scal*(r-s)/2 ;
double ksi = tanh(x);
double der = 1-ksi*ksi;
double dder = -2*ksi*der;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
//double Dr=r-max_dist;
double Dr=r/max_dist-1;
double p=6;
double cutoff_f = 1-(p+1)*(p+2)*pow(Dr,p)*0.5+p*(p+2)*pow(Dr,p+1)-(p+1)*p*pow(Dr,p+2)*0.5 ;
double cutoff_der = p*(p+1)*(p+2)*(pow(Dr,p)-0.5*(pow(Dr,p+1)+pow(Dr,p-1)))/max_dist ;
//double cutoff_f=Dr * Dr;
double PI=3.141592654 ;
double temp=ksi+1.00001;


for (int i = 0; i < rb_size; i++) {
double N=i+1;
double w=PI*N/2 ;
double sin_=sin(w*temp);
double cos_=cos(w*temp);
double bessel=sin_/temp;
double bessel_der=w*cos_/temp-sin_/temp/temp;
double bessel_ddr=-(w*w*sin_/temp)-2*w*cos_/temp/temp+2*sin_/temp/temp/temp;
rb_vals[i]=scaling*bessel*cutoff_f;
rb_ders[i]=scaling*(bessel_der*mult*cutoff_f+bessel*cutoff_der);
rb_ders[i+rb_size * 1] = scaling * bessel_der * mult_scal * cutoff_f;
rb_ders[i+rb_size * 2] = scaling * (bessel_ddr* mult*mult_scal * cutoff_f +  bessel_der * mult_scal_r * cutoff_f
+cutoff_der*bessel_der * mult_scal);
rb_ders[i+rb_size * 3] = scaling * bessel_der * mult_s * cutoff_f;
rb_ders[i+rb_size * 4] = scaling * (bessel_ddr* mult*mult_s * cutoff_f +  bessel_der * mult_s_r * cutoff_f
+cutoff_der*bessel_der * mult_s);
}



}




void RadialBasis_Bessel_sssw::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
                        }
#endif
//int sigma = k%4;
double w=(1+k)*0.1;
double x=scal*(r-s)/2 ;
double ksi = tanh(w*x);
double der = (1-ksi*ksi)*w;
double dder = -2*ksi*der*w;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
//double Dr=r-max_dist;
double Dr=r/max_dist-1;
double p=6;
double cutoff_f = 1-(p+1)*(p+2)*pow(Dr,p)*0.5+p*(p+2)*pow(Dr,p+1)-(p+1)*p*pow(Dr,p+2)*0.5 ;
double cutoff_der = p*(p+1)*(p+2)*(pow(Dr,p)-0.5*(pow(Dr,p+1)+pow(Dr,p-1)))/max_dist ;
//double cutoff_f=Dr * Dr;
double PI=3.141592654 ;
double temp=ksi+1.00001;


for (int i = 0; i < rb_size; i++) {
double N=i+1;
double w=PI*N/2 ;
double sin_=sin(w*temp);
double cos_=cos(w*temp);
double bessel=sin_/temp;
double bessel_der=w*cos_/temp-sin_/temp/temp;
double bessel_ddr=-(w*w*sin_/temp)-2*w*cos_/temp/temp+2*sin_/temp/temp/temp;
rb_vals[i]=scaling*bessel*cutoff_f;
rb_ders[i]=scaling*(bessel_der*mult*cutoff_f+bessel*cutoff_der);
rb_ders[i+rb_size * 1] = scaling * bessel_der * mult_scal * cutoff_f;
rb_ders[i+rb_size * 2] = scaling * (bessel_ddr* mult*mult_scal * cutoff_f +  bessel_der * mult_scal_r * cutoff_f
+cutoff_der*bessel_der * mult_scal);
rb_ders[i+rb_size * 3] = scaling * bessel_der * mult_s * cutoff_f;
rb_ders[i+rb_size * 4] = scaling * (bessel_ddr* mult*mult_s * cutoff_f +  bessel_der * mult_s_r * cutoff_f
+cutoff_der*bessel_der * mult_s);
}



}



void RadialBasis_Chebyshev_Tri::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif
double Pi=3.141592654 ;
double x=scal*(r-s)*0.05*Pi;

double cos_=cos(x);
double sin_=sin(x);

double ksi=sin_;
double der=cos_;
double dder= -1.0*ksi;
//int n=k%2;
//if (n==0){
 //   ksi = cos_;
 //   der = -1.0*sin_;
//} else {
  //  ksi = sin_;
 //   der = cos_;
//}
//dder = -1.0*ksi;


double mult = der*scal*0.05*Pi;
double mult_s_r=-dder*scal*scal*0.0025*Pi*Pi;
double mult_scal_r = der*0.05*Pi+dder*scal*0.0025*Pi*(r-s)*Pi;
double mult_scal = der *(r-s)*0.05*Pi;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
				rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
				rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
				rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}




void RadialBasis_Chebyshev_sss::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)/2 ;
double ksi = tanh(x);
double der = 1-ksi*ksi;
double dder = -2*ksi*der;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);

        

         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
				rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
				rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
				rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}






const std::vector<double> RadialBasis_Chebyshev_sssw::arr = 
    {1.0, 5.0/7.0, 9.0/7.0, 3.0/7.0, 11.0/7.0, 13.0/7.0};
void RadialBasis_Chebyshev_sssw::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)/2 ;
double tanh_ = tanh(x);
double sech_ = 1-tanh_*tanh_;

double w=arr[k];
double ksi;
double der ;
double dder ;
if (w==1.0) {ksi = tanh_;
der = sech_;
dder = -2*ksi*der;}
else {ksi = pow(tanh_,w);
der = w*pow(tanh_,w-1)*sech_;
dder=w*sech_*pow(tanh_,w-2)*((w-1)-(w+1)*tanh_*tanh_);
}



double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
				rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
				rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
				rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}











void RadialBasis_Chebyshev_tanhexp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

int sigma=k;

double x=scal*(r-s)*0.5 ;
double exp_=exp(-0.5559-x);
double u=exp_+(k+1)*0.04;


double tanh_u=tanh(u);
double temp = 2*tanh_u-1;
double ksi= 2*temp*temp - 1;
double dydu=1-tanh_u*tanh_u;
double dyddu=-2*tanh_u*dydu;
double der= -8*temp*exp_*dydu;
double dder= 8*(exp_*exp_*(2*dydu*dydu+temp*dyddu)+temp*dydu*exp_);
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
				rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
				rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
				rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}




void RadialBasis_Chebyshev_tanhexp_w::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

int sigma=k%3;

double x=scal*(r-s)*0.5 ;
double exp_=exp(-0.5559-x);
double u;

if (sigma==0) {u=exp_+0.02;}
else if (sigma==1) {u=exp_+0.15;}
else if (sigma==2) {u=exp_+0.5;}

double tanh_u=tanh(u);
double temp = 2*tanh_u-1;
double ksi= 2*temp*temp - 1;
double dydu=1-tanh_u*tanh_u;
double dyddu=-2*tanh_u*dydu;
double der= -8*temp*exp_*dydu;
double dder= 8*(exp_*exp_*(2*dydu*dydu+temp*dyddu)+temp*dydu*exp_);
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*(r-s)*scal/4;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;
double w=1-ksi;
double t1=ksi * cutoff_f;
double t2=mult * cutoff_f + 2 * ksi * Dr;
rb_vals[0] = scaling * cutoff_f*w;
rb_ders[0] = (2*Dr*w-mult*cutoff_f)*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = -mult_scal*cutoff_f*scaling;//scaling * 0;
rb_ders[0 + rb_size * 2] = -(2*Dr*mult_scal+mult_scal_r*cutoff_f)*scaling;//scaling * 0;
rb_ders[0 + rb_size * 3] = -mult_s*cutoff_f*scaling;//scaling * 0;
rb_ders[0 + rb_size * 4] = -(2*Dr*mult_s+mult_s_r*cutoff_f)*scaling;//scaling * 0;
rb_vals[1] = scaling * t1*w;
rb_ders[1] = scaling * (t2*w-mult*t1);
rb_ders[1 + rb_size * 1] = scaling * (mult_scal * cutoff_f*w-mult_scal*t1);
rb_ders[1 + rb_size * 2] = scaling * ((mult_scal_r * cutoff_f + 2 * mult_scal * Dr)*w-mult*(mult_scal * cutoff_f)-mult_scal_r*t1-mult_scal*t2);
rb_ders[1 + rb_size * 3] = scaling * (mult_s * cutoff_f*w-mult_s*t1);
rb_ders[1 + rb_size * 4] = scaling * ((mult_s_r * cutoff_f + 2 * mult_s * Dr)*w-mult*(mult_s * cutoff_f)-mult_s_r*t1-mult_s*t2);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
				rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
				rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
				rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}



void RadialBasis_Chebyshev_sss_lmp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif
double x= scal*(r-s)/2;
double ksi = tanh(x);
double der = 1-ksi*ksi;
double mult = der*scal/2;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;


rb_vals[0] = scaling *  cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));

rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);

        

         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				        }

}




void RadialBasis_Chebyshev_sssw_lmp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)/2 ;
double tanh_ = tanh(x);
double sech_ = 1-tanh_*tanh_;

int w=1+2*k;
double ksi;
double der ;

if (w==1) {ksi = tanh_;
der = sech_;
}
else {ksi = pow(tanh_,w);
der = w*pow(tanh_,w-1)*sech_;

}


double mult = der*scal/2;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;


rb_vals[0] = scaling *  cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));

rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
				        }

}


void RadialBasis_Chebyshev_s_lmp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)*0.5;
double expm= exp(-x);
double expp= exp(x);
double x_plus = 1+x;
double ksi = -2*(exp((x_plus)*expm-1))+1;
double der = 2*exp((-1+expm)*(x_plus))*x;


double mult = der*scal/2;

double Dr=r-max_dist;
double cutoff_f=Dr * Dr;
rb_vals[0] = scaling *  cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));

rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);




         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                        }

}



void RadialBasis_Chebyshev_s::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
#endif

double x=scal*(r-s)*0.5;
double expm= exp(-x);
double expp= exp(x);
double x_plus = 1+x;
double ksi = -2*(exp((x_plus)*expm-1))+1;
double der = 2*exp((-1+expm)*(x_plus))*x;
             
double dder = -2*exp(-1-2*x+expm*(x_plus))*(expp*(-1+x)+x*x);
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;
rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * (ksi * cutoff_f);
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }


}




void RadialBasis_Chebyshev_ss_lmp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
double x=scal*(r-s)/2 ;
double denom=x*x+1 ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = 1/(denom*sq);

double mult = der*scal/2;

double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));

rb_vals[1] = scaling * ksi * cutoff_f;
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);




         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                        }

}



void RadialBasis_Chebyshev_ssw_lmp::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
double x=scal*(r-s)/2 ;
double denom=x*x+exp(k-1) ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = exp(k-1)/(denom*sq);

double mult = der*scal/2;

double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));

rb_vals[1] = scaling * ksi * cutoff_f;
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);




         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                        }

}



void RadialBasis_Chebyshev_sigma::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
double x=scal*(r-s)*0.5 ;
double ksi;
double der;
double dder;
int sigma = k%3;
if (sigma==0)
{double denom=x*x+1 ;
double sq=sqrt(denom);
ksi = x/sq;
der = 1/(denom*sq);
dder = -3*ksi*der/sq;
}
else if (sigma==1)
{
double expm= exp(-x);
double expp= exp(x);
double x_plus = 1+x;
ksi = -2*(exp((x_plus)*expm-1))+1;
der = 2*exp((-1+expm)*(x_plus))*x;
dder = -2*exp(-1-2*x+expm*(x_plus))*(expp*(-1+x)+x*x);
}
else if (sigma==2)
{
double exp_=exp(-0.5559-x);
double u=exp_+0.02;
double tanh_u=tanh(u);
double temp = 2*tanh_u-1;
ksi= 2*temp*temp - 1;
double dydu=1-tanh_u*tanh_u;
double dyddu=-2*tanh_u*dydu;
der = -8*temp*exp_*dydu;
dder = 8*(exp_*exp_*(2*dydu*dydu+temp*dyddu)+temp*dydu*exp_);
}
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * ksi * cutoff_f;
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}


void RadialBasis_Laguerre_log1p::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pCalc(*this, r, scal, s, true, true);
}

void RadialBasis_Laguerre_log1p_lmp::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pCalc(*this, r, scal, s, true, false);
}

void RadialBasis_Laguerre_log1p_pos::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pPositiveCalc(*this, r, scal, s, true, true);
}

void RadialBasis_Laguerre_log1p_pos_lmp::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pPositiveCalc(*this, r, scal, s, true, false);
}

void RadialBasis_Laguerre_log1p_noenv::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pCalc(*this, r, scal, s, false, true);
}

void RadialBasis_Laguerre_log1p_noenv_lmp::RB_Calc(double r, double scal, double s, int k)
{
	LaguerreLog1pCalc(*this, r, scal, s, false, false);
}

void RadialBasis_Jacobi_sss::RB_Calc(double r, double scal, double s, int k)
{
	JacobiSSSCalc(*this, r, scal, s, k, true, true);
}

void RadialBasis_Jacobi_sss_lmp::RB_Calc(double r, double scal, double s, int k)
{
	JacobiSSSCalc(*this, r, scal, s, k, true, false);
}

void RadialBasis_Jacobi_sss_noweight::RB_Calc(double r, double scal, double s, int k)
{
	JacobiSSSCalc(*this, r, scal, s, k, false, true);
}

void RadialBasis_Jacobi_sss_noweight_lmp::RB_Calc(double r, double scal, double s, int k)
{
	JacobiSSSCalc(*this, r, scal, s, k, false, false);
}




void RadialBasis_Chebyshev_ss::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
double x=scal*(r-s)/2 ;
double denom=x*x+1 ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = 1/(denom*sq);
double dder = -3*ksi*der/sq;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * ksi * cutoff_f;
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}




void RadialBasis_Chebyshev_ssw::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
double x=scal*(r-s)/2 ;
double denom=x*x+exp(k-1) ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = exp(k-1)/(denom*sq);
double dder = -3*ksi*der/sq;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
double Dr=r-max_dist;
double cutoff_f=Dr * Dr;

rb_vals[0] = scaling * cutoff_f;
rb_ders[0] = 2*Dr*scaling;//scaling * (0 * (r - max_dist) * (r - max_dist) + 2 * (r - max_dist));
rb_ders[0 + rb_size * 1] = 0;//scaling * 0;
rb_ders[0 + rb_size * 2] = 0;//scaling * 0;
rb_ders[0 + rb_size * 3] = 0;//scaling * 0;
rb_ders[0 + rb_size * 4] = 0;//scaling * 0;
rb_vals[1] = scaling * ksi * cutoff_f;
rb_ders[1] = scaling * (mult * cutoff_f + 2 * ksi * Dr);
rb_ders[1 + rb_size * 1] = scaling * mult_scal * cutoff_f;
rb_ders[1 + rb_size * 2] = scaling * (mult_scal_r * cutoff_f + 2 * mult_scal * Dr);
rb_ders[1 + rb_size * 3] = scaling * mult_s * cutoff_f;
rb_ders[1 + rb_size * 4] = scaling * (mult_s_r * cutoff_f + 2 * mult_s * Dr);



         for (int i = 2; i < rb_size; i++) {
                rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
                rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                                rb_ders[i + rb_size] = 2 * (mult_scal * rb_vals[i - 1] + ksi * rb_ders[i - 1+ rb_size]) - rb_ders[i - 2+ rb_size];
                                rb_ders[i + 2 * rb_size] = 2 * (mult_scal_r * rb_vals[i - 1] + mult * rb_ders[i - 1 + rb_size] + ksi * rb_ders[i - 1 + rb_size * 2] + mult_scal * rb_ders[i - 1]) - rb_ders[i - 2 + rb_size * 2];
                                rb_ders[i + 3 * rb_size] = 2 * (mult_s * rb_vals[i - 1] + ksi * rb_ders[i + 3 * rb_size-1])- rb_ders[i + 3 * rb_size-2];
                                rb_ders[i + 4 * rb_size] = 2 * (mult_s_r * rb_vals[i - 1] +mult* rb_ders[i + 3 * rb_size - 1] +ksi * rb_ders[i + 4 * rb_size - 1] + mult_s* rb_ders[i - 1]) - rb_ders[i + 4 * rb_size - 2];
        }

}

void RadialBasis_Besselw::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
//int sigma = k%4;
double x=scal*(r-s)/2 ;
double denom=x*x+1+k ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = (1+k)/(denom*sq);
double dder = -3*ksi*der/sq;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
//double Dr=r-max_dist;
double Dr=r/max_dist-1;
double p=6;
double cutoff_f = 1-(p+1)*(p+2)*pow(Dr,p)*0.5+p*(p+2)*pow(Dr,p+1)-(p+1)*p*pow(Dr,p+2)*0.5 ;
double cutoff_der = p*(p+1)*(p+2)*(pow(Dr,p)-0.5*(pow(Dr,p+1)+pow(Dr,p-1)))/max_dist ;
//double cutoff_f=Dr * Dr;
double PI=3.141592654 ;
double temp=ksi+1.00001;


for (int i = 0; i < rb_size; i++) {
double N=i+1;
double w=PI*N/2 ;
double sin_=sin(w*temp);
double cos_=cos(w*temp);
double bessel=sin_/temp;
double bessel_der=w*cos_/temp-sin_/temp/temp;
double bessel_ddr=-(w*w*sin_/temp)-2*w*cos_/temp/temp+2*sin_/temp/temp/temp;
rb_vals[i]=scaling*bessel*cutoff_f;
rb_ders[i]=scaling*(bessel_der*mult*cutoff_f+bessel*cutoff_der);
rb_ders[i+rb_size * 1] = scaling * bessel_der * mult_scal * cutoff_f;
rb_ders[i+rb_size * 2] = scaling * (bessel_ddr* mult*mult_scal * cutoff_f +  bessel_der * mult_scal_r * cutoff_f
+cutoff_der*bessel_der * mult_scal);
rb_ders[i+rb_size * 3] = scaling * bessel_der * mult_s * cutoff_f;
rb_ders[i+rb_size * 4] = scaling * (bessel_ddr* mult*mult_s * cutoff_f +  bessel_der * mult_s_r * cutoff_f
+cutoff_der*bessel_der * mult_s);
}



}


void RadialBasis_Bessel::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
        if (r < min_dist) {
                Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
        }
        if (r > max_dist) {
                ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
                        ", min_dist = " + to_string(min_dist) + '\n');
#endif
//int sigma = k%4;
double x=scal*(r-s)/2 ;
double denom=x*x+1 ;
double sq=sqrt(denom);
double ksi = x/sq;
double der = 1/(denom*sq);
double dder = -3*ksi*der/sq;
double mult = der*scal/2;
double mult_s_r=-dder*scal*scal/4;
double mult_scal_r = der/2+dder*x/2;
double mult_scal = der *(r-s)/2;
double mult_s=-mult;
//double Dr=r-max_dist;
double Dr=r/max_dist-1;
double p=6;
double cutoff_f = 1-(p+1)*(p+2)*pow(Dr,p)*0.5+p*(p+2)*pow(Dr,p+1)-(p+1)*p*pow(Dr,p+2)*0.5 ;
double cutoff_der = p*(p+1)*(p+2)*(pow(Dr,p)-0.5*(pow(Dr,p+1)+pow(Dr,p-1)))/max_dist ;
//double cutoff_f=Dr * Dr;
double PI=3.141592654 ;
double temp=ksi+1.00001;


for (int i = 0; i < rb_size; i++) {
double N=i+1;
double w=PI*N/2 ;
double sin_=sin(w*temp);
double cos_=cos(w*temp);
double bessel=sin_/temp;
double bessel_der=w*cos_/temp-sin_/temp/temp;
double bessel_ddr=-(w*w*sin_/temp)-2*w*cos_/temp/temp+2*sin_/temp/temp/temp;
rb_vals[i]=scaling*bessel*cutoff_f;
rb_ders[i]=scaling*(bessel_der*mult*cutoff_f+bessel*cutoff_der);
rb_ders[i+rb_size * 1] = scaling * bessel_der * mult_scal * cutoff_f;
rb_ders[i+rb_size * 2] = scaling * (bessel_ddr* mult*mult_scal * cutoff_f +  bessel_der * mult_scal_r * cutoff_f
+cutoff_der*bessel_der * mult_scal);
rb_ders[i+rb_size * 3] = scaling * bessel_der * mult_s * cutoff_f;
rb_ders[i+rb_size * 4] = scaling * (bessel_ddr* mult*mult_s * cutoff_f +  bessel_der * mult_s_r * cutoff_f
+cutoff_der*bessel_der * mult_s);
}



}




void RadialBasis_Chebyshev::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	double mult = 2.0 / (max_dist - min_dist);
	double ksi = (2 * r - (min_dist + max_dist)) / (max_dist - min_dist);
        //double ksi = 1-2*(exp(-scal*((r-min_dist)/(max_dist - min_dist)-1))-1)/(exp(scal)-1);
//        double mult = 2.0*scal*exp(-scal*((r-min_dist)/(max_dist - min_dist)-1))/(exp(scal)-1)/(max_dist - min_dist);
//        double mult_scal = 2*(exp(scal)*(-1+exp(scal*(r-max_dist)/(min_dist-max_dist)))+exp(scal*(r-max_dist)/(min_dist-max_dist))*(-1+exp(scal))*(max_dist-r)/(min_dist-max_dist))/(-1+exp(scal))/(-1+exp(scal));
//        double mult_scal_r=2*exp(scal*(r-max_dist)/(min_dist-max_dist))*(min_dist+(-1+scal)*min_dist*exp(scal)+max_dist*(-1-scal+exp(scal))-scal*(-1+exp(scal)*r))
//                           /(-1+exp(scal))/(-1+exp(scal))/(min_dist-max_dist)/(min_dist-max_dist);
        //double mult = 2*scal*exp(-scal*(-1 + (-min_dist + r)/(max_dist - min_dist)))/((max_dist - min_dist)*(exp(scal) - 1));
//        double mult_scal = (-2 + 2*exp(-scal*(-1 + (-min_dist + r)/(max_dist - min_dist))))*exp(scal)/(exp(scal) - 1)/(exp(scal) - 1) - 2*(1 - (-min_dist + r)/(max_dist - min_dist))*exp(-scal*(-1 + (-min_dist + r)/(max_dist - min_dist)))/(exp(scal) - 1);
//        double mult_scal_r = 2*(scal*(1 + (min_dist - r)/(max_dist - min_dist)) - scal*exp(scal)/(exp(scal) - 1) + 1)*exp(scal*(1 + (min_dist - r)/(max_dist - min_dist)))/((max_dist - min_dist)*(exp(scal) - 1)); 
//	rb_vals[0] = scaling * (1 * (ksi - 1)*(ksi - 1));
//	rb_ders[0] = scaling * (0 * (ksi - 1)*(ksi - 1) + 2 * (ksi - 1)*mult);
        
        rb_vals[0] = scaling * (1 * (r - max_dist)*(r - max_dist));
        rb_ders[0] = scaling * (0 * (r - max_dist)*(r - max_dist) + 2 * (r - max_dist));
        rb_vals[1] = scaling * (ksi*(r - max_dist)*(r - max_dist));
        rb_ders[1] = scaling * (mult * (r - max_dist)*(r - max_dist) + 2 * ksi*(r - max_dist));
//	rb_vals[1] = scaling * (ksi*(ksi - 1)*(ksi - 1));
//	rb_ders[1] = scaling * (mult * (3*ksi*ksi-4*ksi+1));
        
	for (int i = 2; i < rb_size; i++) {
		rb_vals[i] = 2 * ksi*rb_vals[i - 1] - rb_vals[i - 2];
		rb_ders[i] = 2 * (mult * rb_vals[i - 1] + ksi * rb_ders[i - 1]) - rb_ders[i - 2];
                
                
	}
}

void RadialBasis_Chebyshev_repuls::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif

	if (r < min_dist)
		r = min_dist;
        double PI=3.141592654 ;
        double p=8;
        double r_d=r/max_dist;
        double d_r=(max_dist-min_dist)/rb_size;
//	double mult = 2*exp(-1*((r-min_dist)/(max_dist-min_dist)-1))/(exp(1)-1)/(max_dist-min_dist);
//	double ksi = 1-2*(exp(-2*((r-min_dist)/(max_dist-min_dist)-1))-1)/(exp(2)-1);
	double ksi = (r-0.0*min_dist)/(max_dist-0.0*min_dist);
        double mult =1/(max_dist-0.0*min_dist);
        double cutoff = 1-(p+1)*(p+2)*pow(ksi,p)*0.5+p*(p+2)*pow(ksi,p+1)-(p+1)*p*pow(ksi,p+2)*0.5 ;
        double cutoff_der = p*(p+1)*(p+2)*(pow(ksi,p)-0.5*(pow(ksi,p+1)+pow(ksi,p-1)))/(max_dist-0.0*min_dist) ;
//        double cutoff = pow(1-ksi, p);
//        double cutoff_der = -1.0*p*pow(1-ksi, p-1)/(max_dist-0.0*min_dist);

        std::vector<double> g;
        std::vector<double> g_der;
        g.resize(rb_size);
        g_der.resize(rb_size);
//        g[0]=scaling*1.0;
//        g_der[0]= 0;
//        rb_vals[0] = g[0] ;
//        rb_ders[0] = 0 ;
//        g[0]=scaling *0.5*cutoff;
//        g_der[0]=scaling *0.5*cutoff_der;
//        g[1]=g[1]*ksi;
//        g_der[1]=g_der[0]*ksi+mult*g[0];
//	rb_vals[0] = g[0]+g[0];
//	rb_ders[0] = g_der[0]+g_der[0];
//	rb_vals[1] = g[0]-g[1];
//	rb_ders[1] = g_der[0]-g_der[1];
	for (int i = 0; i < rb_size; i++) {
                		
	//	g_der[i] = 2 * (mult * g[i - 1] + ksi * g_der[i - 1]) - g_der[i - 2];
	  //      double r_i=r/(min_dist+(i+1)*d_r)-1;
	      //  double r_i=(ksi*rb_size/(i+1)-1)*0.85;
                 double r_i=(ksi/(1-i*(1-min_dist/max_dist)/rb_size))*(1/(1-min_dist/(max_dist-i*d_r)));
                 double r_norm =(min_dist/max_dist/(1-i*(1-min_dist/max_dist)/rb_size))*(1/(1-min_dist/(max_dist-i*d_r)));
               // double r_i=(ksi/pow(1.4, i)-1)*0.85;
                g[i]=log(r_i+1)-r_i;
               // g_der[i]=(1/(r_i+1)-1)*(min_dist+(i+1)*d_r);
              //  g_der[i]=(1/(r_i+1)-1)*rb_size/(i+1)*mult*0.85;
                g_der[i]=(1/(r_i+1)-1)/(1-i*(1-min_dist/max_dist)/rb_size)*mult*(1/(1-min_dist/(max_dist-i*d_r)));
               //   g_der[i]=(1/(r_i+1)-1)/pow(1.4, i)*mult*0.85;
                rb_vals[i] = scaling * g[i] * cutoff/( log(r_norm+1)-r_norm);
                rb_ders[i] = scaling*(cutoff_der*g[i] + cutoff * g_der[i])/(log(r_norm+1)-r_norm) ;
             
	}
	if (r == min_dist)
		for (int i = 0; i < rb_size; i++)
			rb_ders[i] = 0.0;
}

void RadialBasis_Taylor::RB_Calc(double r, double scal, double s, int k)
{
#ifdef MLIP_DEBUG
	if (r < min_dist) {
		Warning("RadialBasis: r<min_dist. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
	if (r > max_dist) {
		ERROR("RadialBasis: r>MaxDist !!!. r = " + to_string(r) +
			", min_dist = " + to_string(min_dist) + '\n');
	}
#endif
        double PI=3.141592654 ;
        double p=2;
        double r_d=r/max_dist;
        double cutoff = pow(1-r_d, p);
        double cutoff_der = -1.0*p*pow(1-r_d, p-1)/max_dist;

	rb_vals[0] = scaling * 1*cutoff;
	rb_ders[0] = scaling * 0;
	for (int i = 1; i < rb_size; i++)
	{
                //rb_vals[i]=scaling*(max_dist-r)*(max_dist-r)*sin((i+1)*PI*r/max_dist)/r  ;
                //rb_ders[i]=scaling*(max_dist-r)*((i+1)*PI*(max_dist-r)*r*cos((i+1)*PI*r/max_dist)-max_dist*(max_dist+r)*sin((i+1)*PI*r/max_dist))/max_dist/r/r ;
	//	rb_ders[i] = scaling * (i * rb_vals[i - 1]/max_dist+cutoff_der*rb_vals[i]/cutoff);
		rb_vals[i] = scaling * r_d * rb_vals[i - 1];
                rb_ders[i] = scaling * (i * rb_vals[i - 1]/max_dist+cutoff_der*rb_vals[i]/cutoff);
	}
}

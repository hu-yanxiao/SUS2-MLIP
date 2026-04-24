/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include <cmath>
#include "mtpr.h"

#ifdef MLIP_MPI
#	include "mpi.h"
#endif

using namespace std;

namespace {

constexpr double kRandomScalMin = 2.0;
constexpr double kRandomScalMax = 4.0;
constexpr double kLaguerreRandomScalMax = 3.0;
constexpr double kRandomShiftMin = 1.5;
constexpr double kRandomShiftMax = 2.5;
constexpr int kEnvGateDefaultChannels = 6;
constexpr double kEnvGateDefaultCutoffRatio = 0.5;
constexpr double kEnvGateDefaultLambdaRaw = -2.0;
constexpr double kEnvGateDefaultLogDensityCoeff = 0.0;
constexpr double kEnvGateMaxLogDensityCoeff = 6.0;

double Clamp01(double value)
{
	return std::max(0.0, std::min(1.0, value));
}

double Lerp(double start, double end, double t)
{
	return start + (end - start) * t;
}

double StableSigmoid(double value)
{
	if (value >= 0.0) {
		const double z = std::exp(-value);
		return 1.0 / (1.0 + z);
	}
	const double z = std::exp(value);
	return z / (1.0 + z);
}

double EnvGateDensityCoeff(double raw_log_density)
{
	return std::exp(std::min(raw_log_density, kEnvGateMaxLogDensityCoeff));
}

double EnvGateDensityCoeffDer(double raw_log_density)
{
	if (raw_log_density >= kEnvGateMaxLogDensityCoeff)
		return 0.0;
	return std::exp(raw_log_density);
}

void BernsteinDegree5(double y, double* basis, double* basis_der)
{
	y = Clamp01(y);
	const double u = 1.0 - y;
	const double y2 = y * y;
	const double y3 = y2 * y;
	const double y4 = y3 * y;
	const double y5 = y4 * y;
	const double u2 = u * u;
	const double u3 = u2 * u;
	const double u4 = u3 * u;
	const double u5 = u4 * u;

	basis[0] = u5;
	basis[1] = 5.0 * y * u4;
	basis[2] = 10.0 * y2 * u3;
	basis[3] = 10.0 * y3 * u2;
	basis[4] = 5.0 * y4 * u;
	basis[5] = y5;

	if (basis_der == nullptr)
		return;

	const double degree4[5] = {
		u4,
		4.0 * y * u3,
		6.0 * y2 * u2,
		4.0 * y3 * u,
		y4
	};
	basis_der[0] = -5.0 * degree4[0];
	for (int q = 1; q < 5; ++q)
		basis_der[q] = 5.0 * (degree4[q - 1] - degree4[q]);
	basis_der[5] = 5.0 * degree4[4];
}

bool IsLaguerreLog1pBasisType(const std::string& basis_type)
{
	return basis_type == "RBLaguerre_log1p"
	    || basis_type == "RBLaguerre_log1p_lmp"
	    || basis_type == "RBLaguerre_log1p_pos"
	    || basis_type == "RBLaguerre_log1p_pos_lmp"
	    || basis_type == "RBLaguerre_log1p_noenv"
	    || basis_type == "RBLaguerre_log1p_noenv_lmp";
}

bool IsLaguerreLog1pBasisType(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && IsLaguerreLog1pBasisType(radial_basis->GetRBTypeString());
}

bool IsJacobiIndexedBasisType(const std::string& basis_type)
{
	return basis_type == "RBJacobi_sss"
	    || basis_type == "RBJacobi_sss_lmp"
	    || basis_type == "RBJacobi_sss_noweight"
	    || basis_type == "RBJacobi_sss_noweight_lmp";
}

bool IsJacobiIndexedBasisType(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && IsJacobiIndexedBasisType(radial_basis->GetRBTypeString());
}

bool UsesPrecomputedLmpTable(const std::string& basis_type)
{
	return basis_type == "RBChebyshev_sss_lmp"
	    || basis_type == "RBLaguerre_log1p_lmp"
	    || basis_type == "RBLaguerre_log1p_pos_lmp"
	    || basis_type == "RBLaguerre_log1p_noenv_lmp"
	    || basis_type == "RBJacobi_sss_lmp"
	    || basis_type == "RBJacobi_sss_noweight_lmp";
}

bool UsesPrecomputedLmpTable(AnyRadialBasis* radial_basis)
{
	return radial_basis != nullptr && UsesPrecomputedLmpTable(radial_basis->GetRBTypeString());
}

double ScalingSlopeUpperBound(AnyRadialBasis* radial_basis)
{
	if (IsLaguerreLog1pBasisType(radial_basis))
		return kLaguerreRandomScalMax;
	return kRandomScalMax;
}

double ScalingSlopeRangeStart(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_slope_range)
		return mtpr.scaling_slope_range_start;
	return ScalingSlopeUpperBound(mtpr.p_RadialBasis);
}

double ScalingSlopeRangeEnd(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_slope_range)
		return mtpr.scaling_slope_range_end;
	return kRandomScalMin;
}

double ScalingShiftRangeStart(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_shift_range)
		return mtpr.scaling_shift_range_start;
	return kRandomShiftMin;
}

double ScalingShiftRangeEnd(const MLMTPR& mtpr)
{
	if (mtpr.custom_scaling_shift_range)
		return mtpr.scaling_shift_range_end;
	return kRandomShiftMax;
}

}  // namespace

int MLMTPR::ScalingCoeffCount() const
{
	return 2 * species_count * species_count * K_;
}

bool MLMTPR::HasCompleteParameters() const
{
	return inited
	    && has_shift_coeffs
	    && has_scal_coeffs
	    && has_radial_coeffs
	    && has_linear_coeffs
	    && (!env_gate_enabled || has_env_gate_coeffs);
}

int MLMTPR::RadialCoeffOffset() const
{
	return species_count + ScalingCoeffCount();
}

int MLMTPR::RadialCoeffBlockSize() const
{
	return p_RadialBasis->rb_size + species_count;
}

int MLMTPR::EnvGateCoeffCount() const
{
	if (!env_gate_enabled)
		return 0;
	return 1 + species_count * env_gate_channel_count;
}

int MLMTPR::EnvGateCoeffOffset() const
{
	return RadialCoeffOffset() + radial_func_count * RadialCoeffBlockSize();
}

int MLMTPR::EnvGateLambdaRawOffset() const
{
	return EnvGateCoeffOffset();
}

int MLMTPR::EnvGateLogDensityCoeffOffset(int type_central, int channel) const
{
	return EnvGateCoeffOffset() + 1 + type_central * env_gate_channel_count + channel;
}

int MLMTPR::LinearCoeffOffset() const
{
	return EnvGateCoeffOffset() + EnvGateCoeffCount();
}

bool MLMTPR::HasEnvGate() const
{
	return env_gate_enabled && has_env_gate_coeffs;
}

double MLMTPR::EnvGateLambda() const
{
	if (!HasEnvGate())
		return 0.0;
	return StableSigmoid(regression_coeffs[EnvGateLambdaRawOffset()]);
}

void MLMTPR::EnableEnvGateDefault()
{
	if (env_gate_enabled && has_env_gate_coeffs)
		return;

	const int old_linear_offset = LinearCoeffOffset();
	const int expected_linear_count = alpha_count + species_count - 1;
	const int old_linear_count =
		static_cast<int>(regression_coeffs.size()) >= old_linear_offset + expected_linear_count
		? expected_linear_count
		: 0;
	std::vector<double> old_linear;
	if (old_linear_count > 0 && static_cast<int>(regression_coeffs.size()) >= old_linear_offset + old_linear_count)
		old_linear.assign(regression_coeffs.begin() + old_linear_offset,
		                  regression_coeffs.begin() + old_linear_offset + old_linear_count);

	env_gate_enabled = true;
	has_env_gate_coeffs = true;
	env_gate_cutoff_ratio = kEnvGateDefaultCutoffRatio;
	env_gate_channel_count = kEnvGateDefaultChannels;

	const int env_offset = EnvGateCoeffOffset();
	const int env_count = EnvGateCoeffCount();
	const int new_linear_offset = LinearCoeffOffset();
	std::vector<double> new_coeffs(new_linear_offset + old_linear_count, 0.0);
	for (int i = 0; i < env_offset && i < static_cast<int>(regression_coeffs.size()); ++i)
		new_coeffs[i] = regression_coeffs[i];

	new_coeffs[EnvGateLambdaRawOffset()] = kEnvGateDefaultLambdaRaw;
	for (int i = 1; i < env_count; ++i)
		new_coeffs[env_offset + i] = kEnvGateDefaultLogDensityCoeff;
	for (int i = 0; i < old_linear_count; ++i)
		new_coeffs[new_linear_offset + i] = old_linear[i];

	regression_coeffs.swap(new_coeffs);
	if (old_linear_count > 0) {
		linear_coeffs.assign(old_linear_count, 0.0);
		for (int i = 0; i < old_linear_count; ++i)
			linear_coeffs[i] = regression_coeffs[new_linear_offset + i];
	}
}

void MLMTPR::DisableEnvGate()
{
	if (!env_gate_enabled)
		return;

	const int old_env_offset = EnvGateCoeffOffset();
	const int old_linear_offset = LinearCoeffOffset();
	const int old_linear_count = has_linear_coeffs
		? alpha_count + species_count - 1
		: 0;

	std::vector<double> new_coeffs(old_env_offset + old_linear_count, 0.0);
	for (int i = 0; i < old_env_offset && i < static_cast<int>(regression_coeffs.size()); ++i)
		new_coeffs[i] = regression_coeffs[i];
	for (int i = 0; i < old_linear_count; ++i)
		new_coeffs[old_env_offset + i] = regression_coeffs[old_linear_offset + i];

	env_gate_enabled = false;
	has_env_gate_coeffs = false;
	env_gate_cutoff_ratio = kEnvGateDefaultCutoffRatio;
	env_gate_channel_count = kEnvGateDefaultChannels;
	regression_coeffs.swap(new_coeffs);
}

double MLMTPR::ComputeEnvGate(const Neighborhood& nbh,
                              std::vector<Vector3>* gate_ders,
                              std::vector<double>* channel_sums,
                              std::vector<double>* rho_dr)
{
	if (!HasEnvGate()) {
		if (gate_ders != nullptr)
			gate_ders->assign(nbh.count, Vector3(0.0, 0.0, 0.0));
		if (channel_sums != nullptr)
			channel_sums->assign(kEnvGateDefaultChannels, 0.0);
		if (rho_dr != nullptr)
			rho_dr->assign(nbh.count, 0.0);
		return 1.0;
	}

	if (env_gate_channel_count != kEnvGateDefaultChannels)
		ERROR("centered_tanh_screen env-gate currently requires 6 Bernstein channels.");

	const int type_central = nbh.my_type;
	if (type_central >= species_count)
		throw MlipException("Too few species count in the MTP potential!");

	const double r_env = env_gate_cutoff_ratio * p_RadialBasis->max_dist;
	if (r_env <= 0.0)
		ERROR("env_gate_cutoff_ratio should be positive.");

	env_gate_density_coeffs_.assign(env_gate_channel_count, 0.0);
	env_gate_channel_sums_.assign(env_gate_channel_count, 0.0);
	env_gate_rho_dr_.assign(nbh.count, 0.0);
	if (gate_ders != nullptr)
		gate_ders->assign(nbh.count, Vector3(0.0, 0.0, 0.0));
	const bool use_env_gate_table =
		env_gate_radial_list.size1 == species_count * species_count
		&& env_gate_radial_list.size2 >= 2
		&& env_gate_radial_list.size3 == env_gate_channel_count;

	for (int q = 0; q < env_gate_channel_count; ++q)
		env_gate_density_coeffs_[q] =
			EnvGateDensityCoeff(regression_coeffs[EnvGateLogDensityCoeffOffset(type_central, q)]);

	double rho = 0.0;
	for (int j = 0; j < nbh.count; ++j) {
		const double r = nbh.dists[j];
		if (r <= 0.0 || r >= r_env)
			continue;
		const int type_outer = nbh.types[j];
		if (type_outer >= species_count)
			throw MlipException("Too few species count in the MTP potential!");

		double weighted_basis_values[6];
		double weighted_basis_ders[6];
		if (use_env_gate_table) {
			int r_list = static_cast<int>(std::floor(r * inv_dr));
			const int last_interval = env_gate_radial_list.size2 - 2;
			if (r_list < 0)
				r_list = 0;
			if (r_list > last_interval)
				r_list = last_interval;
			const int r_next = r_list + 1;
			double ddr = r * inv_dr - r_list;
			if (ddr < 0.0)
				ddr = 0.0;
			if (ddr > 1.0)
				ddr = 1.0;
			const int pair_index = type_central * species_count + type_outer;
			for (int q = 0; q < env_gate_channel_count; ++q) {
				const double v1 = env_gate_radial_list(pair_index, r_list, q);
				const double v2 = env_gate_radial_list(pair_index, r_next, q);
				const double d1 = env_gate_radial_der_list(pair_index, r_list, q);
				const double d2 = env_gate_radial_der_list(pair_index, r_next, q);
				weighted_basis_values[q] = v1 + ddr * (v2 - v1);
				weighted_basis_ders[q] = d1 + ddr * (d2 - d1);
			}
		} else {
			const double y = r / r_env;
			const double dy_dr = 1.0 / r_env;
			const double cutoff = 1.0 - r / r_env;
			const double f_env = cutoff * cutoff;
			const double df_dr = -2.0 * cutoff / r_env;
			double bernstein[6];
			double bernstein_der[6];
			BernsteinDegree5(y, bernstein, bernstein_der);
			for (int q = 0; q < env_gate_channel_count; ++q) {
				weighted_basis_values[q] = f_env * bernstein[q];
				weighted_basis_ders[q] = df_dr * bernstein[q] + f_env * bernstein_der[q] * dy_dr;
			}
		}

		double local_rho_dr = 0.0;
		for (int q = 0; q < env_gate_channel_count; ++q) {
			const double weighted_basis = weighted_basis_values[q];
			env_gate_channel_sums_[q] += weighted_basis;
			rho += env_gate_density_coeffs_[q] * weighted_basis;
			local_rho_dr += env_gate_density_coeffs_[q]
				* weighted_basis_ders[q];
		}
		env_gate_rho_dr_[j] = local_rho_dr;
	}

	const double lambda = EnvGateLambda();
	const double tanh_rho = std::tanh(rho);
	const double sech2_rho = 1.0 - tanh_rho * tanh_rho;
	const double gate = 1.0 - lambda * tanh_rho;

	if (gate_ders != nullptr) {
		for (int j = 0; j < nbh.count; ++j) {
			const double r = nbh.dists[j];
			if (r <= 0.0 || env_gate_rho_dr_[j] == 0.0)
				continue;
			const double dgate_dr = -lambda * sech2_rho * env_gate_rho_dr_[j];
			for (int a = 0; a < 3; ++a)
				(*gate_ders)[j][a] = dgate_dr * nbh.vecs[j][a] / r;
		}
	}
	if (channel_sums != nullptr)
		*channel_sums = env_gate_channel_sums_;
	if (rho_dr != nullptr)
		*rho_dr = env_gate_rho_dr_;

	return gate;
}

bool MLMTPR::IsRedundantRadialSpeciesCoeff(int coeff_index) const
{
	const int radial_begin = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int rb_size = p_RadialBasis->rb_size;
	if (coeff_index < radial_begin)
		return false;
	const int radial_end = radial_begin + radial_func_count * block_size;
	if (coeff_index >= radial_end)
		return false;

	const int relative_index = coeff_index - radial_begin;
	const int mu = relative_index / block_size;
	const int in_block = relative_index % block_size;
	return mu > 0 && in_block >= rb_size;
}

void MLMTPR::BuildActiveCoeffIndices(std::vector<int>& active_coeff_indices, bool exclude_scal_coeffs) const
{
	active_coeff_indices.clear();
	active_coeff_indices.reserve(regression_coeffs.size());
	const int scal_begin = species_count;
	const int scal_end = RadialCoeffOffset();
	for (int coeff_index = 0; coeff_index < static_cast<int>(regression_coeffs.size()); ++coeff_index) {
		if (exclude_scal_coeffs && coeff_index >= scal_begin && coeff_index < scal_end)
			continue;
		if (IsRedundantRadialSpeciesCoeff(coeff_index))
			continue;
		active_coeff_indices.push_back(coeff_index);
	}
}

int MLMTPR::ActiveCoeffCount(bool exclude_scal_coeffs) const
{
	std::vector<int> active_coeff_indices;
	BuildActiveCoeffIndices(active_coeff_indices, exclude_scal_coeffs);
	return static_cast<int>(active_coeff_indices.size());
}

int MLMTPR::ScalingSlopeOffset(int scaling_block, int type_central, int type_outer) const
{
	const int pair_count = species_count * species_count;
	return species_count + 2 * scaling_block * pair_count + type_central * species_count + type_outer;
}

int MLMTPR::ScalingShiftOffset(int scaling_block, int type_central, int type_outer) const
{
	const int pair_count = species_count * species_count;
	return ScalingSlopeOffset(scaling_block, type_central, type_outer) + pair_count;
}

double MLMTPR::OrderedPairStrength(int type_central, int type_outer) const
{
	return std::sqrt(static_cast<double>(type_central + 1))
	     + std::sqrt(static_cast<double>(type_outer + 1));
}

double MLMTPR::NormalizedOrderedPairStrength(int type_central, int type_outer) const
{
	if (species_count <= 1)
		return 0.5;

	const double min_strength = OrderedPairStrength(0, 0);
	const double max_strength = OrderedPairStrength(species_count - 1, species_count - 1);
	if (max_strength <= min_strength)
		return 0.5;

	return Clamp01((OrderedPairStrength(type_central, type_outer) - min_strength) /
	               (max_strength - min_strength));
}

void MLMTPR::SetScalingSlopeRange(double start, double end)
{
	custom_scaling_slope_range = true;
	scaling_slope_range_start = start;
	scaling_slope_range_end = end;
}

void MLMTPR::SetScalingShiftRange(double start, double end)
{
	custom_scaling_shift_range = true;
	scaling_shift_range_start = start;
	scaling_shift_range_end = end;
}

void MLMTPR::InitializeDefaultScalingCoeffs()
{
	const double scal_start = ScalingSlopeRangeStart(*this);
	const double scal_end = ScalingSlopeRangeEnd(*this);
	const double shift_start = ScalingShiftRangeStart(*this);
	const double shift_end = ScalingShiftRangeEnd(*this);
	for (int type_central = 0; type_central < species_count; ++type_central) {
		for (int type_outer = 0; type_outer < species_count; ++type_outer) {
			for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
				const double strength = NormalizedOrderedPairStrength(type_central, type_outer);
				regression_coeffs[ScalingSlopeOffset(scaling_block, type_central, type_outer)] =
					Lerp(scal_start, scal_end, strength);
				regression_coeffs[ScalingShiftOffset(scaling_block, type_central, type_outer)] =
					Lerp(shift_start, shift_end, strength);
			}
		}
	}
}

void MLMTPR::RandomizeScalingCoeffs(std::mt19937_64& generator, double strength_jitter)
{
	const double scal_start = ScalingSlopeRangeStart(*this);
	const double scal_end = ScalingSlopeRangeEnd(*this);
	const double shift_start = ScalingShiftRangeStart(*this);
	const double shift_end = ScalingShiftRangeEnd(*this);
	std::uniform_real_distribution<double> jitter(-strength_jitter, strength_jitter);
	for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
		for (int type_central = 0; type_central < species_count; ++type_central) {
			for (int type_outer = 0; type_outer < species_count; ++type_outer) {
				const double strength = NormalizedOrderedPairStrength(type_central, type_outer);
				const double sample = Clamp01(strength + jitter(generator));
				regression_coeffs[ScalingSlopeOffset(scaling_block, type_central, type_outer)] =
					Lerp(scal_start, scal_end, sample);
				regression_coeffs[ScalingShiftOffset(scaling_block, type_central, type_outer)] =
					Lerp(shift_start, shift_end, sample);
			}
		}
	}
}

void MLMTPR::RandomizeRadialCoeffs(std::mt19937_64& generator, double radial_scale)
{
	std::uniform_real_distribution<double> uniform(-1.0, 1.0);
	const int radial_coeff_offset = RadialCoeffOffset();
	const int block_size = RadialCoeffBlockSize();
	const int rb_size = p_RadialBasis->rb_size;

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int block_offset = radial_coeff_offset + mu * block_size;
		for (int xi = 0; xi < rb_size; ++xi)
			regression_coeffs[block_offset + xi] = radial_scale * uniform(generator);
		for (int type = 0; type < species_count; ++type)
			regression_coeffs[block_offset + rb_size + type] = 1.0;
	}
}

void MLMTPR::RandomizeNonlinearCoeffs(std::mt19937_64& generator, double radial_scale, bool include_scaling, double scaling_strength_jitter)
{
	if (include_scaling)
		RandomizeScalingCoeffs(generator, scaling_strength_jitter);
	RandomizeRadialCoeffs(generator, radial_scale);
}

void MLMTPR::PruneSpecies(const std::vector<int>& old_species_indices)
{
	if (!HasCompleteParameters())
		ERROR("PruneSpecies requires a complete trained model with shift/scal/radial/linear coefficients.");
	if (old_species_indices.empty())
		ERROR("PruneSpecies requires at least one species index.");

	const int old_species_count = species_count;
	const int new_species_count = static_cast<int>(old_species_indices.size());
	std::vector<int> sorted_indices = old_species_indices;
	std::sort(sorted_indices.begin(), sorted_indices.end());
	for (int old_type : old_species_indices) {
		if (old_type < 0 || old_type >= old_species_count)
			ERROR("PruneSpecies species index is out of range.");
	}
	for (int i = 1; i < new_species_count; ++i) {
		if (sorted_indices[i] == sorted_indices[i - 1])
			ERROR("PruneSpecies species indices should be unique.");
	}

	const int rb_size = p_RadialBasis->rb_size;
	const int old_pair_count = old_species_count * old_species_count;
	const int new_pair_count = new_species_count * new_species_count;
	const int old_scaling_offset = old_species_count;
	const int new_scaling_offset = new_species_count;
	const int old_radial_offset = old_species_count + 2 * old_pair_count * K_;
	const int new_radial_offset = new_species_count + 2 * new_pair_count * K_;
	const int old_radial_block_size = rb_size + old_species_count;
	const int new_radial_block_size = rb_size + new_species_count;
	const int old_env_offset = old_radial_offset + radial_func_count * old_radial_block_size;
	const int new_env_offset = new_radial_offset + radial_func_count * new_radial_block_size;
	const int old_env_count = env_gate_enabled ? 1 + old_species_count * env_gate_channel_count : 0;
	const int new_env_count = env_gate_enabled ? 1 + new_species_count * env_gate_channel_count : 0;
	const int old_linear_offset = old_env_offset + old_env_count;
	const int new_linear_offset = new_env_offset + new_env_count;
	const int old_linear_count = alpha_count + old_species_count - 1;
	const int new_linear_count = alpha_count + new_species_count - 1;

	if (static_cast<int>(regression_coeffs.size()) < old_linear_offset + old_linear_count)
		ERROR("PruneSpecies found an inconsistent regression coefficient layout.");

	const std::vector<double> old_coeffs = regression_coeffs;
	std::vector<double> new_coeffs(new_linear_offset + new_linear_count, 0.0);

	for (int new_type = 0; new_type < new_species_count; ++new_type) {
		const int old_type = old_species_indices[new_type];
		new_coeffs[new_type] = old_coeffs[old_type];
	}

	for (int scaling_block = 0; scaling_block < K_; ++scaling_block) {
		for (int new_center = 0; new_center < new_species_count; ++new_center) {
			const int old_center = old_species_indices[new_center];
			for (int new_outer = 0; new_outer < new_species_count; ++new_outer) {
				const int old_outer = old_species_indices[new_outer];
				const int old_slope_offset = old_scaling_offset
					+ 2 * scaling_block * old_pair_count
					+ old_center * old_species_count
					+ old_outer;
				const int old_shift_offset = old_slope_offset + old_pair_count;
				const int new_slope_offset = new_scaling_offset
					+ 2 * scaling_block * new_pair_count
					+ new_center * new_species_count
					+ new_outer;
				const int new_shift_offset = new_slope_offset + new_pair_count;
				new_coeffs[new_slope_offset] = old_coeffs[old_slope_offset];
				new_coeffs[new_shift_offset] = old_coeffs[old_shift_offset];
			}
		}
	}

	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int old_block_offset = old_radial_offset + mu * old_radial_block_size;
		const int new_block_offset = new_radial_offset + mu * new_radial_block_size;
		for (int xi = 0; xi < rb_size; ++xi)
			new_coeffs[new_block_offset + xi] = old_coeffs[old_block_offset + xi];
		for (int new_type = 0; new_type < new_species_count; ++new_type) {
			const int old_type = old_species_indices[new_type];
			new_coeffs[new_block_offset + rb_size + new_type] =
				old_coeffs[old_block_offset + rb_size + old_type];
		}
	}

	if (env_gate_enabled) {
		new_coeffs[new_env_offset] = old_coeffs[old_env_offset];
		for (int new_type = 0; new_type < new_species_count; ++new_type) {
			const int old_type = old_species_indices[new_type];
			for (int q = 0; q < env_gate_channel_count; ++q) {
				new_coeffs[new_env_offset + 1 + new_type * env_gate_channel_count + q] =
					old_coeffs[old_env_offset + 1 + old_type * env_gate_channel_count + q];
			}
		}
	}

	for (int new_type = 0; new_type < new_species_count; ++new_type) {
		const int old_type = old_species_indices[new_type];
		new_coeffs[new_linear_offset + new_type] = old_coeffs[old_linear_offset + old_type];
	}
	for (int moment = 0; moment < alpha_count - 1; ++moment) {
		new_coeffs[new_linear_offset + new_species_count + moment] =
			old_coeffs[old_linear_offset + old_species_count + moment];
	}

	species_count = new_species_count;
	regression_coeffs.swap(new_coeffs);
	linear_coeffs.assign(new_linear_count, 0.0);
	for (int i = 0; i < new_linear_count; ++i)
		linear_coeffs[i] = regression_coeffs[new_linear_offset + i];

	radial_list.resize(0, 0, 0);
	radial_der_list.resize(0, 0, 0);
	env_gate_radial_list.resize(0, 0, 0);
	env_gate_radial_der_list.resize(0, 0, 0);
	max_radial.clear();
	inited = true;
	has_shift_coeffs = true;
	has_scal_coeffs = true;
	has_radial_coeffs = true;
	has_linear_coeffs = true;
	has_env_gate_coeffs = env_gate_enabled;
}


void MLMTPR::Load(const string& filename)
{
	alpha_count = 0;
	energy_cmpnts = NULL;
	stress_cmpnts = NULL;
	has_shift_coeffs = false;
	has_scal_coeffs = false;
	has_radial_coeffs = false;
	has_linear_coeffs = false;
	env_gate_enabled = false;
	has_env_gate_coeffs = false;
	env_gate_cutoff_ratio = kEnvGateDefaultCutoffRatio;
	env_gate_channel_count = kEnvGateDefaultChannels;

	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Cannot open " + filename);


	char tmpline[1000];
	string tmpstr;

	ifs.getline(tmpline, 1000);
	int len = (int)((string)tmpline).length();
	if (len > 0 && tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
		tmpline[len - 1] = '\0';

	if ((string)tmpline == "MTP" ||
		((string)tmpline).compare(0, 10, "version = ") != 0)
	{
		// version reading block
		ifs.getline(tmpline, 1000);
		len = (int)((string)tmpline).length();
		if (len > 0 && tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
			tmpline[len - 1] = '\0';
	}
	if (((string)tmpline).compare(0, 10, "version = ") != 0)
		ERROR("MTP file must contain a version header");
	const string mtp_file_version = ((string)tmpline).substr(10);

		// name/description reading block
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		if (tmpstr == "scaling") // optional 
		{
			ifs.ignore(2);
			ifs >> scaling;
			ifs >> tmpstr;
		}
		if (tmpstr != "L") // optional 
		{
			ERROR("Error reading .mtp file");
		}
		ifs.ignore(2);
		ifs >> L;
		ifs >> tmpstr;
		if (tmpstr != "scaling_map")
		{
			ERROR("Error reading .mtp file");
		}
			ifs.ignore(2);
			ifs >> scaling_map;
			ifs >> tmpstr;
		L += 1;   // L=0,1 ...... L_max

	if (tmpstr != "species_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> species_count;

	ifs >> tmpstr;
	if (tmpstr == "potential_tag")
	{
		getline(ifs, tmpstr);
		ifs >> tmpstr;
	}

	if (tmpstr != "radial_basis_type")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> rbasis_type;
	
	if (rbasis_type == "RBChebyshev")
		p_RadialBasis = new RadialBasis_Chebyshev(ifs);
	else if (rbasis_type == "RBChebyshev_repuls")
		p_RadialBasis = new RadialBasis_Chebyshev_repuls(ifs);
        else if (rbasis_type == "RBChebyshev_s")
                p_RadialBasis = new RadialBasis_Chebyshev_s(ifs);
        else if (rbasis_type == "RBChebyshev_ss")
                p_RadialBasis = new RadialBasis_Chebyshev_ss(ifs);
        else if (rbasis_type == "RBChebyshev_ssw")
                p_RadialBasis = new RadialBasis_Chebyshev_ssw(ifs);
        else if (rbasis_type == "RBChebyshev_sss")
                p_RadialBasis = new RadialBasis_Chebyshev_sss(ifs);
        else if (rbasis_type == "RBChebyshev_sssw")
                p_RadialBasis = new RadialBasis_Chebyshev_sssw(ifs);
		else if (rbasis_type == "RBChebyshev_sss_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sss_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ss_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_ss_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ssw_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_ssw_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_sssw_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sssw_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_s_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_s_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ssss")
                p_RadialBasis = new RadialBasis_Chebyshev_ssss(ifs);
        else if (rbasis_type == "RBChebyshev_sssss")
                p_RadialBasis = new RadialBasis_Chebyshev_sssss(ifs);
        else if (rbasis_type == "RBChebyshev_tanhexp")
                p_RadialBasis = new RadialBasis_Chebyshev_tanhexp(ifs);
        else if (rbasis_type == "RBChebyshev_tanhexp_w")
                p_RadialBasis = new RadialBasis_Chebyshev_tanhexp_w(ifs);
	        else if (rbasis_type == "RBChebyshev_sigma")
	                p_RadialBasis = new RadialBasis_Chebyshev_sigma(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_lmp(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_pos")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_pos(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_pos_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_pos_lmp(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_noenv")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_noenv(ifs);
	        else if (rbasis_type == "RBLaguerre_log1p_noenv_lmp")
	                p_RadialBasis = new RadialBasis_Laguerre_log1p_noenv_lmp(ifs);
	        else if (rbasis_type == "RBJacobi_sss")
	                p_RadialBasis = new RadialBasis_Jacobi_sss(ifs);
	        else if (rbasis_type == "RBJacobi_sss_lmp")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_lmp(ifs);
	        else if (rbasis_type == "RBJacobi_sss_noweight")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_noweight(ifs);
	        else if (rbasis_type == "RBJacobi_sss_noweight_lmp")
	                p_RadialBasis = new RadialBasis_Jacobi_sss_noweight_lmp(ifs);
	        else if (rbasis_type == "RBBessel")
	                p_RadialBasis = new RadialBasis_Bessel(ifs);
	        else if (rbasis_type == "RBBessel_sss")
	                p_RadialBasis = new RadialBasis_Bessel_sss(ifs);
        else if (rbasis_type == "RBBesselw")
                p_RadialBasis = new RadialBasis_Besselw(ifs);
        else if (rbasis_type == "RBBessel_sssw")
                p_RadialBasis = new RadialBasis_Bessel_sssw(ifs);
        else if (rbasis_type == "RBChebyshev_Tri")
                p_RadialBasis = new RadialBasis_Chebyshev_Tri(ifs);
	else if (rbasis_type == "RBShapeev")
		p_RadialBasis = new RadialBasis_Shapeev(ifs);
	else if (rbasis_type == "RBTaylor")
		p_RadialBasis = new RadialBasis_Taylor(ifs);
	else
		ERROR("Wrong radial basis type");

	// We do not need double scaling
	if (p_RadialBasis->scaling != 1.0) {
		scaling *= p_RadialBasis->scaling;
		p_RadialBasis->scaling = 1.0;
	}


	ifs >> tmpstr;
	if (tmpstr != "radial_funcs_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> radial_func_count;
	mu_to_K.resize(radial_func_count);
	mu_to_sigma.resize(radial_func_count);
	for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_sigma[n] = n / L;
	}
	if (scaling_map=="K")
	{
		K_ = radial_func_count / L;
	
	for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n / L;
	}
	}
	else if (scaling_map=="L")
	{
		K_ = L;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n % L;
	}

	}
	else if (scaling_map=="LK")
	{
		K_ = radial_func_count;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n;
	}

	}
	else 
	{
		ERROR("Wrong scaling map");
	}



	//Radial coeffs initialization
	int pairs_count = species_count*species_count;           //number of species pairs

	char foo = ' ';
	int foo_int = 0;

	regression_coeffs.resize(species_count +2*pairs_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));

	inited = true;
        
        ifs >> tmpstr;
        if (tmpstr != "shift_coeffs")
        {

                for (int i = 0; i <species_count; i++)
                regression_coeffs[i] = -1.0;

                }
        else
        {has_shift_coeffs = true;
		ifs.ignore(4);

                for (int i = 0; i < species_count; i++){
                        ifs >> regression_coeffs[i] >> foo;}
                        ifs >> tmpstr; }
    
        
//	ifs >> tmpstr;
        if (tmpstr != "scal_coeffs")
        {
           InitializeDefaultScalingCoeffs();
                }
        else
        {has_scal_coeffs = true;
		ifs.ignore(4);
                
                for (int i = 0; i < 2*pairs_count* K_; i++){
                        ifs >> regression_coeffs[species_count+i] >> foo;}
                        ifs >> tmpstr; }
                
	if (tmpstr == "radial_coeffs")
	{
		has_radial_coeffs = true;

		for (int s1 = 0; s1 < 1; s1++)
			for (int s2 = 0; s2 < 1; s2++)
			{
				ifs >> foo_int >> foo >> foo_int;

				double t;

				for (int i = 0; i < radial_func_count; i++)
				{
					ifs >> foo;
					for (int j = 0; j < p_RadialBasis->rb_size + species_count ; j++)
					{
						ifs >> t >> foo;
						regression_coeffs[species_count+2*pairs_count* K_ +i*(p_RadialBasis->rb_size + species_count) + j] = t;

					}

				}

			}

		ifs >> tmpstr;

	}
	else
	{
		//cout << "Radial coeffs not found, initializing defaults" << endl;
		inited = false;
		has_radial_coeffs = false;

		regression_coeffs.resize(species_count+2*species_count*species_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));


		for (pairs_count = 0; pairs_count < 1; pairs_count++)
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size + species_count; j++)
				{	regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
					i*(p_RadialBasis->rb_size + species_count) + j] = 1e-2;
                                        if (j >= p_RadialBasis->rb_size) {regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
			                                        i*(p_RadialBasis->rb_size + species_count) + j] = 1.1; }
                                }             
			//	regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
			//		i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size)] = 1e-3;


			}
	}

	if (tmpstr == "env_gate_type")
	{
		ifs.ignore(2);
		string env_gate_type;
		ifs >> env_gate_type;
		if (env_gate_type != "centered_tanh_screen" && env_gate_type != "centered_exp_screen")
			ERROR("Unsupported env_gate_type in .mtp file");

		ifs >> tmpstr;
		if (tmpstr != "env_gate_cutoff_ratio")
			ERROR("Error reading env_gate_cutoff_ratio");
		ifs.ignore(2);
		ifs >> env_gate_cutoff_ratio;

		ifs >> tmpstr;
		if (tmpstr != "env_gate_channel_count")
			ERROR("Error reading env_gate_channel_count");
		ifs.ignore(2);
		ifs >> env_gate_channel_count;
		if (env_gate_channel_count != kEnvGateDefaultChannels)
			ERROR("centered_tanh_screen env-gate requires env_gate_channel_count = 6");

		env_gate_enabled = true;
		has_env_gate_coeffs = true;
		const int env_offset = EnvGateCoeffOffset();
		const int env_count = EnvGateCoeffCount();
		regression_coeffs.resize(env_offset + env_count);

		ifs >> tmpstr;
		if (tmpstr == "env_gate_scal_coeffs") {
			ifs.ignore(4);
			double ignored_env_scaling_coeff = 0.0;
			for (int i = 0; i < 2 * species_count * species_count; ++i) {
				ifs >> ignored_env_scaling_coeff >> foo;
				if (ifs.fail())
					ERROR("Error reading env_gate_scal_coeffs");
			}
			ifs >> tmpstr;
		}

		if (tmpstr != "env_gate_lambda_raw")
			ERROR("Error reading env_gate_lambda_raw");
		ifs.ignore(2);
		ifs >> regression_coeffs[EnvGateLambdaRawOffset()];

		ifs >> tmpstr;
		if (tmpstr != "env_gate_log_density_coeffs")
			ERROR("Error reading env_gate_log_density_coeffs");
		ifs.ignore(1000, '{');
		for (int type = 0; type < species_count; ++type) {
			ifs.ignore(1000, '{');
			for (int q = 0; q < env_gate_channel_count; ++q) {
				ifs >> regression_coeffs[EnvGateLogDensityCoeffOffset(type, q)] >> foo;
				if (ifs.fail())
					ERROR("Error reading env_gate_log_density_coeffs");
			}
		}
		ifs.ignore(1000, '\n');
		ifs >> tmpstr;
	}
	else if (mtp_file_version.compare(0, 3, "2.0") == 0)
	{
		ERROR("version = 2.0.0 model must contain an env-gate section");
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

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_basic = new int[alpha_index_basic_count][4];
	alpha_index_basic_.comp0.resize(alpha_index_basic_count);
	alpha_index_basic_.comp1.resize(alpha_index_basic_count);
	alpha_index_basic_.comp2.resize(alpha_index_basic_count);
	alpha_index_basic_.comp3.resize(alpha_index_basic_count);
	if (alpha_index_basic == nullptr)
		ERROR("Memory allocation error");

	int radial_func_max = -1;
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_basic[i][0]  >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
		alpha_index_basic_.comp0[i]=alpha_index_basic[i][0];
		alpha_index_basic_.comp1[i]=alpha_index_basic[i][1];
		alpha_index_basic_.comp2[i]=alpha_index_basic[i][2];
		alpha_index_basic_.comp3[i]=alpha_index_basic[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");

		if (alpha_index_basic[i][0]>radial_func_max)
			radial_func_max = alpha_index_basic[i][0];
	}
	

	//cout << radial_func_count << endl;

	if (radial_func_max!=radial_func_count-1)
		ERROR("Wrong number of radial functions specified");

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_times_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_times = new int[alpha_index_times_count][4];
	alpha_index_times_.comp0.resize(alpha_index_times_count);
	alpha_index_times_.comp1.resize(alpha_index_times_count);
	alpha_index_times_.comp2.resize(alpha_index_times_count);
	alpha_index_times_.comp3.resize(alpha_index_times_count);
	if (alpha_index_times == nullptr)
		ERROR("Memory allocation error");

	for (int i = 0; i < alpha_index_times_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
		alpha_index_times_.comp0[i]=alpha_index_times[i][0];
		alpha_index_times_.comp1[i]=alpha_index_times[i][1];
		alpha_index_times_.comp2[i]=alpha_index_times[i][2];
		alpha_index_times_.comp3[i]=alpha_index_times[i][3];
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
		ERROR("Memory allocation error");

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


	//Reading linear coeffs
	ifs >> tmpstr;
	if (tmpstr != "species_coeffs")
	{
		inited = false;
		//cout << "Linear coeffs not found, initializing defaults, species_count = " << species_count << endl;
		linear_coeffs.resize(alpha_count + species_count - 1);
		for (int i = 0; i < alpha_count + species_count - 1; i++)
			linear_coeffs[i] = 1e-3;
	}
	else
	{
		has_linear_coeffs = true;
		ifs.ignore(4);

		linear_coeffs.resize(species_count);
		for (int i = 0; i < species_count; i++)
			ifs >> linear_coeffs[i] >> foo;


		ifs >> tmpstr;

		if (tmpstr != "moment_coeffs")
			ERROR("Cannot read linear coeffs");

		ifs.ignore(2);

		linear_coeffs.resize(alpha_count + species_count - 1);

		ifs.ignore(10, '{');


		for (int i = 0; i < alpha_count - 1; i++)
			ifs >> linear_coeffs[i + species_count] >> foo;

	}
	if (UsesPrecomputedLmpTable(rbasis_type))
	{
		radial_list.resize(species_count * species_count, 200002, radial_func_count);
		radial_der_list.resize(species_count * species_count, 200002, radial_func_count);
		radial_list.set(0);
		radial_der_list.set(0);

		inv_dr = 200000 / p_RadialBasis->max_dist;
		double dr = 1 / inv_dr;
		const int C = species_count;
		const int R = p_RadialBasis->rb_size;
		int k_;
		double factor;
		for (int i = 0; i < C; i++)
		{
			for (int j = 0; j < C; j++)
			{
					for (int n = 0; n < 200001; n++)
					{
						for (int mu = 0; mu < radial_func_count; mu++)
						{
							k_ = mu_to_K[mu];
							const int basis_k = UsesJacobiIndexedBasis()
								? JacobiIndexedBlockForMu(mu)
								: mu_to_sigma[k_];

							p_RadialBasis->RB_Calc(dr * n,
								regression_coeffs[C + 2 * k_ * C * C + C * i + j],
								1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * i + j],
								basis_k);
						for (int xi = 0; xi < R; xi++)
						{
							factor = regression_coeffs[C + 2 * C * C * K_ + mu * (R + C) + xi]
							       * regression_coeffs[C + 2 * C * C * K_ + R + i]
							       * regression_coeffs[C + 2 * C * C * K_ + R + j];
							radial_list(i * C + j, n, mu) += p_RadialBasis->rb_vals[xi] * scaling * factor;
							radial_der_list(i * C + j, n, mu) += p_RadialBasis->rb_ders[xi] * scaling * factor;
						}
					}
				}
			}
		}
		if (HasEnvGate())
		{
			env_gate_radial_list.resize(species_count * species_count, 200002, env_gate_channel_count);
			env_gate_radial_der_list.resize(species_count * species_count, 200002, env_gate_channel_count);
			env_gate_radial_list.set(0);
			env_gate_radial_der_list.set(0);
			const double r_env = env_gate_cutoff_ratio * p_RadialBasis->max_dist;
			if (r_env <= 0.0)
				ERROR("env_gate_cutoff_ratio should be positive.");
			for (int i = 0; i < C; i++)
			{
				for (int j = 0; j < C; j++)
				{
					for (int n = 0; n < 200002; n++)
					{
					const double r = dr * n;
					if (r <= 0.0 || r >= r_env)
						continue;
					const double y = r / r_env;
					const double dy_dr = 1.0 / r_env;
					const double cutoff = 1.0 - r / r_env;
					const double f_env = cutoff * cutoff;
						const double df_dr = -2.0 * cutoff / r_env;
						double bernstein[6];
						double bernstein_der[6];
						BernsteinDegree5(y, bernstein, bernstein_der);
						for (int q = 0; q < env_gate_channel_count; ++q)
						{
							env_gate_radial_list(i * C + j, n, q) = f_env * bernstein[q];
							env_gate_radial_der_list(i * C + j, n, q) =
								df_dr * bernstein[q] + f_env * bernstein_der[q] * dy_dr;
						}
					}
				}
			}
		}
		else
		{
			env_gate_radial_list.resize(0, 0, 0);
			env_gate_radial_der_list.resize(0, 0, 0);
		}
	}
	else
	{
		radial_list.resize(0, 0, 0);
		radial_der_list.resize(0, 0, 0);
		env_gate_radial_list.resize(0, 0, 0);
		env_gate_radial_der_list.resize(0, 0, 0);
	}
		MemAlloc();
		DistributeCoeffs();
}



void MLMTPR::CalcDescriptors(Configuration& cfg, ofstream& ofs)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << nbh.my_type;
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << basis_vals[1 + i] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}

void MLMTPR::CalcpartialE(Configuration& cfg, ofstream& ofs)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << linear_coeffs[species_count + i]*basis_vals[1 + i]*linear_coeffs[nbh.my_type] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}






void MLMTPR::Save(const string& filename)
{
	ofstream ofs(filename);
	ofs.setf(ios::scientific);
	ofs.precision(15);

	ofs << "MTP\n";
	ofs << "version = " << (HasEnvGate() ? "2.0.0" : "1.1.0") << "\n";
	ofs << "potential_name = " << pot_desc << endl;
	if(scaling != 1.0)
		ofs << "scaling = " << scaling << endl;
	ofs << "L = " << L-1 << endl;
	ofs << "scaling_map = " << scaling_map << endl;
	ofs << "species_count = " << species_count << endl;
	ofs << "potential_tag = " << "" << endl;
	p_RadialBasis->WriteRadialBasis(ofs);
	ofs << "\tradial_funcs_count = " << radial_func_count << '\n';
             
        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;


//        ofs << "\tradial_coeffs" << '\n';


        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count* K_; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[species_count+i];
        }
        ofs << '}' << endl;


	ofs << "\tradial_coeffs" << '\n';

	int q = RadialCoeffOffset();
	for (int i = 0; i < 1; i++)
		for (int j = 0; j < 1; j++)
		{
			ofs <<"\t\t"<< i << "-" << j << "\n";

			for (int k = 0; k < radial_func_count; k++)
			{
				ofs << "\t\t\t{";
				for (int l = 0; l < p_RadialBasis->rb_size + species_count; l++)
				{
					ofs << regression_coeffs[q++];
					if (l != p_RadialBasis->rb_size - 1 + species_count)
						ofs << ", ";
					else
						ofs << "}" << endl;
				}
			}
		}

	if (HasEnvGate()) {
		ofs << "env_gate_type = centered_tanh_screen\n";
		ofs << "env_gate_cutoff_ratio = " << env_gate_cutoff_ratio << '\n';
		ofs << "env_gate_channel_count = " << env_gate_channel_count << '\n';
		ofs << "env_gate_lambda_raw = " << regression_coeffs[EnvGateLambdaRawOffset()] << '\n';
		ofs << "env_gate_log_density_coeffs = {";
		for (int type = 0; type < species_count; ++type) {
			if (type != 0)
				ofs << ", ";
			ofs << "{";
			for (int channel = 0; channel < env_gate_channel_count; ++channel) {
				if (channel != 0)
					ofs << ", ";
				ofs << regression_coeffs[EnvGateLogDensityCoeffOffset(type, channel)];
			}
			ofs << "}";
		}
		ofs << "}\n";
	}

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
	for (int i = 0; i < alpha_scalar_moments - 1; i++)
		ofs << alpha_moment_mapping[i] << ", ";
	ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";

	ofs << "species_coeffs = {";
	for (int i = 0; i < species_count; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i];
	}
	ofs << '}' << endl;

	ofs << "moment_coeffs = {";
	for (int i = 0; i < alpha_count - 1; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i + species_count];
	}
	ofs << '}';

	ofs.close();
}


void MLMTPR::Save_2(const string& filename)
{
        if (HasEnvGate())
                ERROR("Save_2 pair-expanded export does not support SUS2-2.0 env-gate models.");
        ofstream ofs(filename);
        ofs.setf(ios::scientific);
        ofs.precision(15);

        ofs << "MTP\n";
        ofs << "version = 1.1.0\n";
        ofs << "potential_name = " << pot_desc << endl;
        if(scaling != 1.0)
                ofs << "scaling = " << scaling << endl;
        ofs << "species_count = " << species_count << endl;
        ofs << "potential_tag = " << "" << endl;
        p_RadialBasis->WriteRadialBasis(ofs);
        ofs << "\tradial_funcs_count = " << radial_func_count << '\n';

        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;
  

        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i+species_count];
        }
        ofs << '}' << endl;


        ofs << "\tradial_coeffs" << '\n';

        int q = species_count*species_count;

        for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                {
                        ofs <<"\t\t"<< i << "-" << j << "\n";

                        for (int k = 0; k < radial_func_count; k++)
                        {
                                ofs << "\t\t\t{";
                                for (int l = 0; l < p_RadialBasis->rb_size ; l++)
                                {
                                        ofs << regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+l]*regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+i]
                                               *regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+j] ;
                                        if (l != p_RadialBasis->rb_size - 1)
                                                ofs << ", ";
                                        else
                                                ofs << "}" << endl;
                                }
                        }
                }

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
        for (int i = 0; i < alpha_scalar_moments - 1; i++)
                ofs << alpha_moment_mapping[i] << ", ";
        ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";

        ofs << "species_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i];
        }
        ofs << '}' << endl;

        ofs << "moment_coeffs = {";
        for (int i = 0; i < alpha_count - 1; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i + species_count];
        }
        ofs << '}';

        ofs.close();
}





void MLMTPR::CalcEFSComponents(Configuration& cfg)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);
	CalcEFSComponents(cfg, neighborhoods);
}

void MLMTPR::CalcEFSComponents(Configuration& cfg, const Neighborhoods& neighborhoods)
{
	int n = alpha_count + species_count - 1;

	if (cfg.size() != forces_cmpnts.size1)
		forces_cmpnts.resize(cfg.size(), n, 3);

	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));
	forces_cmpnts.set(0);
	memset(stress_cmpnts, 0, n * sizeof(stress_cmpnts[0]));

	for (int ind = 0; ind < cfg.size(); ind++) {
		const Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncsDers(nbh);

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

		energy_cmpnts[nbh.my_type] += basis_vals[0];

		for (int k = species_count; k < n; k++) {
			int i = k - species_count + 1;


			energy_cmpnts[k] += basis_vals[i];

			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++) {
					forces_cmpnts(ind, k, a) += basis_ders(i, j, a);
					forces_cmpnts(nbh.inds[j], k, a) -= basis_ders(i, j, a);
				}

			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						stress_cmpnts[k][a][b] -= basis_ders(i, j, a) * nbh.vecs[j][b];
		}
	}
}

void MLMTPR::CalcEComponents(Configuration& cfg)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist); 

	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));

	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

		energy_cmpnts[nbh.my_type] += basis_vals[0];

		for (int k = species_count; k < n; k++)
		{
			int i = k - species_count + 1;
			energy_cmpnts[k] += basis_vals[i];
		}
	}
}

void MLMTPR::CalcBasisFuncs(Neighborhood& Neighborhood, double* bf_vals)
{
	int C = species_count;						//number of different species in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	int type_central = Neighborhood.my_type;
	std::vector<double>& val_ = radial_vals_buffer_;

	if (type_central>=species_count)
				throw MlipException("Too few species count in the MTP potential!");

	const double env_gate = ComputeEnvGate(Neighborhood, nullptr, nullptr, nullptr);

	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
                int type_outer = Neighborhood.types[j];
		
                
//		int type_outer = Neighborhood.types[j];/

			dist_powers_[0] = 1;
			//coords_powers_[0] = Vector3(1, 1, 1);
			coords_powers_x[0]=1;
			coords_powers_y[0]=1;
			coords_powers_z[0]=1;
			for (int k = 1; k < max_alpha_index_basic_; k++) {
				dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
				coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
				coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
				coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];

			}
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					Neighborhood.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					val_[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
			}

		for (int i = 0; i < alpha_index_basic_count; i++) {
			double val = 0;
			int mu = alpha_index_basic_.comp0[i];
			const int radial_base = mu_to_radial_eval_block_[mu] * R;
			/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				p_RadialBasis->rb_vals[xi] *= scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				p_RadialBasis->rb_ders[xi] *= scaling; */


			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val += regression_coeffs[C+2*C*C*K_+ mu * (R+C) + xi] *regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_outer]* val_[radial_base + xi];
		
			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;		

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
		}
	}

	if (HasEnvGate())
		for (int i = 0; i < alpha_index_basic_count; ++i)
			moment_vals[i] *= env_gate;

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	bf_vals[0] = 1.0;  // setting the constant basis function

	for (int i = 0; i < alpha_scalar_moments; i++) 
		bf_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
	
}

void MLMTPR::CalcBasisFuncsDers(const Neighborhood& Neighborhood)
{
	int C = species_count;						//number of different species in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	if (Neighborhood.count != moment_ders.size2)
		moment_ders.resize(alpha_moments_count, Neighborhood.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_ders.set(0);
	int type_central = Neighborhood.my_type;
	std::vector<double>& val_ = radial_vals_buffer_;
	std::vector<double>& der_ = basis_radial_ders_buffer_;
	const auto checked_eval_base = [&](int eval_block, const char* stage) -> int {
		if (eval_block < 0 || eval_block >= static_cast<int>(radial_eval_to_scaling_block_.size()))
			ERROR(std::string("CalcBasisFuncsDers: eval_block out of range at ") + stage);
		const int radial_base = eval_block * R;
		if (radial_base < 0 || radial_base + R > static_cast<int>(val_.size())
			|| radial_base + R > static_cast<int>(der_.size()))
			ERROR(std::string("CalcBasisFuncsDers: radial buffer out of range at ") + stage);
		return radial_base;
	};

	if (type_central>=species_count)
				throw MlipException("Too few species count in the MTP potential!");

	std::vector<Vector3> env_gate_ders;
	const double env_gate = ComputeEnvGate(Neighborhood, &env_gate_ders, nullptr, nullptr);

	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
               // int type_outer = Neighborhood.types[j];
		// calculates vals and ders for j-th atom in the neighborhood
	//	p_RadialBasis->RB_Calc(Neighborhood.dists[j]);
	        int type_outer = Neighborhood.types[j];
	        
                
	//	int type_outer = Neighborhood.types[j];

			dist_powers_[0] = 1;
			coords_powers_x[0]=1;
			coords_powers_y[0]=1;
			coords_powers_z[0]=1;
			for (int k = 1; k < max_alpha_index_basic_; k++) {
				dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
				coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
				coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
				coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];
			}
			
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int radial_base = checked_eval_base(eval_block, "buffer fill");
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					Neighborhood.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++) {
					val_[radial_base + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					der_[radial_base + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			}
			for (int i = 0; i < alpha_index_basic_count; i++) {

				double val = 0;
				double der = 0;
				int mu = alpha_index_basic_.comp0[i];
				if (mu < 0 || mu >= static_cast<int>(mu_to_radial_eval_block_.size()))
					ERROR("CalcBasisFuncsDers: mu out of range");
				const int radial_base = checked_eval_base(mu_to_radial_eval_block_[mu], "basic accumulation");
				/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					p_RadialBasis->rb_vals[xi] *= scaling;
				for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
					p_RadialBasis->rb_ders[xi] *= scaling; */

			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
			{

				// here \phi_xi(r) is RadialBasis::vals[xi]
					val += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * val_[radial_base + xi];
					der += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * der_[radial_base + xi];


			}

			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / Neighborhood.dists[j];

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;

			mult0 *= der / Neighborhood.dists[j];
			moment_ders(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_ders(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_ders(i, j, 2) += mult0 * NeighbVect_j[2];



			if (alpha_index_basic_.comp1[i] != 0) {
				moment_ders(i, j, 0) += val * alpha_index_basic_.comp1[i]
					* coords_powers_x[alpha_index_basic_.comp1[i] - 1]
					* pow1
					* pow2;
			}
			if (alpha_index_basic_.comp2[i] != 0) {
				moment_ders(i, j, 1) += val * alpha_index_basic_.comp2[i]
					* pow0
					* coords_powers_y[alpha_index_basic_.comp2[i] - 1]
					* pow2;
			}
			if (alpha_index_basic_.comp3[i] != 0) {
				moment_ders(i, j, 2) += val * alpha_index_basic_.comp3[i]
					* pow0
					* pow1
					* coords_powers_z[alpha_index_basic_.comp3[i] - 1];
			}
		}
	}

	if (HasEnvGate()) {
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const double raw_moment = moment_vals[i];
			for (int j = 0; j < Neighborhood.count; ++j) {
				for (int a = 0; a < 3; ++a)
					moment_ders(i, j, a) =
						env_gate * moment_ders(i, j, a) + raw_moment * env_gate_ders[j][a];
			}
			moment_vals[i] = env_gate * raw_moment;
		}
	}



	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;

		for (int j = 0; j < Neighborhood.count; j++) {
			for (int a = 0; a < 3; a++) {
				moment_ders(alpha_index_times_.comp3[i], j, a) += val2 * (moment_ders(alpha_index_times_.comp0[i], j, a) * val1 + val0 * moment_ders(alpha_index_times_.comp1[i], j, a));
			}
		}
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	basis_vals[0] = 1.0;  // setting the constant basis function

	if (basis_ders.size2 != Neighborhood.count) // TODO: remove this check?
		basis_ders.resize(alpha_count, Neighborhood.count, 3);
	memset(&basis_ders(0, 0, 0), 0, 3 * Neighborhood.count*sizeof(double));

	for (int i = 0; i < alpha_scalar_moments; i++) {
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
		memcpy(&basis_ders(1 + i, 0, 0), &moment_ders(alpha_moment_mapping[i], 0, 0), 3 * Neighborhood.count*sizeof(double));
	}
}

bool MLMTPR::UsesJacobiIndexedBasis() const
{
	return IsJacobiIndexedBasisType(p_RadialBasis);
}

int MLMTPR::JacobiIndexedBlockForMu(int mu) const
{
	if (!UsesJacobiIndexedBasis())
		return 0;
	if (L < 0)
		ERROR("RBJacobi indexed basis requires non-negative L");
	// Internal L stores the number of angular channels, i.e. (L_max + 1).
	const int block_span = L;
	if (block_span <= 0)
		ERROR("RBJacobi indexed basis requires positive Jacobi block span");
	if (radial_func_count % block_span != 0)
		ERROR("RBJacobi indexed basis requires radial_funcs_count to be divisible by internal L-channel count");
	const int jacobi_block = mu / block_span;
	if (jacobi_block > 5)
		ERROR("RBJacobi indexed basis supports at most 6 Jacobi blocks: k=0..5");
	return jacobi_block;
}

void MLMTPR::MemAlloc()
{
	int n = alpha_count - 1 + species_count;

	energy_cmpnts = new double[n];
	forces_cmpnts.reserve(n * 3);

	stress_cmpnts = (double(*)[3][3])malloc(n * sizeof(stress_cmpnts[0]));

	moment_vals = new double[alpha_moments_count];
	basis_vals = new double[alpha_count];
	site_energy_ders_wrt_moments_.resize(alpha_moments_count);
	grad_mom_vals_.resize(alpha_moments_count);
	grad_dloss_dsenders_.resize(alpha_moments_count);
	grad_dloss_dmom_.resize(alpha_moments_count);

	max_alpha_index_basic_ = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic_ = max(max_alpha_index_basic_,
			alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i]);
	max_alpha_index_basic_++;
	dist_powers_.resize(max_alpha_index_basic_);
	coords_powers_.resize(max_alpha_index_basic_);
	coords_powers_x.resize(max_alpha_index_basic_);
	coords_powers_y.resize(max_alpha_index_basic_);
	coords_powers_z.resize(max_alpha_index_basic_);
	grad_dist_powers_.resize(max_alpha_index_basic_);
	grad_coords_powers_x_.resize(max_alpha_index_basic_);
	grad_coords_powers_y_.resize(max_alpha_index_basic_);
	grad_coords_powers_z_.resize(max_alpha_index_basic_);
	grad_mu_contract_vals_.resize(radial_func_count);
	grad_mu_contract_ders_.resize(radial_func_count);
	grad_mu_contract_ders_s_.resize(radial_func_count);
	grad_mu_contract_ders_ss_.resize(radial_func_count);
	grad_mu_contract_coord_ders_s_.resize(radial_func_count);
	grad_mu_contract_coord_ders_ss_.resize(radial_func_count);
	lmp_radial_vals_buffer_.resize(radial_func_count);
	lmp_radial_ders_buffer_.resize(radial_func_count);
	mu_to_radial_eval_block_.resize(radial_func_count);
	radial_eval_to_scaling_block_.clear();
	radial_eval_to_basis_k_.clear();
	std::map<std::pair<int, int>, int> radial_eval_lookup;
	for (int mu = 0; mu < radial_func_count; ++mu) {
		const int scaling_block = mu_to_K[mu];
		const int basis_k = UsesJacobiIndexedBasis()
			? JacobiIndexedBlockForMu(mu)
			: mu_to_sigma[scaling_block];
		const std::pair<int, int> eval_key(scaling_block, basis_k);
		auto insert_result = radial_eval_lookup.emplace(eval_key, static_cast<int>(radial_eval_lookup.size()));
		const int eval_block = insert_result.first->second;
		mu_to_radial_eval_block_[mu] = eval_block;
		if (insert_result.second) {
			radial_eval_to_scaling_block_.push_back(scaling_block);
			radial_eval_to_basis_k_.push_back(basis_k);
		}
	}
	const int radial_eval_block_count = static_cast<int>(radial_eval_to_scaling_block_.size());
	radial_vals_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size);
	radial_ders_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size * 5);
	basis_radial_ders_buffer_.resize(radial_eval_block_count * p_RadialBasis->rb_size);
	basic_total_degree_cache_.resize(alpha_index_basic_count);
	basic_scaling_block_cache_.resize(alpha_index_basic_count);
	basic_radial_eval_block_cache_.resize(alpha_index_basic_count);
	basic_radial_offset_cache_.resize(alpha_index_basic_count);
	const int radial_coeff_base = species_count + 2 * species_count * species_count * K_;
	const int radial_stride = p_RadialBasis->rb_size + species_count;
	for (int i = 0; i < alpha_index_basic_count; i++) {
		const int mu = alpha_index_basic_.comp0[i];
		basic_total_degree_cache_[i] =
			alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
		basic_scaling_block_cache_[i] = mu_to_K[mu];
		basic_radial_eval_block_cache_[i] = mu_to_radial_eval_block_[mu];
		basic_radial_offset_cache_[i] = radial_coeff_base + mu * radial_stride;
	}

	}

MLMTPR::~MLMTPR()
{
	if (moment_vals != NULL) delete[] moment_vals;
	if (basis_vals != NULL) delete[] basis_vals;

	if (alpha_moment_mapping != NULL) delete[] alpha_moment_mapping;
	if (alpha_index_times != NULL) delete[] alpha_index_times;
	if (alpha_index_basic != NULL) delete[] alpha_index_basic;

	moment_vals = NULL;
	basis_vals = NULL;
	alpha_moment_mapping = NULL;
	alpha_index_times = NULL;
	alpha_index_basic = NULL;

	if (energy_cmpnts != NULL) delete[] energy_cmpnts;
	if (stress_cmpnts != NULL) free(stress_cmpnts);

	if (p_RadialBasis!=NULL)
	delete p_RadialBasis;


}

void MLMTPR::RadialCoeffsInit(ifstream& ifs_rad)
{
	if (!ifs_rad.is_open())
		ERROR("Radial functions file is not open");

	string types, type1, type2;

	int pairs_count = 0;           //number of species pairs

	types = "";
	type1 = "";
	type2 = "";
	char foo = ' ';

	ifs_rad >> types;

	if (types == "radial_coeffs")
	{

		ifs_rad >> types;

		while (!ifs_rad.eof())
		{
			int i = 0;
			pairs_count++;
			regression_coeffs.resize(pairs_count*radial_func_count*(p_RadialBasis->rb_size));

			while (types.at(i) != '-')
				type1 += types.at(i++);

			while (i != types.length() - 1)
			{
				if (types.at(i + 1) != '+')
					type2 += types.at(++i);
				else
					i++;
			}


			double t;
			for (i = 0; i < radial_func_count; i++) 
			{
				ifs_rad >> foo;
				for (int j = 0; j <= p_RadialBasis->rb_size; j++)
				{
					ifs_rad >> t >> foo;
					regression_coeffs[(pairs_count - 1)*radial_func_count*(p_RadialBasis->rb_size) +
						i*(p_RadialBasis->rb_size) + j] = t;


				}


			}

			types = "";
			type1 = "";
			type2 = "";

			ifs_rad >> types;

		};

		if (species_count == 0)
			species_count = (int)sqrt(pairs_count);

		if (abs(species_count*species_count - pairs_count) > 1e-8)
			ERROR("WRONG NUMBER OF RADIAL FUNCTION BLOCKS, MUST BE species_count^2");
	}
	else
	{

		if (species_count == 0)
			species_count = 2;


		//cout << "Radial coeffs not found, initializing defaults" << endl;

		regression_coeffs.resize(species_count*species_count*radial_func_count*(p_RadialBasis->rb_size));


		for (pairs_count = 0; pairs_count < species_count*species_count; pairs_count++)//!!! pairs_count \xEF\xE5\xF0\xE5\xEE\xEF\xF0\xE5\xE4\xE5\xEB\xB8\xED
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size; j++)
					regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + j] = 1e-6;

				regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size-1)] = 1e-3;


			}

	}


}

void MLMTPR::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	buff_site_energy_ = 0.0;
        buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);
    //const auto& radial_list_ref = get_radial_list();
    //const auto& radial_der_list_ref = get_radial_der_list();
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function
    std::vector<double>& val_ = lmp_radial_vals_buffer_;
		std::vector<double>& der_ = lmp_radial_ders_buffer_;
	std::vector<double>& mu_contract_vals = grad_mu_contract_vals_;
	std::vector<double>& mu_contract_ders = grad_mu_contract_ders_;

	if (nbh.count != moment_jacobian_.size2)
		moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_jacobian_.set(0);


	int type_central = nbh.my_type;

	if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

	std::vector<Vector3> env_gate_ders;
	const double env_gate = ComputeEnvGate(nbh, &env_gate_ders, nullptr, nullptr);

	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
//		p_RadialBasis->RB_Calc(nbh.dists[j]);
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_vals[xi] *= scaling;
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_ders[xi] *= scaling;
                int type_outer = nbh.types[j];
                
        const double r=nbh.dists[j];
		dist_powers_[0] = 1;
		coords_powers_x[0]=1;
		coords_powers_y[0]=1;
		coords_powers_z[0]=1;
		for (int k = 1; k < max_alpha_index_basic_; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * r;
			coords_powers_x[k]=coords_powers_x[k-1]*NeighbVect_j[0];
			coords_powers_y[k]=coords_powers_y[k-1]*NeighbVect_j[1];
			coords_powers_z[k]=coords_powers_z[k-1]*NeighbVect_j[2];
		}
		if (UsesPrecomputedLmpTable(p_RadialBasis))
		{
		int r_list;
		r_list=std::floor(r*inv_dr);
		const int r_next=r_list+1;
		const int shift=C * type_central + type_outer;
		const double ddr=r*inv_dr-r_list;



		for (int m = 0; m < K; m++)
		{
		const double v1=radial_list(shift,r_list,m);
		const double v2=radial_list(shift,r_next,m);
		const double d1=radial_der_list(shift,r_list,m);
		const double d2=radial_der_list(shift,r_next,m);

		val_[m] =v1+ddr*(v2-v1);
		der_[m] = d1+ddr*(d2-d1);

		}
		for (int i = 0; i < alpha_index_basic_count; i++)
		{
		double val = 0, der = 0;
        const int comp1=alpha_index_basic_.comp1[i];
        const int comp2=alpha_index_basic_.comp2[i];
        const int comp3=alpha_index_basic_.comp3[i];
		const int mu = alpha_index_basic_.comp0[i];
        val=val_[mu];
        der=der_[mu];
		//val =radial_list(C * type_central + type_outer,r_list,mu);
		//der = radial_der_list(C * type_central + type_outer,r_list,mu);
        const int k = comp1+comp2+comp3;
        //int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / r;

			double pow0 = coords_powers_x[comp1];
			double pow1 = coords_powers_y[comp2];
			double pow2 = coords_powers_z[comp3];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / r;
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];


            moment_jacobian_(i, j, 0) += (comp1 != 0) ?
            val * comp1 * coords_powers_x[comp1 - 1] * pow1 * pow2 : 0.0;

        moment_jacobian_(i, j, 1) += (comp2 != 0) ?
            val * comp2 * pow0 * coords_powers_y[alpha_index_basic_.comp2[i] - 1] * pow2 : 0.0;

        moment_jacobian_(i, j, 2) += (comp3 != 0) ?
            val * comp3 * pow0 * pow1 * coords_powers_z[comp3 - 1] : 0.0;



        }


		}
		else{
			std::vector<double>& val_ = radial_vals_buffer_;
			std::vector<double>& der_ = radial_ders_buffer_;
			FillWithZero(val_);
			FillWithZero(der_);
			for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block)
			{
				const int scaling_block = radial_eval_to_scaling_block_[eval_block];
				const int basis_k = radial_eval_to_basis_k_[eval_block];
				p_RadialBasis->RB_Calc(
					nbh.dists[j],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
					1.0 * regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
					basis_k);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					val_[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
				for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
					der_[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
			}
			const int shared_type_offset = C + 2 * C * C * K_ + R;
			const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
			const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
			const double type_scale = center_type_coeff * outer_type_coeff;

			for (int mu = 0; mu < K; mu++) {
				const int eval_block = mu_to_radial_eval_block_[mu];
				const int radial_offset = C + 2 * C * C * K_ + mu * (R + C);
				const int radial_base = eval_block * R;
				const int deriv_base = 5 * radial_base;
				double dot_val = 0.0;
				double dot_der = 0.0;

			for (int xi = 0; xi < R; xi++) {
				const double radial_coeff = regression_coeffs[radial_offset + xi];
				dot_val += radial_coeff * val_[radial_base + xi];
				dot_der += radial_coeff * der_[deriv_base + xi];
			}

			mu_contract_vals[mu] = dot_val;
			mu_contract_ders[mu] = dot_der;
		}

		for (int i = 0; i < alpha_index_basic_count; i++) {
			double val = 0, der = 0;
			int mu = alpha_index_basic_.comp0[i];
			val = type_scale * mu_contract_vals[mu];
			der = type_scale * mu_contract_ders[mu];
			int k = alpha_index_basic_.comp1[i] + alpha_index_basic_.comp2[i] + alpha_index_basic_.comp3[i];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_x[alpha_index_basic_.comp1[i]];
			double pow1 = coords_powers_y[alpha_index_basic_.comp2[i]];
			double pow2 = coords_powers_z[alpha_index_basic_.comp3[i]];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_jacobian_(i, j, 0) += val * alpha_index_basic_.comp1[i]
					* coords_powers_x[alpha_index_basic_.comp1[i] - 1]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_jacobian_(i, j, 1) += val * alpha_index_basic_.comp2[i]
					* pow0
					* coords_powers_y[alpha_index_basic_.comp2[i] - 1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_jacobian_(i, j, 2) += val * alpha_index_basic_.comp3[i]
					* pow0
					* pow1
					* coords_powers_z[alpha_index_basic_.comp3[i] - 1];
			}
		}

		}

	
		//Repulsive term
		if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
		if (nbh.dists[j] < p_RadialBasis->min_dist)
		{
			double multiplier = 10000;
			buff_site_energy_ += multiplier*(exp(-10*(nbh.dists[j]-1)) - exp(-10*(p_RadialBasis->min_dist-1)));
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1))/ nbh.dists[j])*nbh.vecs[j][a];
		}
	}

	if (HasEnvGate()) {
		for (int i = 0; i < alpha_index_basic_count; ++i) {
			const double raw_moment = moment_vals[i];
			for (int j = 0; j < nbh.count; ++j) {
				for (int a = 0; a < 3; ++a)
					moment_jacobian_(i, j, a) =
						env_gate * moment_jacobian_(i, j, a) + raw_moment * env_gate_ders[j][a];
			}
			moment_vals[i] = env_gate * raw_moment;
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times_.comp0[i]];
		double val1 = moment_vals[alpha_index_times_.comp1[i]];
		int val2 = alpha_index_times_.comp2[i];
		moment_vals[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
	}

	// renewing maximum absolute values
		for (int i = 0; i < alpha_scalar_moments; i++)
			max_linear[i] = max(max_linear[i],abs(linear_coeffs[species_count + i]*moment_vals[alpha_moment_mapping[i]]));


	// convolving with coefficients
	buff_site_energy_ +=  regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];


	for (int i = 0; i < alpha_scalar_moments; i++)
	{	buff_site_energy_ +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
         buff_site_energy_0 +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
}

	//if (wgt_eqtn_forces != 0) //!!! CHECK IT!!!
	{
		// Backpropagation starts

		// Backpropagation step 1: site energy derivative is the corresponding linear combination
		memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

		for (int i = 0; i < alpha_scalar_moments; i++)
			site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i]*linear_mults[i];

		// SAME BUT UNSAFE:
		// memcpy(&site_energy_ders_wrt_moments_[0], &basis_coeffs[1],
		//		alpha_scalar_moments * sizeof(site_energy_ders_wrt_moments_[0]));

		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = moment_vals[alpha_index_times_.comp0[i]];
			double val1 = moment_vals[alpha_index_times_.comp1[i]];
			int val2 = alpha_index_times_.comp2[i];

			site_energy_ders_wrt_moments_[alpha_index_times_.comp1[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times_.comp0[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val1;
		}

		// Backpropagation step 3: multiply by the Jacobian:
		for (int i = 0; i < alpha_index_basic_count; i++)
			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a) * linear_coeffs[nbh.my_type];

	}
}

void MLMTPR::PrepareEvalCaches()
{
	LinCoeff();
}

void MLMTPR::AccumulateCombinationGrad(	const Neighborhood& nbh,
										std::vector<double>& out_grad_accumulator,
										const double se_weight,
										const Vector3* se_ders_weights) 
{
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function

	int coeff_count = LinearCoeffOffset();

	buff_site_energy_ders_.resize(nbh.count);
	out_grad_accumulator.resize(CoeffCount());
	double* grad_out = out_grad_accumulator.data();

	buff_site_energy_ = 0.0;
	FillWithZero(buff_site_energy_ders_);

	{
		site_energy_ders_wrt_moments_.resize(alpha_moments_count);
		std::vector<double>& mom_val = grad_mom_vals_;
		std::vector<double>& dloss_dsenders = grad_dloss_dsenders_;
		std::vector<double>& dloss_dmom = grad_dloss_dmom_;
		std::vector<double>& mu_contract_vals = grad_mu_contract_vals_;
		std::vector<double>& mu_contract_ders = grad_mu_contract_ders_;
			std::vector<double>& mu_contract_ders_s = grad_mu_contract_ders_s_;
			std::vector<double>& mu_contract_ders_ss = grad_mu_contract_ders_ss_;
			std::vector<double>& mu_contract_coord_ders_s = grad_mu_contract_coord_ders_s_;
			std::vector<double>& mu_contract_coord_ders_ss = grad_mu_contract_coord_ders_ss_;
			const double* linear_scalar_coeffs = linear_coeffs.data() + species_count;
			const double* linear_mults_data = linear_mults.data();
			std::vector<double> env_raw_basic_vals;
			std::vector<double> env_raw_sender_dots;
			double env_gate_sender_dot_sum = 0.0;
			if (HasEnvGate())
				env_raw_basic_vals.assign(alpha_index_basic_count, 0.0);
			if (HasEnvGate() && se_ders_weights != nullptr)
				env_raw_sender_dots.assign(alpha_index_basic_count, 0.0);
			FillWithZero(mom_val);
			FillWithZero(site_energy_ders_wrt_moments_);
			FillWithZero(dloss_dsenders);
		FillWithZero(dloss_dmom);
		int type_central = nbh.my_type;

		if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

			std::vector<Vector3> env_gate_ders;
			std::vector<double> env_gate_channel_sums;
			std::vector<double> env_gate_rho_dr;
			const double env_gate = ComputeEnvGate(nbh, &env_gate_ders, &env_gate_channel_sums, &env_gate_rho_dr);
			const int radial_coeff_base = C + 2 * C * C * K_;
			const int shared_type_offset = radial_coeff_base + R;
			const double site_linear_coeff = linear_coeffs[nbh.my_type];
			const size_t neighbor_count = static_cast<size_t>(nbh.count);
			const size_t power_stride = static_cast<size_t>(max_alpha_index_basic_);
			const size_t radial_val_stride = static_cast<size_t>(K_) * R;
			const size_t radial_der_stride = radial_val_stride * 5;
			const size_t mu_stride = static_cast<size_t>(K);

			grad_neighbor_dist_powers_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_x_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_y_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_coords_powers_z_cache_.resize(neighbor_count * power_stride);
			grad_neighbor_radial_vals_cache_.resize(neighbor_count * radial_val_stride);
			grad_neighbor_radial_ders_cache_.resize(neighbor_count * radial_der_stride);
			grad_neighbor_mu_contract_vals_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_s_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_ders_ss_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_coord_ders_s_cache_.resize(neighbor_count * mu_stride);
			grad_neighbor_mu_contract_coord_ders_ss_cache_.resize(neighbor_count * mu_stride);

			struct NeighborGradCache {
				double* dist_powers;
				double* coords_powers_x;
				double* coords_powers_y;
				double* coords_powers_z;
				double* radial_vals;
				double* radial_ders;
				double* mu_contract_vals;
				double* mu_contract_ders;
				double* mu_contract_ders_s;
				double* mu_contract_ders_ss;
				double* mu_contract_coord_ders_s;
				double* mu_contract_coord_ders_ss;
			};

			auto neighbor_cache = [&](int neighbor_index) {
				const size_t offset = static_cast<size_t>(neighbor_index);
				return NeighborGradCache{
					grad_neighbor_dist_powers_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_x_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_y_cache_.data() + offset * power_stride,
					grad_neighbor_coords_powers_z_cache_.data() + offset * power_stride,
					grad_neighbor_radial_vals_cache_.data() + offset * radial_val_stride,
					grad_neighbor_radial_ders_cache_.data() + offset * radial_der_stride,
					grad_neighbor_mu_contract_vals_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_s_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_ders_ss_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_coord_ders_s_cache_.data() + offset * mu_stride,
					grad_neighbor_mu_contract_coord_ders_ss_cache_.data() + offset * mu_stride
				};
			};

			auto fill_neighbor_buffers = [&](int type_outer, double r, const NeighborGradCache& cache) {
				for (int eval_block = 0; eval_block < static_cast<int>(radial_eval_to_scaling_block_.size()); ++eval_block) {
					const int scaling_block = radial_eval_to_scaling_block_[eval_block];
					const int basis_k = radial_eval_to_basis_k_[eval_block];
					p_RadialBasis->RB_Calc(
						r,
						regression_coeffs[C + 2 * scaling_block * C * C + C * type_central + type_outer],
						regression_coeffs[C + 2 * scaling_block * C * C + C * C + C * type_central + type_outer],
						basis_k);
					for (int xi = 0; xi < R; xi++)
						cache.radial_vals[eval_block * R + xi] = p_RadialBasis->rb_vals[xi] * scaling;
					for (int xi = 0; xi < R * 5; xi++)
						cache.radial_ders[eval_block * R * 5 + xi] = p_RadialBasis->rb_ders[xi] * scaling;
				}
			};

			auto prepare_neighbor_state = [&](const Vector3& neighb_vec, int type_outer, double r, const NeighborGradCache& cache) {
				cache.dist_powers[0] = 1.0;
				cache.coords_powers_x[0] = 1.0;
				cache.coords_powers_y[0] = 1.0;
				cache.coords_powers_z[0] = 1.0;
				for (int k = 1; k < max_alpha_index_basic_; k++) {
					cache.dist_powers[k] = cache.dist_powers[k - 1] * r;
					cache.coords_powers_x[k] = cache.coords_powers_x[k - 1] * neighb_vec[0];
					cache.coords_powers_y[k] = cache.coords_powers_y[k - 1] * neighb_vec[1];
					cache.coords_powers_z[k] = cache.coords_powers_z[k - 1] * neighb_vec[2];
				}

				fill_neighbor_buffers(type_outer, r, cache);

					for (int mu = 0; mu < K; mu++) {
						const int eval_block = mu_to_radial_eval_block_[mu];
						const int radial_offset = radial_coeff_base + mu * (R + C);
						const int radial_base = eval_block * R;
						const int deriv_base = 5 * radial_base;
				double dot_val = 0.0;
				double dot_der = 0.0;
				double dot_der_s = 0.0;
				double dot_der_ss = 0.0;
				double dot_coord_der_s = 0.0;
				double dot_coord_der_ss = 0.0;

					for (int xi = 0; xi < R; xi++) {
						const double radial_coeff = regression_coeffs[radial_offset + xi];
						dot_val += radial_coeff * cache.radial_vals[radial_base + xi];
						dot_der += radial_coeff * cache.radial_ders[deriv_base + xi];
						dot_der_s += radial_coeff * cache.radial_ders[deriv_base + xi + R];
						dot_coord_der_s += radial_coeff * cache.radial_ders[deriv_base + xi + 2 * R];
						dot_der_ss += radial_coeff * cache.radial_ders[deriv_base + xi + 3 * R];
						dot_coord_der_ss += radial_coeff * cache.radial_ders[deriv_base + xi + 4 * R];
					}

					cache.mu_contract_vals[mu] = dot_val;
					cache.mu_contract_ders[mu] = dot_der;
					cache.mu_contract_ders_s[mu] = dot_der_s;
					cache.mu_contract_ders_ss[mu] = dot_der_ss;
					cache.mu_contract_coord_ders_s[mu] = dot_coord_der_s;
					cache.mu_contract_coord_ders_ss[mu] = dot_coord_der_ss;
				}
			};

			for (int j = 0; j < nbh.count; j++) {
				const Vector3& neighb_vec = nbh.vecs[j];
				const int type_outer = nbh.types[j];
				const double r = nbh.dists[j];
				const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
				const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
				const double type_scale = center_type_coeff * outer_type_coeff;
				const NeighborGradCache cache = neighbor_cache(j);

				prepare_neighbor_state(neighb_vec, type_outer, r, cache);

				for (int i = 0; i < alpha_index_basic_count; i++) {
					const int mu = alpha_index_basic_.comp0[i];
					const int k = basic_total_degree_cache_[i];
					const double powk = 1.0 / cache.dist_powers[k];
					const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
					const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
					const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
					const double mult0 = pow0 * pow1 * pow2;
					const double val = type_scale * cache.mu_contract_vals[mu] * powk;
					const double der = type_scale * cache.mu_contract_ders[mu];

					mom_val[i] += val * mult0;

					if (se_ders_weights != nullptr && !HasEnvGate()) {
					double local_der = der * powk - k * val / r;
					double jac_x = mult0 * local_der * neighb_vec[0] / r;
					double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

					dloss_dsenders[i] += se_ders_weights[j][0] * jac_x;
					dloss_dsenders[i] += se_ders_weights[j][1] * jac_y;
					dloss_dsenders[i] += se_ders_weights[j][2] * jac_z;
				}
			}

			if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
				const double multiplier = 10000.0;
				buff_site_energy_ += multiplier * (exp(-10 * (r - 1)) - exp(-10 * (p_RadialBasis->min_dist - 1)));
			}
		}

		if (HasEnvGate()) {
			for (int i = 0; i < alpha_index_basic_count; ++i)
				env_raw_basic_vals[i] = mom_val[i];

			if (se_ders_weights != nullptr) {
				for (int j = 0; j < nbh.count; ++j) {
					const Vector3& neighb_vec = nbh.vecs[j];
					const double r = nbh.dists[j];
					const double inv_r = 1.0 / r;
					env_gate_sender_dot_sum += se_ders_weights[j][0] * env_gate_ders[j][0]
						+ se_ders_weights[j][1] * env_gate_ders[j][1]
						+ se_ders_weights[j][2] * env_gate_ders[j][2];
					const int type_outer = nbh.types[j];
					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const NeighborGradCache cache = neighbor_cache(j);
					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const int mu = alpha_index_basic_.comp0[i];
						const int k = basic_total_degree_cache_[i];
						const double powk = 1.0 / cache.dist_powers[k];
						const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
						const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
						const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
						const double mult0 = pow0 * pow1 * pow2;
						const double val = type_scale * cache.mu_contract_vals[mu] * powk;
						const double der = type_scale * cache.mu_contract_ders[mu];
						double local_der = der * powk - k * val / r;
						double jac_x = mult0 * local_der * neighb_vec[0] * inv_r;
						double jac_y = mult0 * local_der * neighb_vec[1] * inv_r;
						double jac_z = mult0 * local_der * neighb_vec[2] * inv_r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

						const double raw_sender_dot = se_ders_weights[j][0] * jac_x
							+ se_ders_weights[j][1] * jac_y
							+ se_ders_weights[j][2] * jac_z;
						env_raw_sender_dots[i] += raw_sender_dot;
						jac_x = env_gate * jac_x + env_raw_basic_vals[i] * env_gate_ders[j][0];
						jac_y = env_gate * jac_y + env_raw_basic_vals[i] * env_gate_ders[j][1];
						jac_z = env_gate * jac_z + env_raw_basic_vals[i] * env_gate_ders[j][2];
						dloss_dsenders[i] += se_ders_weights[j][0] * jac_x
							+ se_ders_weights[j][1] * jac_y
							+ se_ders_weights[j][2] * jac_z;
					}
				}
			}

			for (int i = 0; i < alpha_index_basic_count; ++i)
				mom_val[i] *= env_gate;
		}

		// Next: calculating non-elementary b_i
		for (int i = 0; i < alpha_index_times_count; i++) {
			double val0 = mom_val[alpha_index_times_.comp0[i]];
			double val1 = mom_val[alpha_index_times_.comp1[i]];
			double val2 = alpha_index_times_.comp2[i];
			mom_val[alpha_index_times_.comp3[i]] += val2 * val0 * val1;
		}

			// renewing maximum absolute values
			for (int i = 0; i < alpha_scalar_moments; i++)
				max_linear[i] = max(max_linear[i],abs(linear_scalar_coeffs[i] * mom_val[alpha_moment_mapping[i]]));


			// convolving with coefficients
			buff_site_energy_ +=   regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];
			for (int i = 0; i < alpha_scalar_moments; i++)
				buff_site_energy_ += linear_scalar_coeffs[i] * linear_mults_data[i] * mom_val[alpha_moment_mapping[i]] * linear_coeffs[nbh.my_type];

		// Backpropagation starts

			// Backpropagation step 1: site energy derivative is the corresponding linear combination
			for (int i = 0; i < alpha_scalar_moments; i++)
				site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_scalar_coeffs[i] * linear_mults_data[i];


		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = mom_val[alpha_index_times_.comp0[i]];
			double val1 = mom_val[alpha_index_times_.comp1[i]];
			double val2 = alpha_index_times_.comp2[i];

			site_energy_ders_wrt_moments_[alpha_index_times_.comp1[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times_.comp0[i]] +=
				site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]]
				* val2 * val1;
		}

			for (int i = 0; i < alpha_index_basic_count; i++)
				dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];


			if (se_ders_weights)
			{

				for (int i = 0; i < alpha_index_times_count; i++) {
					const double val0 = mom_val[alpha_index_times_.comp0[i]];
				const double val1 = mom_val[alpha_index_times_.comp1[i]];
				const double val2 = alpha_index_times_.comp2[i];

				dloss_dsenders[alpha_index_times_.comp3[i]] += dloss_dsenders[alpha_index_times_.comp1[i]] * val2 * val0;
				dloss_dsenders[alpha_index_times_.comp3[i]] += dloss_dsenders[alpha_index_times_.comp0[i]] * val2 * val1;


				}

				for (int i = 0; i < alpha_index_times_count; i++) {
					const double val2 = alpha_index_times_.comp2[i];

					dloss_dmom[alpha_index_times_.comp1[i]] += dloss_dsenders[alpha_index_times_.comp0[i]] *
						site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]] * val2;

					dloss_dmom[alpha_index_times_.comp0[i]] += dloss_dsenders[alpha_index_times_.comp1[i]] *
						site_energy_ders_wrt_moments_[alpha_index_times_.comp3[i]] * val2;
				}

				for (int i = alpha_index_times_count - 1; i >= 0; i--) {
					double val0 = mom_val[alpha_index_times_.comp0[i]];
					double val1 = mom_val[alpha_index_times_.comp1[i]];
					const double val2 = alpha_index_times_.comp2[i];

					dloss_dmom[alpha_index_times_.comp0[i]] += dloss_dmom[alpha_index_times_.comp3[i]] * val2 * val1;
					dloss_dmom[alpha_index_times_.comp1[i]] += dloss_dmom[alpha_index_times_.comp3[i]] * val2 * val0;
				}

				}

			if (HasEnvGate()) {
				double gate_prefactor = 0.0;
				for (int i = 0; i < alpha_index_basic_count; ++i)
					gate_prefactor += site_linear_coeff * dloss_dmom[i] * env_raw_basic_vals[i];

				double rho = 0.0;
				for (int q = 0; q < env_gate_channel_count; ++q) {
					const double density_coeff =
						EnvGateDensityCoeff(regression_coeffs[EnvGateLogDensityCoeffOffset(type_central, q)]);
					rho += density_coeff * env_gate_channel_sums[q];
				}
				const double lambda = EnvGateLambda();
				const double tanh_rho = std::tanh(rho);
				const double sech2_rho = 1.0 - tanh_rho * tanh_rho;
				double gate_der_prefactor = 0.0;
				if (se_ders_weights != nullptr) {
					for (int i = 0; i < alpha_index_basic_count; ++i) {
						const double coord_weight = site_linear_coeff * site_energy_ders_wrt_moments_[i];
						gate_prefactor += coord_weight * env_raw_sender_dots[i];
						gate_der_prefactor += coord_weight * env_raw_basic_vals[i];
					}
				}
				const double dgate_dlambda_raw = -lambda * (1.0 - lambda) * tanh_rho;
				grad_out[EnvGateLambdaRawOffset()] += gate_prefactor * dgate_dlambda_raw;
				for (int q = 0; q < env_gate_channel_count; ++q) {
					const double density_coeff_der =
						EnvGateDensityCoeffDer(regression_coeffs[EnvGateLogDensityCoeffOffset(type_central, q)]);
					const double dgate_dlog_density =
						-lambda * sech2_rho * density_coeff_der * env_gate_channel_sums[q];
					grad_out[EnvGateLogDensityCoeffOffset(type_central, q)] +=
						gate_prefactor * dgate_dlog_density;
				}
				if (se_ders_weights != nullptr && gate_der_prefactor != 0.0) {
					const double dgate_der_prefactor_dlambda_raw =
						-lambda * (1.0 - lambda) * sech2_rho;
					for (int j = 0; j < nbh.count; ++j) {
						const double r = nbh.dists[j];
						if (r <= 0.0 || env_gate_rho_dr[j] == 0.0)
							continue;
						const Vector3& neighb_vec = nbh.vecs[j];
						const double sender_projection =
							(se_ders_weights[j][0] * neighb_vec[0]
							 + se_ders_weights[j][1] * neighb_vec[1]
							 + se_ders_weights[j][2] * neighb_vec[2]) / r;
						grad_out[EnvGateLambdaRawOffset()] += gate_der_prefactor
							* sender_projection * dgate_der_prefactor_dlambda_raw
							* env_gate_rho_dr[j];
					}

					const bool use_env_gate_table =
						env_gate_radial_list.size1 == species_count * species_count
						&& env_gate_radial_list.size2 >= 2
						&& env_gate_radial_list.size3 == env_gate_channel_count;
					for (int j = 0; j < nbh.count; ++j) {
						const double r = nbh.dists[j];
						if (r <= 0.0)
							continue;
						const Vector3& neighb_vec = nbh.vecs[j];
						const double sender_projection =
							(se_ders_weights[j][0] * neighb_vec[0]
							 + se_ders_weights[j][1] * neighb_vec[1]
							 + se_ders_weights[j][2] * neighb_vec[2]) / r;
						if (sender_projection == 0.0)
							continue;

						double weighted_basis_der[6] = {0, 0, 0, 0, 0, 0};
						if (use_env_gate_table) {
							int r_list = static_cast<int>(std::floor(r * inv_dr));
							const int last_interval = env_gate_radial_list.size2 - 2;
							if (r_list < 0) r_list = 0;
							if (r_list > last_interval) r_list = last_interval;
							const int r_next = r_list + 1;
							double ddr = r * inv_dr - r_list;
							if (ddr < 0.0) ddr = 0.0;
							if (ddr > 1.0) ddr = 1.0;
							const int pair_index = type_central * species_count + nbh.types[j];
							for (int q = 0; q < env_gate_channel_count; ++q) {
								const double d1 = env_gate_radial_der_list(pair_index, r_list, q);
								const double d2 = env_gate_radial_der_list(pair_index, r_next, q);
								weighted_basis_der[q] = d1 + ddr * (d2 - d1);
							}
						} else {
							const double r_env = env_gate_cutoff_ratio * p_RadialBasis->max_dist;
							if (r >= r_env)
								continue;
							const double y = r / r_env;
							const double dy_dr = 1.0 / r_env;
							const double cutoff = 1.0 - r / r_env;
							const double f_env = cutoff * cutoff;
							const double df_dr = -2.0 * cutoff / r_env;
							double bernstein[6];
							double bernstein_der[6];
							BernsteinDegree5(y, bernstein, bernstein_der);
							for (int q = 0; q < env_gate_channel_count; ++q)
								weighted_basis_der[q] = df_dr * bernstein[q] + f_env * bernstein_der[q] * dy_dr;
						}

						for (int q = 0; q < env_gate_channel_count; ++q) {
							const int coeff_offset = EnvGateLogDensityCoeffOffset(type_central, q);
							const double density_coeff_der =
								EnvGateDensityCoeffDer(regression_coeffs[coeff_offset]);
							const double dgate_der_prefactor_dlog_density =
								2.0 * lambda * tanh_rho * sech2_rho
								* density_coeff_der * env_gate_channel_sums[q];
							const double dgate_der_dlog_density =
								dgate_der_prefactor_dlog_density * env_gate_rho_dr[j]
								- lambda * sech2_rho * density_coeff_der * weighted_basis_der[q];
							grad_out[coeff_offset] += gate_der_prefactor * sender_projection
								* dgate_der_dlog_density;
						}
					}
				}
			}

			FillWithZero(buff_site_energy_ders_);
			if (se_ders_weights == nullptr) {
				for (int j = 0; j < nbh.count; j++) {
					const Vector3& neighb_vec = nbh.vecs[j];
					const int type_outer = nbh.types[j];
					const double r = nbh.dists[j];
					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const NeighborGradCache cache = neighbor_cache(j);

					for (int i = 0; i < alpha_index_basic_count; i++) {
						const int mu = alpha_index_basic_.comp0[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int radial_eval_block = basic_radial_eval_block_cache_[i];
						const int k = basic_total_degree_cache_[i];
						const int radial_offset = basic_radial_offset_cache_[i];
						const double powk = 1.0 / cache.dist_powers[k];
						const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
						const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
						const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
						const double mult0 = pow0 * pow1 * pow2;
						const double dloss_weight = site_linear_coeff * dloss_dmom[i];
						const int radial_base = radial_eval_block * R;
						const double mu_dot_val = cache.mu_contract_vals[mu];
						const double mu_dot_der = cache.mu_contract_ders[mu];
						const double mu_dot_der_s = cache.mu_contract_ders_s[mu];
						const double mu_dot_der_ss = cache.mu_contract_ders_ss[mu];
						const double radial_basis_scale = powk * mult0;
						const double gate_scale = HasEnvGate() ? env_gate : 1.0;
						const double val = type_scale * mu_dot_val * powk;
						const double der = type_scale * mu_dot_der;
						const double center_grad = outer_type_coeff * mu_dot_val * radial_basis_scale;
						const double outer_grad = center_type_coeff * mu_dot_val * radial_basis_scale;
						const double sigma_grad = type_scale * mu_dot_der_s * radial_basis_scale;
						const double sigma_ss_grad = type_scale * mu_dot_der_ss * radial_basis_scale;

						for (int xi = 0; xi < R; xi++) {
							const double rb_val = cache.radial_vals[radial_base + xi];
							const double basic_grad = rb_val * radial_basis_scale;
							grad_out[radial_offset + xi] += dloss_weight * gate_scale * basic_grad * type_scale;
						}

						double local_der = der * powk - k * val / r;
						double jac_x = mult0 * local_der * neighb_vec[0] / r;
						double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

						if (HasEnvGate()) {
							jac_x = env_gate * jac_x + env_raw_basic_vals[i] * env_gate_ders[j][0];
							jac_y = env_gate * jac_y + env_raw_basic_vals[i] * env_gate_ders[j][1];
							jac_z = env_gate * jac_z + env_raw_basic_vals[i] * env_gate_ders[j][2];
						}

						buff_site_energy_ders_[j][0] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_x;
						buff_site_energy_ders_[j][1] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_y;
						buff_site_energy_ders_[j][2] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_z;

						const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
						grad_out[shared_type_offset + type_central] += dloss_weight * gate_scale * center_grad;
						grad_out[shared_type_offset + type_outer] += dloss_weight * gate_scale * outer_grad;
						grad_out[sigma_coeff_offset] += dloss_weight * gate_scale * sigma_grad;
						grad_out[sigma_coeff_offset + C * C] += dloss_weight * gate_scale * sigma_ss_grad;
					}

					if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
						const double multiplier = 10000.0;
						for (int a = 0; a < 3; a++)
							buff_site_energy_ders_[j][a] += -10.0 * multiplier * (exp(-10.0 * (r - 1.0)) / r) * neighb_vec[a];
					}
				}

				if (shift_)
					grad_out[type_central] += se_weight;
				grad_out[coeff_count + type_central] += se_weight * ((buff_site_energy_ - regression_coeffs[nbh.my_type]) / linear_coeffs[nbh.my_type]);

				for (int i = 0; i < alpha_scalar_moments; i++) {
					const int moment_index = alpha_moment_mapping[i];
					grad_out[i + coeff_count + species_count] +=
						linear_coeffs[nbh.my_type] * se_weight * mom_val[moment_index] * linear_mults_data[i];
				}
			} else {
				for (int j = 0; j < nbh.count; j++) {
					const Vector3& neighb_vec = nbh.vecs[j];
					const Vector3& se_weight_vec = se_ders_weights[j];
					const int type_outer = nbh.types[j];
					const double r = nbh.dists[j];
					const double center_type_coeff = regression_coeffs[shared_type_offset + type_central];
					const double outer_type_coeff = regression_coeffs[shared_type_offset + type_outer];
					const double type_scale = center_type_coeff * outer_type_coeff;
					const double inv_r = 1.0 / r;
					const NeighborGradCache cache = neighbor_cache(j);

					for (int i = 0; i < alpha_index_basic_count; i++) {
						const int mu = alpha_index_basic_.comp0[i];
						const int scaling_block = basic_scaling_block_cache_[i];
						const int radial_eval_block = basic_radial_eval_block_cache_[i];
						const int k = basic_total_degree_cache_[i];
						const int radial_offset = basic_radial_offset_cache_[i];
						const double powk = 1.0 / cache.dist_powers[k];
						const double pow0 = cache.coords_powers_x[alpha_index_basic_.comp1[i]];
						const double pow1 = cache.coords_powers_y[alpha_index_basic_.comp2[i]];
						const double pow2 = cache.coords_powers_z[alpha_index_basic_.comp3[i]];
						const double mult0 = pow0 * pow1 * pow2;
						const double dloss_weight = site_linear_coeff * dloss_dmom[i];
						const double coord_weight = site_linear_coeff * site_energy_ders_wrt_moments_[i];
						const int radial_base = radial_eval_block * R;
						const int deriv_base = 5 * radial_base;
						const double mu_dot_val = cache.mu_contract_vals[mu];
						const double mu_dot_der = cache.mu_contract_ders[mu];
						const double mu_dot_der_s = cache.mu_contract_ders_s[mu];
						const double mu_dot_der_ss = cache.mu_contract_ders_ss[mu];
						const double mu_dot_coord_der_s = cache.mu_contract_coord_ders_s[mu];
						const double mu_dot_coord_der_ss = cache.mu_contract_coord_ders_ss[mu];
						const double radial_basis_scale = powk * mult0;
						const double gate_scale = HasEnvGate() ? env_gate : 1.0;
						const double val = type_scale * mu_dot_val * powk;
						const double der = type_scale * mu_dot_der;
						double center_grad = outer_type_coeff * mu_dot_val * radial_basis_scale;
						double outer_grad = center_type_coeff * mu_dot_val * radial_basis_scale;
						double sigma_grad = type_scale * mu_dot_der_s * radial_basis_scale;
						double sigma_ss_grad = type_scale * mu_dot_der_ss * radial_basis_scale;
						double coord_center_grad = 0.0;
						double coord_outer_grad = 0.0;
						double coord_sigma_grad = 0.0;
						double coord_sigma_ss_grad = 0.0;

						for (int xi = 0; xi < R; xi++) {
							const double rb_val = cache.radial_vals[radial_base + xi];
							const double rb_der = cache.radial_ders[deriv_base + xi];
							const double rb_der_s = cache.radial_ders[deriv_base + xi + R];
							const double rb_der_coord_s = cache.radial_ders[deriv_base + xi + 2 * R];
							const double rb_der_ss = cache.radial_ders[deriv_base + xi + 3 * R];
							const double rb_der_coord_ss = cache.radial_ders[deriv_base + xi + 4 * R];
							const double basic_grad = rb_val * radial_basis_scale;
							double radial_grad = dloss_weight * basic_grad;
							double derx = neighb_vec[0] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double dery = neighb_vec[1] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double derz = neighb_vec[2] * inv_r * (rb_der * radial_basis_scale - rb_val * k * radial_basis_scale * inv_r);
							double derx_s = neighb_vec[0] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double dery_s = neighb_vec[1] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double derz_s = neighb_vec[2] * inv_r * (rb_der_coord_s * radial_basis_scale - rb_der_s * k * radial_basis_scale * inv_r);
							double derx_ss = neighb_vec[0] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);
							double dery_ss = neighb_vec[1] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);
							double derz_ss = neighb_vec[2] * inv_r * (rb_der_coord_ss * radial_basis_scale - rb_der_ss * k * radial_basis_scale * inv_r);

							if (alpha_index_basic_.comp1[i] != 0) {
								const double coord_term = alpha_index_basic_.comp1[i]
									* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
									* pow1 * pow2;
								derx += rb_val * powk * coord_term;
								derx_s += rb_der_s * powk * coord_term;
								derx_ss += rb_der_ss * powk * coord_term;
							}
							if (alpha_index_basic_.comp2[i] != 0) {
								const double coord_term = alpha_index_basic_.comp2[i]
									* pow0
									* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
									* pow2;
								dery += rb_val * powk * coord_term;
								dery_s += rb_der_s * powk * coord_term;
								dery_ss += rb_der_ss * powk * coord_term;
							}
							if (alpha_index_basic_.comp3[i] != 0) {
								const double coord_term = alpha_index_basic_.comp3[i]
									* pow0 * pow1
									* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];
								derz += rb_val * powk * coord_term;
								derz_s += rb_der_s * powk * coord_term;
								derz_ss += rb_der_ss * powk * coord_term;
							}

							const double coord_grad = se_weight_vec[0] * derx
								+ se_weight_vec[1] * dery
								+ se_weight_vec[2] * derz;
							const double coord_grad_s = se_weight_vec[0] * derx_s
								+ se_weight_vec[1] * dery_s
								+ se_weight_vec[2] * derz_s;
							const double coord_grad_ss = se_weight_vec[0] * derx_ss
								+ se_weight_vec[1] * dery_ss
								+ se_weight_vec[2] * derz_ss;

							radial_grad = dloss_weight * gate_scale * basic_grad;
							radial_grad += coord_weight * (gate_scale * coord_grad
								+ env_gate_sender_dot_sum * basic_grad);
							grad_out[radial_offset + xi] += radial_grad * type_scale;
							coord_center_grad += gate_scale * outer_type_coeff * coord_grad;
							coord_outer_grad += gate_scale * center_type_coeff * coord_grad;
							coord_sigma_grad += gate_scale * type_scale * coord_grad_s;
							coord_sigma_ss_grad += gate_scale * type_scale * coord_grad_ss;
						}

						double local_der = der * powk - k * val / r;
						double jac_x = mult0 * local_der * neighb_vec[0] / r;
						double jac_y = mult0 * local_der * neighb_vec[1] / r;
						double jac_z = mult0 * local_der * neighb_vec[2] / r;

						if (alpha_index_basic_.comp1[i] != 0)
							jac_x += val * alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
						if (alpha_index_basic_.comp2[i] != 0)
							jac_y += val * alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
						if (alpha_index_basic_.comp3[i] != 0)
							jac_z += val * alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];

						if (HasEnvGate()) {
							jac_x = env_gate * jac_x + env_raw_basic_vals[i] * env_gate_ders[j][0];
							jac_y = env_gate * jac_y + env_raw_basic_vals[i] * env_gate_ders[j][1];
							jac_z = env_gate * jac_z + env_raw_basic_vals[i] * env_gate_ders[j][2];
						}

						buff_site_energy_ders_[j][0] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_x;
						buff_site_energy_ders_[j][1] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_y;
						buff_site_energy_ders_[j][2] += site_linear_coeff * site_energy_ders_wrt_moments_[i] * jac_z;

						double derx = neighb_vec[0] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double dery = neighb_vec[1] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double derz = neighb_vec[2] * inv_r * (mu_dot_der * radial_basis_scale - mu_dot_val * k * radial_basis_scale * inv_r);
						double derx_s = neighb_vec[0] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double dery_s = neighb_vec[1] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double derz_s = neighb_vec[2] * inv_r * (mu_dot_coord_der_s * radial_basis_scale - mu_dot_der_s * k * radial_basis_scale * inv_r);
						double derx_ss = neighb_vec[0] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);
						double dery_ss = neighb_vec[1] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);
						double derz_ss = neighb_vec[2] * inv_r * (mu_dot_coord_der_ss * radial_basis_scale - mu_dot_der_ss * k * radial_basis_scale * inv_r);

						if (alpha_index_basic_.comp1[i] != 0) {
							const double coord_term = alpha_index_basic_.comp1[i]
								* cache.coords_powers_x[alpha_index_basic_.comp1[i] - 1]
								* pow1 * pow2;
							derx += mu_dot_val * powk * coord_term;
							derx_s += mu_dot_der_s * powk * coord_term;
							derx_ss += mu_dot_der_ss * powk * coord_term;
						}
						if (alpha_index_basic_.comp2[i] != 0) {
							const double coord_term = alpha_index_basic_.comp2[i]
								* pow0
								* cache.coords_powers_y[alpha_index_basic_.comp2[i] - 1]
								* pow2;
							dery += mu_dot_val * powk * coord_term;
							dery_s += mu_dot_der_s * powk * coord_term;
							dery_ss += mu_dot_der_ss * powk * coord_term;
						}
						if (alpha_index_basic_.comp3[i] != 0) {
							const double coord_term = alpha_index_basic_.comp3[i]
								* pow0 * pow1
								* cache.coords_powers_z[alpha_index_basic_.comp3[i] - 1];
							derz += mu_dot_val * powk * coord_term;
							derz_s += mu_dot_der_s * powk * coord_term;
							derz_ss += mu_dot_der_ss * powk * coord_term;
						}

						const double coord_grad = se_weight_vec[0] * derx
							+ se_weight_vec[1] * dery
							+ se_weight_vec[2] * derz;
						const double coord_grad_s = se_weight_vec[0] * derx_s
							+ se_weight_vec[1] * dery_s
							+ se_weight_vec[2] * derz_s;
						const double coord_grad_ss = se_weight_vec[0] * derx_ss
							+ se_weight_vec[1] * dery_ss
							+ se_weight_vec[2] * derz_ss;

						const int sigma_coeff_offset = C + 2 * C * C * scaling_block + type_central * C + type_outer;
						grad_out[shared_type_offset + type_central] += dloss_weight * gate_scale * center_grad
							+ coord_weight * (gate_scale * outer_type_coeff * coord_grad
								+ env_gate_sender_dot_sum * center_grad);
						grad_out[shared_type_offset + type_outer] += dloss_weight * gate_scale * outer_grad
							+ coord_weight * (gate_scale * center_type_coeff * coord_grad
								+ env_gate_sender_dot_sum * outer_grad);
						grad_out[sigma_coeff_offset] += dloss_weight * gate_scale * sigma_grad
							+ coord_weight * (gate_scale * type_scale * coord_grad_s
								+ env_gate_sender_dot_sum * sigma_grad);
						grad_out[sigma_coeff_offset + C * C] += dloss_weight * gate_scale * sigma_ss_grad
							+ coord_weight * (gate_scale * type_scale * coord_grad_ss
								+ env_gate_sender_dot_sum * sigma_ss_grad);
					}

					if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls" && r < p_RadialBasis->min_dist) {
						const double multiplier = 10000.0;
						for (int a = 0; a < 3; a++)
							buff_site_energy_ders_[j][a] += -10.0 * multiplier * (exp(-10.0 * (r - 1.0)) / r) * neighb_vec[a];
					}
				}

				if (shift_)
					grad_out[type_central] += se_weight;
				double center_linear_grad = se_weight * ((buff_site_energy_ - regression_coeffs[nbh.my_type]) / linear_coeffs[nbh.my_type]);
				for (int i = 0; i < alpha_scalar_moments; i++)
					center_linear_grad += dloss_dsenders[alpha_moment_mapping[i]] * linear_mults_data[i] * linear_scalar_coeffs[i];
				grad_out[coeff_count + type_central] += center_linear_grad;

				for (int i = 0; i < alpha_scalar_moments; i++) {
					const int moment_index = alpha_moment_mapping[i];
					const double scalar_grad = se_weight * mom_val[moment_index] + dloss_dsenders[moment_index];
					grad_out[i + coeff_count + species_count] +=
						linear_coeffs[nbh.my_type] * scalar_grad * linear_mults_data[i];
				}
			}

	}

}

void MLMTPR::AddPenaltyGrad(const double coeff,													// Must calculate add penalty and optionally (if out_penalty_grad_accumulator != nullptr) its gradient w.r.t. coefficients multiplied by coeff to out_penalty
							double& out_penalty_accumulator,
							Array1D* out_penalty_grad_accumulator)
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	for (int k = 0; k < K; k++)
	{
		double norm = 0;

		for (int i = 0; i < 1; i++)
			for (int j = 0; j < 1; j++)
				for (int l = 0; l < R; l++)
					norm += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];


		out_penalty_accumulator += coeff * (norm - 1) * (norm - 1);

		if (out_penalty_grad_accumulator != nullptr)
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] += coeff * 4 * (norm - 1) * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];
	}



	for (int k1 = 0; k1 < K; k1++)
		for (int k2 = k1 + 1; k2 < K; k2++)
		{
			double scal = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						scal += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*(R+C) + k2*(R+C) + l];

			out_penalty_accumulator += coeff * scal*scal;

			if (out_penalty_grad_accumulator != nullptr)
				for (int i = 0; i < 1; i++)
					for (int j = 0; j < 1; j++)
						for (int l = 0; l < R; l++)
						{
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l];
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l];
						}
		}
}

void MLMTPR::Orthogonalize()
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	for (int k = 0; k < K; k++)
	{
//		for (int k2 = 0; k2 < k; k2++) {
//			double norm = 0;
//
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						norm += regression_coeffs[  k*(R+C) + l] * regression_coeffs[ k*(R+C) + l];
//
//			double scal = 0;
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						scal += regression_coeffs[ k*(R+C) + l] * regression_coeffs[ k2*(R+C) + l];
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						regression_coeffs[ k*(R+C) + l] -= regression_coeffs[ k2*(R+C) + l] * scal;
//		}


		{
			double norm = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						norm += regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l] * regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l]+1e-30;

			norm = sqrt(norm);
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						regression_coeffs[C+2*C*C * K_ + k*(R+C) + l] /= norm;
		}
	}
}

void MLMTPR::Perform_scaling()
{
	//#ifdef MLIP_MPI
	//#else
	//#endif

	int C = species_count;
	int K = radial_func_count;
	int R = p_RadialBasis->rb_size;
	const int linear_offset = LinearCoeffOffset();

	std::vector<double> total_max(alpha_scalar_moments);
	total_max.clear();

	for (int j = 0; j < alpha_scalar_moments; j++)
	{

#ifdef MLIP_MPI
		MPI_Reduce(&max_linear[j], &total_max[j], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&total_max[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		max_linear[j] = total_max[j];
#endif
		regression_coeffs[linear_offset + species_count + j] *= linear_mults[j];

		if (max_linear[j] > 0)
			linear_mults[j] = 1e10 / max_linear[j];
		else
			linear_mults[j] = 1e10;

		linear_mults[j] = max(linear_mults[j], 1e-14);
		linear_mults[j] = min(linear_mults[j], 1e14);

		//cout << "mult=" << linear_mults[j] << endl;

		regression_coeffs[linear_offset + species_count + j] /= linear_mults[j];
		max_linear[j] = 1e-10;
	}
}

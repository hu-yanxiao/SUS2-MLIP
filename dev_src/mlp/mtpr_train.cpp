/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <ctime>
#include <limits>
#include <numeric>
#include <random>
#include <sys/stat.h>
#include <unistd.h>

#include "mtpr_train.h"

using namespace std;

char* mtpfnm;
char* cfgfnm;

int prank = 0;
int psize = 1;

vector<Configuration> training_set;
vector<Configuration> validSet;

struct DatasetStats {
	long long cfg_count = 0;
	long long atom_count = 0;
};

struct MomentCoeffStats {
	double center_log = 0.0;
	double spread_log = 0.0;
	std::size_t count = 0;
	bool valid = false;
};

struct RescaleEvalResult {
	double scaling = 0.0;
	MomentCoeffStats stats;
	std::vector<double> coeffs;
	bool valid = false;
};

constexpr std::size_t kNeighborhoodCacheBudgetBytesPerRank = 4ULL * 1024ULL * 1024ULL * 1024ULL;

namespace {

std::string CurrentTimestamp()
{
	std::time_t now = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
	return std::string(buf);
}

std::string SanitizeShardTag(const std::string& value)
{
	std::string sanitized;
	sanitized.reserve(value.size());
	for (char ch : value) {
		if ((ch >= 'a' && ch <= 'z') ||
		    (ch >= 'A' && ch <= 'Z') ||
		    (ch >= '0' && ch <= '9'))
			sanitized.push_back(ch);
		else
			sanitized.push_back('_');
	}
	return sanitized;
}

std::string MakeShardDir(const std::string& cfgfnm, const std::string& label)
{
	std::string basename = cfgfnm;
	size_t slash = basename.find_last_of("/\\");
	if (slash != std::string::npos)
		basename = basename.substr(slash + 1);
	return ".cfg_shards_" + label + "_" + SanitizeShardTag(basename) + "_" +
		std::to_string(static_cast<long long>(std::time(nullptr))) + "_" +
		std::to_string(static_cast<long long>(getpid()));
}

DatasetStats LoadConfigsFromFile(const std::string& cfgfnm, std::vector<Configuration>& target)
{
	DatasetStats stats;
	target.clear();
	Configuration cfg;
	std::ifstream ifs(cfgfnm, std::ios::binary);
	for (; cfg.Load(ifs); ) {
		target.push_back(cfg);
		stats.cfg_count++;
		stats.atom_count += cfg.size();
	}
	return stats;
}

std::pair<double, double> ParseRangeOption(const std::string& value, const std::string& opt_name)
{
	const std::size_t comma = value.find(',');
	if (comma == std::string::npos || value.find(',', comma + 1) != std::string::npos)
		ERROR(opt_name + " should contain exactly two comma-separated doubles");

	const std::string first = value.substr(0, comma);
	const std::string second = value.substr(comma + 1);
	if (first.empty() || second.empty())
		ERROR(opt_name + " should contain exactly two comma-separated doubles");

	try {
		return std::make_pair(std::stod(first), std::stod(second));
	} catch (const std::exception&) {
		ERROR(opt_name + " should contain exactly two comma-separated doubles");
	}
}

MomentCoeffStats ComputeMomentCoeffStats(const MLMTPR& mtpr)
{
	constexpr double kMomentCoeffEps = 1.0e-12;
	MomentCoeffStats stats;
	std::vector<double> logs;
	logs.reserve(mtpr.linear_coeffs.size());

	for (int i = 0; i < static_cast<int>(mtpr.linear_coeffs.size()) - mtpr.species_count; ++i) {
		const double coeff = std::abs(mtpr.linear_coeffs[i + mtpr.species_count]);
		if (coeff > kMomentCoeffEps)
			logs.push_back(std::log(coeff));
	}

	if (logs.empty())
		return stats;

	std::sort(logs.begin(), logs.end());
	stats.count = logs.size();
	stats.center_log = logs[logs.size() / 2];

	double sq_sum = 0.0;
	for (double value : logs) {
		const double diff = value - stats.center_log;
		sq_sum += diff * diff;
	}
	stats.spread_log = std::sqrt(sq_sum / logs.size());
	stats.valid = true;
	return stats;
}

double RescaleMetric(const MomentCoeffStats& stats)
{
	constexpr double kMetricCenterWeight = 3.0;
	return stats.spread_log * stats.spread_log
	     + kMetricCenterWeight * stats.center_log * stats.center_log;
}

long long SumAtoms(const std::vector<Configuration>& configs)
{
	long long atoms = 0;
	for (const Configuration& cfg : configs)
		atoms += cfg.size();
	return atoms;
}

long long EstimateConfigWork(const Configuration& cfg)
{
	return cfg.size();
}

void LogShardBalance(const std::string& label,
					 const std::vector<long long>& cfg_counts,
					 const std::vector<long long>& atom_counts,
					 const std::vector<long long>& cost_counts)
{
	if (cfg_counts.empty())
		return;

	const auto cfg_minmax = std::minmax_element(cfg_counts.begin(), cfg_counts.end());
	const auto atom_minmax = std::minmax_element(atom_counts.begin(), atom_counts.end());
	const auto cost_minmax = std::minmax_element(cost_counts.begin(), cost_counts.end());
	const double atom_avg = std::accumulate(atom_counts.begin(), atom_counts.end(), 0.0) / atom_counts.size();
	const double cost_avg = std::accumulate(cost_counts.begin(), cost_counts.end(), 0.0) / cost_counts.size();

	std::cout << "[" << CurrentTimestamp() << "] "
			  << label
			  << " shard balance: cfg[min,max]=" << *cfg_minmax.first << "," << *cfg_minmax.second
			  << " atoms[min,max,avg]=" << *atom_minmax.first << "," << *atom_minmax.second << "," << atom_avg
			  << " load[min,max,avg]=" << *cost_minmax.first << "," << *cost_minmax.second << "," << cost_avg
			  << std::endl;
}

long long GlobalAtomCount(const std::vector<Configuration>& configs)
{
	long long atoms = SumAtoms(configs);
#ifdef MLIP_MPI
	long long total_atoms = 0;
	MPI_Allreduce(&atoms, &total_atoms, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	return total_atoms;
#else
	return atoms;
#endif
}

std::vector<Neighborhoods> BuildNeighborhoods(const std::vector<Configuration>& configs, double cutoff)
{
	std::vector<Neighborhoods> neighborhoods;
	neighborhoods.reserve(configs.size());
	for (size_t idx = 0; idx < configs.size(); ++idx) {
		const Configuration& cfg = configs[idx];
		double min_lattice_len = 0.0;
		double max_lattice_len = 0.0;
		if (cfg.lattice.det() != 0.0) {
			min_lattice_len = max_lattice_len = std::sqrt(
				cfg.lattice[0][0] * cfg.lattice[0][0] +
				cfg.lattice[0][1] * cfg.lattice[0][1] +
				cfg.lattice[0][2] * cfg.lattice[0][2]);
			for (int a = 0; a < 3; ++a) {
				const double len = std::sqrt(
					cfg.lattice[a][0] * cfg.lattice[a][0] +
					cfg.lattice[a][1] * cfg.lattice[a][1] +
					cfg.lattice[a][2] * cfg.lattice[a][2]);
				min_lattice_len = std::min(min_lattice_len, len);
				max_lattice_len = std::max(max_lattice_len, len);
			}
		}
		const auto start_time = std::chrono::steady_clock::now();
		neighborhoods.emplace_back(cfg, cutoff);
		const double elapsed_seconds = std::chrono::duration<double>(
			std::chrono::steady_clock::now() - start_time).count();
		if (elapsed_seconds > 10.0) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "Neighborhood build done"
			          << " rank=" << prank
			          << " local_cfg=" << idx
			          << " atoms=" << cfg.size()
			          << " elapsed_s=" << elapsed_seconds
			          << std::endl;
		}
	}
	return neighborhoods;
}

bool TryBuildNeighborhoods(const std::vector<Configuration>& configs,
						   double cutoff,
						   std::vector<Neighborhoods>& neighborhoods_out)
{
	try {
		neighborhoods_out = BuildNeighborhoods(configs, cutoff);
		return true;
	}
	catch (const std::bad_alloc&) {
		neighborhoods_out.clear();
		neighborhoods_out.shrink_to_fit();
		return false;
	}
}

double EstimateConfigurationVolume(const Configuration& cfg)
{
	double volume = std::abs(cfg.lattice.det());
	if (volume > 1.0e-12)
		return volume;

	if (cfg.size() == 0)
		return 0.0;

	Vector3 min_pos = cfg.pos(0);
	Vector3 max_pos = cfg.pos(0);
	for (int i = 1; i < cfg.size(); ++i) {
		for (int a = 0; a < 3; ++a) {
			min_pos[a] = std::min(min_pos[a], cfg.pos(i, a));
			max_pos[a] = std::max(max_pos[a], cfg.pos(i, a));
		}
	}

	const double dx = std::max(1.0e-6, max_pos[0] - min_pos[0]);
	const double dy = std::max(1.0e-6, max_pos[1] - min_pos[1]);
	const double dz = std::max(1.0e-6, max_pos[2] - min_pos[2]);
	return dx * dy * dz;
}

std::size_t EstimateNeighborhoodCacheBytes(const std::vector<Configuration>& configs, double cutoff)
{
	const double sphere_volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;
	long double total_bytes = 0.0L;
	for (const Configuration& cfg : configs) {
		if (cfg.size() == 0)
			continue;

		const double volume = EstimateConfigurationVolume(cfg);
		const double number_density = volume > 1.0e-12 ? static_cast<double>(cfg.size()) / volume : 0.0;
		const double expected_neighbors = std::max(1.0, sphere_volume * number_density);
		const long double entries = static_cast<long double>(cfg.size()) * expected_neighbors;
		total_bytes += static_cast<long double>(cfg.size()) * 160.0L;
		total_bytes += entries * 96.0L;
	}
	if (total_bytes < 0.0L)
		return 0;
	if (total_bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
		return std::numeric_limits<std::size_t>::max();
	return static_cast<std::size_t>(total_bytes);
}

bool ShouldCacheNeighborhoods(const std::vector<Configuration>& configs, double cutoff, std::size_t* estimated_bytes = nullptr)
{
	const std::size_t bytes = EstimateNeighborhoodCacheBytes(configs, cutoff);
	if (estimated_bytes != nullptr)
		*estimated_bytes = bytes;

	const char* env = std::getenv("MLIP_CACHE_NEIGHBORHOODS");
	if (env != nullptr) {
		if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0)
			return false;
		if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0)
			return true;
	}
	return bytes <= kNeighborhoodCacheBudgetBytesPerRank;
}

std::vector<size_t> BuildRescaleSubsetIndices(const std::vector<Configuration>& source
#ifdef MLIP_MPI
											   , MPI_Comm comm,
											   int comm_size
#endif
)
{
	constexpr size_t kRescaleCfgLimitPerRank = 5;

	if (source.empty())
		return {};

	long long global_cfg_count = static_cast<long long>(source.size());
#ifdef MLIP_MPI
	long long local_cfg_count = static_cast<long long>(source.size());
	MPI_Allreduce(&local_cfg_count, &global_cfg_count, 1, MPI_LONG_LONG_INT, MPI_SUM, comm);
	const int active_size = std::max(comm_size, 1);
#else
	const int active_size = 1;
#endif
	if (global_cfg_count <= static_cast<long long>(kRescaleCfgLimitPerRank) * active_size)
		return {};

	std::vector<size_t> order(source.size());
	std::iota(order.begin(), order.end(), 0);
	std::stable_sort(order.begin(), order.end(),
					 [&](size_t lhs, size_t rhs) {
						 if (source[lhs].size() != source[rhs].size())
							 return source[lhs].size() < source[rhs].size();
						 return lhs < rhs;
					 });

	std::vector<size_t> picks;
	picks.reserve(std::min(source.size(), kRescaleCfgLimitPerRank));
	auto push_unique = [&picks](size_t idx) {
		if (std::find(picks.begin(), picks.end(), idx) == picks.end())
			picks.push_back(idx);
	};

	const size_t n = order.size();
	push_unique(order[n - 1]);          // max
	if (n >= 2) push_unique(order[n - 2]); // second max
	push_unique(order[n / 2]);          // median
	if (n >= 2) push_unique(order[1]);  // second min
	push_unique(order[0]);              // min

	return picks;
}

std::vector<Configuration> GatherConfigurationsByIndex(const std::vector<Configuration>& source,
														 const std::vector<size_t>& indices)
{
	std::vector<Configuration> subset;
	subset.reserve(indices.size());
	for (size_t idx : indices)
		subset.push_back(source[idx]);
	return subset;
}

std::vector<Neighborhoods> GatherNeighborhoodsByIndex(const std::vector<Neighborhoods>& source,
														const std::vector<size_t>& indices)
{
	std::vector<Neighborhoods> subset;
	subset.reserve(indices.size());
	for (size_t idx : indices)
		subset.push_back(source[idx]);
	return subset;
}

void BroadcastString(std::string& value)
{
#ifdef MLIP_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int length = static_cast<int>(value.size());
	MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank != 0)
		value.resize(length);
	if (length > 0)
		MPI_Bcast(&value[0], length, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
	(void)value;
#endif
}

DatasetStats LoadConfigsViaRank0Shards(const std::string& cfgfnm,
									   int proc_rank,
									   int proc_size,
									   std::vector<Configuration>& target,
									   const std::string& label)
{
	if (proc_size <= 1)
		return LoadConfigsFromFile(cfgfnm, target);

	DatasetStats stats;
	std::string shard_dir;

	if (proc_rank == 0) {
		shard_dir = MakeShardDir(cfgfnm, label);
		if (mkdir(shard_dir.c_str(), 0777) != 0)
			ERROR("Unable to create shard directory " + shard_dir);

		std::vector<Configuration> all_cfgs;
		{
			Configuration cfg;
			std::ifstream ifs(cfgfnm, std::ios::binary);
			for (; cfg.Load(ifs); )
				all_cfgs.push_back(cfg);
		}
		std::vector<size_t> order(all_cfgs.size());
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(),
						 [&](size_t lhs, size_t rhs) {
							 return EstimateConfigWork(all_cfgs[lhs]) > EstimateConfigWork(all_cfgs[rhs]);
						 });

		std::vector<std::ofstream> shard_streams(proc_size);
		std::vector<long long> shard_cfg_counts(proc_size, 0);
		std::vector<long long> shard_atom_counts(proc_size, 0);
		std::vector<long long> shard_costs(proc_size, 0);
		for (int rank = 0; rank < proc_size; rank++) {
			const std::string shard_path = shard_dir + "/rank_" + std::to_string(rank) + ".bin";
			shard_streams[rank].open(shard_path, std::ios::binary | std::ios::trunc);
			if (!shard_streams[rank].is_open())
				ERROR("Unable to open shard file " + shard_path);
		}

		for (size_t idx : order) {
			const Configuration& cfg = all_cfgs[idx];
			const int owner = static_cast<int>(
				std::min_element(shard_costs.begin(), shard_costs.end()) - shard_costs.begin());
			cfg.SaveBin(shard_streams[owner]);
			shard_cfg_counts[owner] += 1;
			shard_atom_counts[owner] += cfg.size();
			shard_costs[owner] += EstimateConfigWork(cfg);
		}
		for (std::ofstream& ofs : shard_streams)
			ofs.close();
		std::vector<Configuration>().swap(all_cfgs);
		LogShardBalance(label, shard_cfg_counts, shard_atom_counts, shard_costs);
	}

	BroadcastString(shard_dir);

	const std::string local_shard = shard_dir + "/rank_" + std::to_string(proc_rank) + ".bin";
	stats = LoadConfigsFromFile(local_shard, target);

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (proc_rank == 0) {
		for (int rank = 0; rank < proc_size; rank++) {
			const std::string shard_path = shard_dir + "/rank_" + std::to_string(rank) + ".bin";
			std::remove(shard_path.c_str());
		}
		rmdir(shard_dir.c_str());
	}

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	return stats;
}

} // namespace

DatasetStats AddConfigs(const string cfgfnm, NonLinearRegression& dtr, int proc_rank, int proc_size)
{
	(void)dtr;

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	return LoadConfigsViaRank0Shards(cfgfnm, proc_rank, proc_size, training_set, "train");
}

void Rescale(MTPR_trainer& trainer, MLMTPR& mtpr, const std::vector<Neighborhoods>* training_neighborhoods)
{
#ifdef MLIP_MPI
	if (!trainer.TrainRankActive())
		return;
	const int train_rank = trainer.TrainRank();
	const int train_size = trainer.TrainSize();
	MPI_Comm train_comm = trainer.TrainComm();
#else
	const int train_rank = prank;
	const int train_size = 1;
#endif
	const std::vector<size_t> rescale_subset_indices = BuildRescaleSubsetIndices(training_set
#ifdef MLIP_MPI
		, train_comm, train_size
#endif
	);
	std::vector<Configuration> rescale_subset =
		GatherConfigurationsByIndex(training_set, rescale_subset_indices);
	std::vector<Neighborhoods> rescale_subset_neighborhoods;
	const std::vector<Neighborhoods>* full_neighborhoods_ptr = training_neighborhoods;
	if (!rescale_subset.empty()) {
		if (full_neighborhoods_ptr != nullptr && !full_neighborhoods_ptr->empty())
			rescale_subset_neighborhoods = GatherNeighborhoodsByIndex(*full_neighborhoods_ptr, rescale_subset_indices);
		else
			rescale_subset_neighborhoods = BuildNeighborhoods(rescale_subset, mtpr.CutOff());
	}
	std::vector<Configuration>* rescale_training_set = &training_set;
	const std::vector<Neighborhoods>* rescale_training_neighborhoods = full_neighborhoods_ptr;
	if (!rescale_subset.empty())
	{
		rescale_training_set = &rescale_subset;
		rescale_training_neighborhoods = &rescale_subset_neighborhoods;
	}
	const long long rescale_atoms_local = SumAtoms(*rescale_training_set);
	const long long full_atoms_local = SumAtoms(training_set);
	long long rescale_atoms_total = rescale_atoms_local;
	long long full_atoms_total = full_atoms_local;
	long long rescale_cfg_total = static_cast<long long>(rescale_training_set->size());
	long long full_cfg_total = static_cast<long long>(training_set.size());
#ifdef MLIP_MPI
	MPI_Allreduce(&rescale_atoms_local, &rescale_atoms_total, 1, MPI_LONG_LONG_INT, MPI_SUM, train_comm);
	MPI_Allreduce(&full_atoms_local, &full_atoms_total, 1, MPI_LONG_LONG_INT, MPI_SUM, train_comm);
	long long rescale_cfg_local = static_cast<long long>(rescale_training_set->size());
	long long full_cfg_local = static_cast<long long>(training_set.size());
	MPI_Allreduce(&rescale_cfg_local, &rescale_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, train_comm);
	MPI_Allreduce(&full_cfg_local, &full_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, train_comm);
#endif

	constexpr int kCenterMaxIters = 6;
	constexpr int kBracketExpandMaxIters = 6;
	constexpr double kCenterTolLog = 0.15;
	constexpr double kScalingFloor = 1.0e-12;
	constexpr double kBracketFactor = 2.0;
	const double factors[5] = {1.0 / 1.08, 1.0 / 1.03, 1.0, 1.03, 1.08};

	if (train_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] Rescaling..." << std::endl;
		if (rescale_atoms_total < full_atoms_total || rescale_cfg_total < full_cfg_total) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "Rescale subset: "
			          << rescale_cfg_total << "/" << full_cfg_total << " structures, "
			          << rescale_atoms_total << "/" << full_atoms_total << " atoms"
		          << std::endl;
		}
	}

	auto evaluate_scaling = [&](double scaling_value, const std::string& context) {
		RescaleEvalResult result;
		result.scaling = scaling_value;
		result.coeffs.resize(mtpr.CoeffCount());
		mtpr.scaling = scaling_value;
		trainer.TrainLinear(prank,
						   *rescale_training_set,
						   rescale_training_neighborhoods,
						   context);
		std::memcpy(result.coeffs.data(), mtpr.Coeff(), mtpr.CoeffCount() * sizeof(double));
		mtpr.LinCoeff();
		result.stats = ComputeMomentCoeffStats(mtpr);
		result.valid = result.stats.valid;
		return result;
	};

	auto log_eval = [&](const std::string& prefix, const RescaleEvalResult& eval) {
		if (train_rank == 0)
			std::cout << "[" << CurrentTimestamp() << "] "
			          << prefix
			          << " scaling=" << eval.scaling
			          << " count=" << eval.stats.count
			          << " center_log=" << eval.stats.center_log
			          << " spread_log=" << eval.stats.spread_log
			          << " typical_scale=" << std::exp(eval.stats.center_log)
			          << std::endl;
	};

	RescaleEvalResult center_eval =
		evaluate_scaling(mtpr.scaling, "rescale center seed");
	if (!center_eval.valid) {
		if (train_rank == 0)
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "Rescale aborted: no valid moment coefficients" << std::endl;
		return;
	}
	log_eval("Rescale center seed", center_eval);

	RescaleEvalResult best_center_eval = center_eval;
	RescaleEvalResult positive_eval;
	RescaleEvalResult negative_eval;
	bool have_positive = false;
	bool have_negative = false;

	if (center_eval.stats.center_log >= 0.0) {
		positive_eval = center_eval;
		have_positive = true;
	} else {
		negative_eval = center_eval;
		have_negative = true;
	}

	if (std::abs(center_eval.stats.center_log) >= kCenterTolLog) {
		RescaleEvalResult bracket_eval = center_eval;
		for (int iter = 0; iter < kBracketExpandMaxIters && !(have_positive && have_negative); ++iter) {
			double trial_scaling = bracket_eval.scaling;
			if (bracket_eval.stats.center_log > 0.0)
				trial_scaling = std::max(kScalingFloor, bracket_eval.scaling * kBracketFactor);
			else
				trial_scaling = std::max(kScalingFloor, bracket_eval.scaling / kBracketFactor);

			bracket_eval = evaluate_scaling(
				trial_scaling,
				"rescale bracket " + std::to_string(iter + 1) + "/" + std::to_string(kBracketExpandMaxIters));
			if (!bracket_eval.valid)
				break;
			log_eval("Rescale bracket", bracket_eval);
			if (std::abs(bracket_eval.stats.center_log) < std::abs(best_center_eval.stats.center_log))
				best_center_eval = bracket_eval;
			if (bracket_eval.stats.center_log >= 0.0) {
				positive_eval = bracket_eval;
				have_positive = true;
			} else {
				negative_eval = bracket_eval;
				have_negative = true;
			}
		}
	}

	if (have_positive && have_negative) {
		for (int iter = 0; iter < kCenterMaxIters; ++iter) {
			const double log_pos = std::log(positive_eval.scaling);
			const double log_neg = std::log(negative_eval.scaling);
			double log_trial = 0.5 * (log_pos + log_neg);
			const double center_pos = positive_eval.stats.center_log;
			const double center_neg = negative_eval.stats.center_log;
			if (std::abs(center_pos - center_neg) > 1.0e-12) {
				log_trial = (log_pos * center_neg - log_neg * center_pos) / (center_neg - center_pos);
				log_trial = std::max(std::min(log_trial, std::max(log_pos, log_neg)), std::min(log_pos, log_neg));
				const double span = std::abs(log_pos - log_neg);
				if (std::abs(log_trial - log_pos) < 0.1 * span || std::abs(log_trial - log_neg) < 0.1 * span)
					log_trial = 0.5 * (log_pos + log_neg);
			}
			const double trial_scaling = std::max(kScalingFloor, std::exp(log_trial));
			RescaleEvalResult trial_eval = evaluate_scaling(
				trial_scaling,
				"rescale center solve " + std::to_string(iter + 1) + "/" + std::to_string(kCenterMaxIters));
			if (!trial_eval.valid)
				break;
			log_eval("Rescale center solve", trial_eval);
			if (std::abs(trial_eval.stats.center_log) < std::abs(best_center_eval.stats.center_log))
				best_center_eval = trial_eval;
			if (std::abs(trial_eval.stats.center_log) < kCenterTolLog) {
				best_center_eval = trial_eval;
				break;
			}
			if (trial_eval.stats.center_log >= 0.0)
				positive_eval = trial_eval;
			else
				negative_eval = trial_eval;
		}
	}

	mtpr.scaling = best_center_eval.scaling;
	std::memcpy(mtpr.Coeff(), best_center_eval.coeffs.data(), mtpr.CoeffCount() * sizeof(double));

	const double base_scaling = best_center_eval.scaling;
	double best_metric = std::numeric_limits<double>::infinity();
	double best_scaling = base_scaling;
	MomentCoeffStats best_stats = best_center_eval.stats;
	std::vector<double> best_coeffs(mtpr.CoeffCount());
	std::memcpy(best_coeffs.data(), best_center_eval.coeffs.data(), mtpr.CoeffCount() * sizeof(double));

	for (int j = 0; j < 5; ++j) {
		const double trial_scaling = std::max(kScalingFloor, base_scaling * factors[j]);
		RescaleEvalResult trial_eval = evaluate_scaling(
			trial_scaling,
			"rescale fine trial " + std::to_string(j + 1) + "/5");
		if (!trial_eval.valid)
			continue;
		const double metric = RescaleMetric(trial_eval.stats);
		if (train_rank == 0) {
			std::cout << "[" << CurrentTimestamp() << "] "
			          << "Rescale fine stats center_log=" << trial_eval.stats.center_log
			          << " spread_log=" << trial_eval.stats.spread_log
			          << " metric=" << metric
			          << std::endl;
		}
		if (metric < best_metric) {
			best_metric = metric;
			best_scaling = trial_scaling;
			best_stats = trial_eval.stats;
			std::memcpy(best_coeffs.data(), trial_eval.coeffs.data(), mtpr.CoeffCount() * sizeof(double));
		}
	}

	mtpr.scaling = best_scaling;
	if (best_metric < std::numeric_limits<double>::infinity())
		std::memcpy(mtpr.Coeff(), best_coeffs.data(), mtpr.CoeffCount() * sizeof(double));
#ifdef MLIP_MPI
	MPI_Bcast(mtpr.Coeff(), mtpr.CoeffCount(), MPI_DOUBLE, 0, train_comm);
#endif
	if (train_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "Rescaling selected scaling=" << mtpr.scaling;
		if (best_stats.valid) {
			std::cout << " center_log=" << best_stats.center_log
			          << " spread_log=" << best_stats.spread_log
			          << " typical_scale=" << std::exp(best_stats.center_log)
			          << " metric=" << best_metric;
		}
		std::cout << std::endl;
	}
}

void Train_MTPR(std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	//args[0] - potname
	//args[1] - ts_name

	double weight_energy = 1.0;
	if (opts["energy-weight"] != "")
		weight_energy = stod(opts["energy-weight"]);
        double weight_std = 0.0;
                if(opts["std-weight"] != "")
                        weight_std = stod(opts["std-weight"]);
        

        double weight_stdd = 0.0000;
                if(opts["stdd-weight"] != "")
                        weight_stdd = stod(opts["stdd-weight"]);

	double weight_force = 0.01;
	if (opts["force-weight"] != "")
		weight_force = stod(opts["force-weight"]);

	double weight_stress = 0.001;
	if (opts["stress-weight"] != "")
		weight_stress = stod(opts["stress-weight"]);

	double scale_by_force = 0.0;
	if (opts["scale-by-force"] != "")
		scale_by_force = stod(opts["scale-by-force"]);

	string validfnm = "";
	if (opts["valid-cfgs"] != "")
		validfnm = opts["valid-cfgs"];

	int maxits = 1000;
	if (opts["max-iter"] != "")
		maxits = stoi(opts["max-iter"]);

	bool skip_preinit = false;
	if (opts["skip-preinit"] != "")
		skip_preinit = true;
        bool do_shift =true;
        if (opts["shift"] != "")
                do_shift = false;

        bool do_lin = false;
        if (opts["do-lin"] != "")
                do_lin = true;
	bool do_lin_rescale = false;
	if (opts["do-lin-rescale"] != "")
		do_lin_rescale = true;
	bool fine_tune = false;
	if (opts["fine-tune"] != "")
		fine_tune = true;
	int do_lin_step_limit = 1000;
	if (opts["do-lin-steps"] != "")
		do_lin_step_limit = stoi(opts["do-lin-steps"]);
	if (do_lin_step_limit < 0)
		ERROR("--do-lin-steps should be >= 0");
	int do_lin_frequency = 50;
	if (opts["do-lin-freq"] != "")
		do_lin_frequency = stoi(opts["do-lin-freq"]);
	if (do_lin_frequency <= 0)
		ERROR("--do-lin-freq should be > 0");


	string curr_fnm = "";
	if (opts["curr-pot-name"] != "")
		curr_fnm = opts["curr-pot-name"];

	string bfgs_trace_fnm = "";
	if (opts["bfgs-trace-file"] != "")
		bfgs_trace_fnm = opts["bfgs-trace-file"];

	string trained_fnm = "Trained.mtp_";
	if (opts["trained-pot-name"] != "")
		trained_fnm = opts["trained-pot-name"];

	double bfgs_conv_tol = 1e-3;
	if (opts["bfgs-conv-tol"] != "")
		bfgs_conv_tol = stod(opts["bfgs-conv-tol"]);
        
        bool do_sample = true;
	if (opts["do-samp"] != "")
		do_sample = false;


	string weighting = "vibrations";
	if (opts["weighting"] != "")
		weighting = opts["weighting"];

	bool custom_scal_range = false;
	std::pair<double, double> scal_range;
	if (opts["scal-range"] != "") {
		scal_range = ParseRangeOption(opts["scal-range"], "--scal-range");
		custom_scal_range = true;
	}

	bool custom_s_range = false;
	std::pair<double, double> s_range;
	if (opts["s-range"] != "") {
		s_range = ParseRangeOption(opts["s-range"], "--s-range");
		custom_s_range = true;
	}

	if (opts["init-params"] == "")
		opts["init-params"] = "random";
	if (opts["init-params"] != "random" && opts["init-params"] != "same")
		ERROR("--init-params should be 'random' or 'same'");

	bool mindist_update = false;
	if (opts["update-mindist"] != "")
		mindist_update = true;

	SetTagLogStream("dev", &std::cout);
	int end = 1;
	MLMTPR mtpr = MLMTPR();
	if (custom_scal_range)
		mtpr.SetScalingSlopeRange(scal_range.first, scal_range.second);
	if (custom_s_range)
		mtpr.SetScalingShiftRange(s_range.first, s_range.second);
	for (int i = 0; i < end; i++) {
		try {
			mtpr.Load(args[0]);
			end = 1;
		}
		catch (MlipException& exp) {
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}
	if (fine_tune && !mtpr.HasCompleteParameters())
		ERROR("--fine-tune requires a complete trained model with shift/scal/radial/linear coefficients.");
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
#endif

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	MTPR_trainer trainer(&mtpr, weight_energy, weight_force, weight_stress, scale_by_force, 1e-9, curr_fnm, 0);
	//LOSS FUNCTIONAL MODIFICATION!!!
	trainer.weighting = weighting;
        trainer.do_lin=do_lin;
	trainer.do_lin_rescale = do_lin_rescale;
	trainer.do_lin_step_limit = do_lin_step_limit;
	trainer.do_lin_frequency = do_lin_frequency;
	trainer.freeze_scal_coeffs = fine_tune;
	trainer.std_scaling = weight_std;
        trainer.stdd_scaling = weight_stdd;
	trainer.linstop = bfgs_conv_tol;	//if in 100 iterations loss decreases less than this, BFGS is finished
	trainer.curr_pot_name = curr_fnm;
	trainer.bfgs_trace_file = bfgs_trace_fnm;

	if (prank == 0)
		std::cout << "SUS2-MLIP developer version (2026-04-17)"
		          << " | potential from " << args[0]
		          << ", database: " << args[1] << std::endl;
	if (prank == 0 && custom_scal_range)
		std::cout << "scal-range override: " << scal_range.first << ", " << scal_range.second << std::endl;
	if (prank == 0 && custom_s_range)
		std::cout << "s-range override: " << s_range.first << ", " << s_range.second << std::endl;
	if (prank == 0 && fine_tune)
		std::cout << "fine-tune mode enabled: scal_coeffs frozen; initial rescale+linear solve will run before BFGS" << std::endl;

	Configuration cfg;
	DatasetStats train_stats_local;
	DatasetStats valid_stats_local;
	if (validfnm != "")
	{
		try
		{
			valid_stats_local = LoadConfigsViaRank0Shards(validfnm, prank, psize, validSet, "valid");
			if (prank == 0)
				std::cout << "validation set: " << validfnm << std::endl;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
		}
	}

	for (int i = 0; i < end; i++)
	{
		try
		{
#ifdef MLIP_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			training_set.clear();
			train_stats_local = AddConfigs(args[1], trainer, prank, psize);
			end = 1;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}
#ifdef MLIP_MPI
	trainer.ConfigureTrainComm(!training_set.empty(), prank, psize);
#endif

	long long train_cfg_total = train_stats_local.cfg_count;
	long long train_atom_total = train_stats_local.atom_count;
	long long valid_cfg_total = valid_stats_local.cfg_count;
	long long valid_atom_total = valid_stats_local.atom_count;
#ifdef MLIP_MPI
	long long train_cfg_local = train_stats_local.cfg_count;
	long long train_atom_local = train_stats_local.atom_count;
	long long valid_cfg_local = valid_stats_local.cfg_count;
	long long valid_atom_local = valid_stats_local.atom_count;
	MPI_Allreduce(&train_cfg_local, &train_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&train_atom_local, &train_atom_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&valid_cfg_local, &valid_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&valid_atom_local, &valid_atom_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
	std::size_t estimated_cache_bytes = 0;
	const bool cache_linear_neighborhoods = ShouldCacheNeighborhoods(training_set, mtpr.CutOff(), &estimated_cache_bytes);
	std::vector<Neighborhoods> linear_training_neighborhoods;
	const std::vector<Neighborhoods>* linear_training_neighborhoods_ptr = nullptr;
	if (cache_linear_neighborhoods) {
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Building linear neighborhoods..."
			          << " estimated_bytes=" << estimated_cache_bytes
			          << " budget_bytes=" << kNeighborhoodCacheBudgetBytesPerRank
			          << std::endl;
#ifdef MLIP_MPI
		double nbh_build_start = MPI_Wtime();
#else
		double nbh_build_start = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
		const bool built_ok = TryBuildNeighborhoods(training_set, mtpr.CutOff(), linear_training_neighborhoods);
#ifdef MLIP_MPI
		double nbh_build_seconds = MPI_Wtime() - nbh_build_start;
		int built_local = built_ok ? 1 : 0;
		int built_global = 0;
		MPI_Allreduce(&built_local, &built_global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		double nbh_build_min = 0.0;
		double nbh_build_max = 0.0;
		MPI_Reduce(&nbh_build_seconds, &nbh_build_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&nbh_build_seconds, &nbh_build_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
#else
		double nbh_build_seconds = static_cast<double>(clock()) / CLOCKS_PER_SEC - nbh_build_start;
		int built_global = built_ok ? 1 : 0;
		double nbh_build_min = nbh_build_seconds;
		double nbh_build_max = nbh_build_seconds;
#endif
		if (built_global) {
			linear_training_neighborhoods_ptr = &linear_training_neighborhoods;
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] Linear neighborhoods ready"
				          << " build_s[min,max]=" << nbh_build_min << "," << nbh_build_max
				          << std::endl;
		} else if (prank == 0) {
			std::cout << "[" << CurrentTimestamp() << "] Linear neighborhood cache fallback"
			          << " build_s[min,max]=" << nbh_build_min << "," << nbh_build_max
			          << " reason=bad_alloc" << std::endl;
		}
	} else if (prank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] Skipping full neighborhood cache"
		          << " estimated_bytes=" << estimated_cache_bytes
		          << " budget_bytes=" << kNeighborhoodCacheBudgetBytesPerRank
		          << std::endl;
	}

	int radial_first_coeff_repairs = 0;
	if (maxits > 0 && mtpr.has_radial_coeffs) {
		if (prank == 0)
			radial_first_coeff_repairs = mtpr.EnforcePositiveRadialFirstCoeffs();
#ifdef MLIP_MPI
		MPI_Bcast(&radial_first_coeff_repairs, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&mtpr.Coeff()[0], mtpr.CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
		if (radial_first_coeff_repairs > 0) {
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] "
				          << "Repaired " << radial_first_coeff_repairs
				          << " negative/zero radial first coefficients; running one linear solve"
				          << std::endl;
			if (trainer.TrainRankActive())
				trainer.TrainLinear(prank,
				                    training_set,
				                    linear_training_neighborhoods_ptr,
				                    "radial first-coeff sign repair");
			trainer.BroadcastCoeffsWorld();
		}
	}

        if (prank == 0) {std::cout <<"num_of_species: " <<mtpr.species_count  <<std::endl;
                         std::cout <<"training structures: " << train_cfg_total << std::endl;
                         std::cout <<"training atoms: " << train_atom_total << std::endl;
                         if (validfnm != "") {
                                std::cout <<"validation structures: " << valid_cfg_total << std::endl;
                                std::cout <<"validation atoms: " << valid_atom_total << std::endl;
                         }
                         std::cout <<"num_of_parameters: " <<mtpr.ActiveCoeffCount(fine_tune)  <<std::endl;
                         std::cout <<"num_of_stored_coefficients: " <<mtpr.regression_coeffs.size()  <<std::endl;
                         std::cout <<"num_of_scalar_basis_functions: " <<mtpr.alpha_scalar_moments  <<std::endl;
						 std::cout << "num_of_L_channels: " << mtpr.L << std::endl;
						 std::cout << "num_of_scaling_channels:" << mtpr.K_ << std::endl;
						 std::cout << "scaling map:" <<  std::endl;
						 for (int i =0; i<mtpr.mu_to_K.size();i++)
						 {
							std::cout << mtpr.mu_to_K[i]<< "  " ;
						 }
						 std::cout << std::endl;

		                    }
	// Random initialization of nonlinear coefficients
	if (opts["init-params"] == "random" && !mtpr.inited) {
		if (prank == 0) {
			std::random_device rand_device;
			std::mt19937_64 generator(rand_device());

			std::cout << "Random initialization of nonlinear coefficients" << std::endl;
			mtpr.RandomizeNonlinearCoeffs(generator, 5e-3, true, 0.12);
		}
    //     if (prank == 0) {std::cout << mtpr.regression_coeffs[mtpr.radial_func_count*( mtpr.Get_RB_size() + mtpr.species_count)-4] << std::endl;}
#ifdef MLIP_MPI
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Random init coeff broadcast start" << std::endl;
		MPI_Bcast(&mtpr.Coeff()[0], mtpr.CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Random init coeff broadcast done" << std::endl;
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Random init barrier start" << std::endl;
                MPI_Barrier(MPI_COMM_WORLD);
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Random init barrier done" << std::endl;
#endif
	}
         trainer.shift(do_shift);
	if (!mtpr.inited && maxits > 0 && !skip_preinit) {
		trainer.max_step_count = 75;
//		trainer.random_sample(prank, training_set);
//		if (prank == 0){
//		                        mtpr.Save(trained_fnm);
//		                                                mtpr.Save_2("unfixed1.mtp");
//		                                                                  
//
               if (prank == 0){
			std::cout << "[" << CurrentTimestamp() << "] Saving ini.mtp start" << std::endl;
                        mtpr.Save("ini.mtp");
			std::cout << "[" << CurrentTimestamp() << "] Saving ini.mtp done" << std::endl;
                      
                      }

	       if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Entering Rescale" << std::endl;
               if (trainer.TrainRankActive())
	               Rescale(trainer, mtpr, linear_training_neighborhoods_ptr);
               if (do_sample && trainer.TrainRankActive()){
	               trainer.random_sample(prank, training_set, 10, linear_training_neighborhoods_ptr);
               }
		if (prank == 0)
			std::cout << "Pre-training started" << std::endl;

		if (trainer.TrainRankActive())
			trainer.Train(training_set);
		trainer.BroadcastCoeffsWorld();

		
                //trainer.random_sample(prank, training_set, 20);
		if (prank == 0)
			std::cout << "Pre-training ended" << std::endl;
	}

	if (fine_tune && maxits > 0) {
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Fine-tune initial Rescale start" << std::endl;
		if (trainer.TrainRankActive())
			Rescale(trainer, mtpr, linear_training_neighborhoods_ptr);
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Fine-tune initial linear solve start" << std::endl;
		if (trainer.TrainRankActive())
			trainer.TrainLinear(prank,
			                    training_set,
			                    linear_training_neighborhoods_ptr,
			                    "fine-tune initial linear solve");
		trainer.BroadcastCoeffsWorld();
		if (prank == 0)
			std::cout << "[" << CurrentTimestamp() << "] Fine-tune initial rescale+linear solve done" << std::endl;
	}

	//getting the lowest min_dist for the training setZ
	double min_dist = 999;
	for (int i = 0; i < training_set.size(); i++) {
		if (training_set[i].MinDist() < min_dist)
			min_dist = training_set[i].MinDist();
	}

	double total_min_dist = min_dist;

	//finding minimum distance in configuration among the processes
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&min_dist, &total_min_dist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

	if (mindist_update) {
		if (prank == 0)
		{
			std::cout << "Found configuration with mindist=" << total_min_dist << ", MTP's mindist will be decreased\n";
		}
		mtpr.p_RadialBasis->min_dist = 0.99 * total_min_dist;
	}

	trainer.max_step_count = maxits;				//maximum step count (linesearch doesn't count)

	if (maxits > 0) {
		if (prank == 0) {
			std::cout << "Modified by Hu Yanxiao. " << std::endl;
                        std::cout << "BFGS iterations count set to " << trainer.max_step_count << std::endl;
			std::cout << "BFGS convergence tolerance set to " << bfgs_conv_tol << std::endl;
			if (trainer.do_lin) {
				std::cout << "do-lin enabled for first " << trainer.do_lin_step_limit
				          << " BFGS steps, frequency " << trainer.do_lin_frequency << std::endl;
				std::cout << "do-lin-rescale "
				          << (trainer.do_lin_rescale ? "enabled" : "disabled")
				          << std::endl;
			} else {
				std::cout << "do-lin disabled" << std::endl;
			}
                       
			if ((weight_energy != 0) || (weight_force != 0) || (weight_stress != 0)) {
				std::cout << "Energy weight: " << weight_energy << std::endl;
				std::cout << "Force weight: " << weight_force << std::endl;
				std::cout << "Stress weight: " << weight_stress << std::endl;
                                std::cout << "std weight: " << trainer.std_scaling << std::endl;
                                std::cout << "center_std weight: " << trainer.stdd_scaling << std::endl;
			}
		}
                //trainer.std_scaling /= 10000 ;
	//        trainer.random_sample(prank, training_set);
//          	Rescale(trainer, mtpr);
		bool f = true;
		trainer.shift(f);
		if (trainer.TrainRankActive())
			trainer.Train(training_set);
		trainer.BroadcastCoeffsWorld();
		//string train_name = "pp.mtp";
//		if (prank == 0)
//			mtpr.Save("loop_1.mtp");
//		if (prank == 0) {std::cout << "loop_2 start:" << std::endl;}
//		//trainer.std_scaling *= 10 ;
//		trainer.TrainLinear(prank, training_set);
//		Rescale(trainer, mtpr);
 //               trainer.Train(training_set);
  //              if (prank == 0) {mtpr.Save("loop_2.mtp");}
//                        
//                if (prank == 0) {std::cout << "loop_3 start:" << std::endl;}
//                //trainer.std_scaling *= 10 ;
//                trainer.TrainLinear(prank, training_set);
//                Rescale(trainer, mtpr);
//                trainer.Train(training_set);
//                if (prank == 0) {mtpr.Save("loop_3.mtp");}
//                if (prank == 0) {std::cout << "loop_4 start:" << std::endl;}
                //trainer.std_scaling *= 100 ;
//                trainer.TrainLinear(prank, training_set);
 //               Rescale(trainer, mtpr);
  //              trainer.Train(training_set);
   //             if (prank == 0) {mtpr.Save("loop_4.mtp");}

		if (prank == 0){
			mtpr.Save(trained_fnm);
     //                   mtpr.Save_2("unfixed1.mtp");
                      }
     //           Rescale(trainer, mtpr);
     //           if (prank==0){ mtpr.Save_2("unfixed2.mtp");
      //                 }
	}
	ErrorMonitor errmon, bufferrmon;
	std::cout.precision(15);
	bool have_train_summary = trainer.HasLastTrainErrorSummary();
	MTPR_trainer::TrainErrorSummary train_summary;
	if (have_train_summary)
		train_summary = trainer.LastTrainErrorSummary();
#ifdef MLIP_MPI
	int have_train_summary_int = have_train_summary ? 1 : 0;
	MPI_Bcast(&have_train_summary_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
	have_train_summary = (have_train_summary_int != 0);
	if (have_train_summary) {
		double train_summary_values[6];
		if (prank == 0) {
			train_summary_values[0] = train_summary.energy_mae_mev_atom;
			train_summary_values[1] = train_summary.energy_rmse_mev_atom;
			train_summary_values[2] = train_summary.force_mae_mev_a;
			train_summary_values[3] = train_summary.force_rmse_mev_a;
			train_summary_values[4] = train_summary.stress_mae_ev;
			train_summary_values[5] = train_summary.stress_rmse_ev;
		}
		MPI_Bcast(train_summary_values, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		train_summary.energy_mae_mev_atom = train_summary_values[0];
		train_summary.energy_rmse_mev_atom = train_summary_values[1];
		train_summary.force_mae_mev_a = train_summary_values[2];
		train_summary.force_rmse_mev_a = train_summary_values[3];
		train_summary.stress_mae_ev = train_summary_values[4];
		train_summary.stress_rmse_ev = train_summary_values[5];
	}
#endif
	if (have_train_summary) {
		if (prank == 0) {
			std::cout << "\n=== Train Summary ===\n"
			          << std::fixed << std::setprecision(3)
			          << "Structures           : " << train_cfg_total << "\n"
			          << "Atoms                : " << train_atom_total << "\n"
			          << "Energy MAE (meV/atom): " << train_summary.energy_mae_mev_atom << "\n"
			          << "Energy RMSE(meV/atom): " << train_summary.energy_rmse_mev_atom << "\n"
			          << "Force MAE  (meV/A)   : " << train_summary.force_mae_mev_a << "\n"
			          << "Force RMSE (meV/A)   : " << train_summary.force_rmse_mev_a << "\n"
			          << "Stress MAE (eV)      : " << train_summary.stress_mae_ev << "\n"
			          << "Stress RMSE(eV)      : " << train_summary.stress_rmse_ev << "\n"
			          << "=====================\n";
		}
	} else {
		errmon.reset();
		double train_energy_abs_sum = 0.0;
		double train_energy_sq_weighted_sum = 0.0;
		double train_force_abs_component_sum = 0.0;
		double train_force_sq_component_sum = 0.0;
		double train_stress_abs_component_sum = 0.0;
		double train_stress_sq_component_sum = 0.0;
		long long train_force_component_count = 0;
		long long train_stress_component_count = 0;

		for (Configuration& cfg_orig : training_set)
		{
			cfg = cfg_orig;
			mtpr.CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
			if (cfg_orig.has_energy() && cfg.has_energy())
			{
				const double dE = cfg_orig.energy - cfg.energy;
				train_energy_abs_sum += std::abs(dE);
				train_energy_sq_weighted_sum += dE * dE / cfg.size();
			}
			if (cfg_orig.has_forces() && cfg.has_forces()) {
				for (int i = 0; i < cfg.size(); i++)
					for (int a = 0; a < 3; a++)
					{
						const double dF = cfg.force(i)[a] - cfg_orig.force(i)[a];
						train_force_abs_component_sum += std::abs(dF);
						train_force_sq_component_sum += dF * dF;
					}
				train_force_component_count += static_cast<long long>(3) * cfg.size();
			}
			if (cfg_orig.has_stresses() && cfg.has_stresses()) {
				const double d0 = cfg.stresses[0][0] - cfg_orig.stresses[0][0];
				const double d1 = cfg.stresses[1][1] - cfg_orig.stresses[1][1];
				const double d2 = cfg.stresses[2][2] - cfg_orig.stresses[2][2];
				const double d3 = cfg.stresses[1][2] - cfg_orig.stresses[1][2];
				const double d4 = cfg.stresses[0][2] - cfg_orig.stresses[0][2];
				const double d5 = cfg.stresses[0][1] - cfg_orig.stresses[0][1];
				train_stress_abs_component_sum += std::abs(d0) + std::abs(d1) + std::abs(d2) +
				                                  std::abs(d3) + std::abs(d4) + std::abs(d5);
				train_stress_sq_component_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
				train_stress_component_count += 6;
			}

		}
		bufferrmon.reset();
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double train_energy_abs_sum_global = 0.0;
		double train_energy_sq_weighted_sum_global = 0.0;
		double train_force_abs_component_sum_global = 0.0;
		double train_force_sq_component_sum_global = 0.0;
		double train_stress_abs_component_sum_global = 0.0;
		double train_stress_sq_component_sum_global = 0.0;
		long long train_force_component_count_global = 0;
		long long train_stress_component_count_global = 0;
		MPI_Reduce(&train_energy_abs_sum, &train_energy_abs_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_energy_sq_weighted_sum, &train_energy_sq_weighted_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_abs_component_sum, &train_force_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_sq_component_sum, &train_force_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_abs_component_sum, &train_stress_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_sq_component_sum, &train_stress_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_component_count, &train_force_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_component_count, &train_stress_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			if (prank == 0)
			{
				const double train_energy_mae_mev_atom =
					train_atom_total > 0 ? 1000.0 * train_energy_abs_sum_global / static_cast<double>(train_atom_total) : 0.0;
				const double train_energy_rmse_mev_atom =
					train_atom_total > 0 ? 1000.0 * std::sqrt(train_energy_sq_weighted_sum_global / static_cast<double>(train_atom_total)) : 0.0;
				const double train_force_mae_mev_a =
					train_force_component_count_global > 0 ? 1000.0 * train_force_abs_component_sum_global / static_cast<double>(train_force_component_count_global) : 0.0;
				const double train_force_rmse_mev_a =
					train_force_component_count_global > 0 ? 1000.0 * std::sqrt(train_force_sq_component_sum_global / static_cast<double>(train_force_component_count_global)) : 0.0;
				const double train_stress_mae_ev =
					train_stress_component_count_global > 0 ? train_stress_abs_component_sum_global / static_cast<double>(train_stress_component_count_global) : 0.0;
				const double train_stress_rmse_ev =
					train_stress_component_count_global > 0 ? std::sqrt(train_stress_sq_component_sum_global / static_cast<double>(train_stress_component_count_global)) : 0.0;
				std::cout << "\n=== Train Summary ===\n"
				          << std::fixed << std::setprecision(3)
				          << "Structures           : " << train_cfg_total << "\n"
				          << "Atoms                : " << train_atom_total << "\n"
				          << "Energy MAE (meV/atom): " << train_energy_mae_mev_atom << "\n"
				          << "Energy RMSE(meV/atom): " << train_energy_rmse_mev_atom << "\n"
				          << "Force MAE  (meV/A)   : " << train_force_mae_mev_a << "\n"
				          << "Force RMSE (meV/A)   : " << train_force_rmse_mev_a << "\n"
				          << "Stress MAE (eV)      : " << train_stress_mae_ev << "\n"
				          << "Stress RMSE(eV)      : " << train_stress_rmse_ev << "\n"
				          << "=====================\n";
			}

#else
		const double train_energy_mae_mev_atom =
			train_atom_total > 0 ? 1000.0 * train_energy_abs_sum / static_cast<double>(train_atom_total) : 0.0;
		const double train_energy_rmse_mev_atom =
			train_atom_total > 0 ? 1000.0 * std::sqrt(train_energy_sq_weighted_sum / static_cast<double>(train_atom_total)) : 0.0;
		const double train_force_mae_mev_a =
			train_force_component_count > 0 ? 1000.0 * train_force_abs_component_sum / static_cast<double>(train_force_component_count) : 0.0;
		const double train_force_rmse_mev_a =
			train_force_component_count > 0 ? 1000.0 * std::sqrt(train_force_sq_component_sum / static_cast<double>(train_force_component_count)) : 0.0;
		const double train_stress_mae_ev =
			train_stress_component_count > 0 ? train_stress_abs_component_sum / static_cast<double>(train_stress_component_count) : 0.0;
		const double train_stress_rmse_ev =
			train_stress_component_count > 0 ? std::sqrt(train_stress_sq_component_sum / static_cast<double>(train_stress_component_count)) : 0.0;
		std::cout << "\n=== Train Summary ===\n"
		          << std::fixed << std::setprecision(3)
		          << "Structures           : " << train_cfg_total << "\n"
		          << "Atoms                : " << train_atom_total << "\n"
		          << "Energy MAE (meV/atom): " << train_energy_mae_mev_atom << "\n"
		          << "Energy RMSE(meV/atom): " << train_energy_rmse_mev_atom << "\n"
		          << "Force MAE  (meV/A)   : " << train_force_mae_mev_a << "\n"
		          << "Force RMSE (meV/A)   : " << train_force_rmse_mev_a << "\n"
		          << "Stress MAE (eV)      : " << train_stress_mae_ev << "\n"
		          << "Stress RMSE(eV)      : " << train_stress_rmse_ev << "\n"
		          << "=====================\n";
#endif
	}

	errmon.reset();
	bufferrmon.reset();
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (validfnm != "")
	{
		double valid_energy_abs_sum = 0.0;
		double valid_energy_sq_weighted_sum = 0.0;
		double valid_force_abs_component_sum = 0.0;
		double valid_force_sq_component_sum = 0.0;
		double valid_stress_abs_component_sum = 0.0;
		double valid_stress_sq_component_sum = 0.0;
		long long valid_force_component_count = 0;
		long long valid_stress_component_count = 0;
		for (Configuration& cfg_orig : validSet)
		{
			cfg = cfg_orig;
			mtpr.CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
			if (cfg_orig.has_energy() && cfg.has_energy())
			{
				const double dE = cfg_orig.energy - cfg.energy;
				valid_energy_abs_sum += std::abs(dE);
				valid_energy_sq_weighted_sum += dE * dE / cfg.size();
			}
			if (cfg_orig.has_forces() && cfg.has_forces()) {
				for (int i = 0; i < cfg.size(); i++)
					for (int a = 0; a < 3; a++)
					{
						const double dF = cfg.force(i)[a] - cfg_orig.force(i)[a];
						valid_force_abs_component_sum += std::abs(dF);
						valid_force_sq_component_sum += dF * dF;
					}
				valid_force_component_count += static_cast<long long>(3) * cfg.size();
			}
			if (cfg_orig.has_stresses() && cfg.has_stresses()) {
				const double d0 = cfg.stresses[0][0] - cfg_orig.stresses[0][0];
				const double d1 = cfg.stresses[1][1] - cfg_orig.stresses[1][1];
				const double d2 = cfg.stresses[2][2] - cfg_orig.stresses[2][2];
				const double d3 = cfg.stresses[1][2] - cfg_orig.stresses[1][2];
				const double d4 = cfg.stresses[0][2] - cfg_orig.stresses[0][2];
				const double d5 = cfg.stresses[0][1] - cfg_orig.stresses[0][1];
				valid_stress_abs_component_sum += std::abs(d0) + std::abs(d1) + std::abs(d2) +
				                                  std::abs(d3) + std::abs(d4) + std::abs(d5);
				valid_stress_sq_component_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
				valid_stress_component_count += 6;
			}
		}
		bufferrmon.reset();
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double valid_energy_abs_sum_global = 0.0;
		double valid_force_abs_component_sum_global = 0.0;
		double valid_stress_abs_component_sum_global = 0.0;
		long long valid_force_component_count_global = 0;
		long long valid_stress_component_count_global = 0;
		MPI_Reduce(&valid_energy_abs_sum, &valid_energy_abs_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double valid_energy_sq_weighted_sum_global = 0.0;
		double valid_force_sq_component_sum_global = 0.0;
		double valid_stress_sq_component_sum_global = 0.0;
		MPI_Reduce(&valid_force_abs_component_sum, &valid_force_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_abs_component_sum, &valid_stress_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_energy_sq_weighted_sum, &valid_energy_sq_weighted_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_force_sq_component_sum, &valid_force_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_sq_component_sum, &valid_stress_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_force_component_count, &valid_force_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_component_count, &valid_stress_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if (prank == 0)
		{
			const double valid_energy_mae_mev_atom =
				valid_atom_total > 0 ? 1000.0 * valid_energy_abs_sum_global / static_cast<double>(valid_atom_total) : 0.0;
			const double valid_energy_rmse_mev_atom =
				valid_atom_total > 0 ? 1000.0 * std::sqrt(valid_energy_sq_weighted_sum_global / static_cast<double>(valid_atom_total)) : 0.0;
			const double valid_force_mae_mev_a =
				valid_force_component_count_global > 0 ? 1000.0 * valid_force_abs_component_sum_global / static_cast<double>(valid_force_component_count_global) : 0.0;
			const double valid_force_rmse_mev_a =
				valid_force_component_count_global > 0 ? 1000.0 * std::sqrt(valid_force_sq_component_sum_global / static_cast<double>(valid_force_component_count_global)) : 0.0;
			const double valid_stress_mae_ev =
				valid_stress_component_count_global > 0 ? valid_stress_abs_component_sum_global / static_cast<double>(valid_stress_component_count_global) : 0.0;
			const double valid_stress_rmse_ev =
				valid_stress_component_count_global > 0 ? std::sqrt(valid_stress_sq_component_sum_global / static_cast<double>(valid_stress_component_count_global)) : 0.0;
			std::cout << "\n=== Validation Summary ===\n"
			          << std::fixed << std::setprecision(3)
			          << "Structures           : " << valid_cfg_total << "\n"
			          << "Atoms                : " << valid_atom_total << "\n"
			          << "Energy MAE (meV/atom): " << valid_energy_mae_mev_atom << "\n"
			          << "Energy RMSE(meV/atom): " << valid_energy_rmse_mev_atom << "\n"
			          << "Force MAE  (meV/A)   : " << valid_force_mae_mev_a << "\n"
			          << "Force RMSE (meV/A)   : " << valid_force_rmse_mev_a << "\n"
			          << "Stress MAE (eV)      : " << valid_stress_mae_ev << "\n"
			          << "Stress RMSE(eV)      : " << valid_stress_rmse_ev << "\n"
			          << "==========================\n";
		}
#else
		const double valid_energy_mae_mev_atom =
			valid_atom_total > 0 ? 1000.0 * valid_energy_abs_sum / static_cast<double>(valid_atom_total) : 0.0;
		const double valid_energy_rmse_mev_atom =
			valid_atom_total > 0 ? 1000.0 * std::sqrt(valid_energy_sq_weighted_sum / static_cast<double>(valid_atom_total)) : 0.0;
		const double valid_force_mae_mev_a =
			valid_force_component_count > 0 ? 1000.0 * valid_force_abs_component_sum / static_cast<double>(valid_force_component_count) : 0.0;
		const double valid_force_rmse_mev_a =
			valid_force_component_count > 0 ? 1000.0 * std::sqrt(valid_force_sq_component_sum / static_cast<double>(valid_force_component_count)) : 0.0;
		const double valid_stress_mae_ev =
			valid_stress_component_count > 0 ? valid_stress_abs_component_sum / static_cast<double>(valid_stress_component_count) : 0.0;
		const double valid_stress_rmse_ev =
			valid_stress_component_count > 0 ? std::sqrt(valid_stress_sq_component_sum / static_cast<double>(valid_stress_component_count)) : 0.0;
		std::cout << "\n=== Validation Summary ===\n"
		          << std::fixed << std::setprecision(3)
		          << "Structures           : " << valid_cfg_total << "\n"
		          << "Atoms                : " << valid_atom_total << "\n"
		          << "Energy MAE (meV/atom): " << valid_energy_mae_mev_atom << "\n"
		          << "Energy RMSE(meV/atom): " << valid_energy_rmse_mev_atom << "\n"
		          << "Force MAE  (meV/A)   : " << valid_force_mae_mev_a << "\n"
		          << "Force RMSE (meV/A)   : " << valid_force_rmse_mev_a << "\n"
		          << "Stress MAE (eV)      : " << valid_stress_mae_ev << "\n"
		          << "Stress RMSE(eV)      : " << valid_stress_rmse_ev << "\n"
		          << "==========================\n";
#endif
	}
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Finalize();
#endif
}

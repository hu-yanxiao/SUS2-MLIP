/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../src/mlp/mlp.h"

#include "../../src/common/stdafx.h"
#include "../../src/mlip_wrapper.h"
#include "../../dev_src/mlp/dev_self_test.h"
#include "../../src/mlp/self_test.h"
#include "../../src/mlp/mlp.h"
#include "../../src/mlp/train.h"
#include "../../src/mlp/calc_errors.h"
#include "../mtpr.h" 
#include "../mtpr_trainer.h" 
#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace std;

namespace {

std::vector<int> ParseSpeciesIndexList(const std::string& value)
{
	std::vector<int> indices;
	std::stringstream stream(value);
	std::string token;
	while (std::getline(stream, token, ',')) {
		token.erase(std::remove_if(token.begin(), token.end(),
			[](unsigned char ch) { return std::isspace(ch) != 0; }),
			token.end());
		if (token.empty())
			ERROR("--species should be a comma-separated list of integer species indices.");
		std::size_t parsed_chars = 0;
		int index = 0;
		try {
			index = std::stoi(token, &parsed_chars);
		} catch (const std::exception&) {
			ERROR("--species should be a comma-separated list of integer species indices.");
		}
		if (parsed_chars != token.size())
			ERROR("--species should be a comma-separated list of integer species indices.");
		indices.push_back(index);
	}
	if (indices.empty())
		ERROR("--species should contain at least one species index.");
	return indices;
}

std::string FormatSpeciesMapping(const std::vector<int>& old_species_indices)
{
	std::ostringstream oss;
	for (int i = 0; i < static_cast<int>(old_species_indices.size()); ++i) {
		if (i != 0)
			oss << ", ";
		oss << old_species_indices[i] << "->" << i;
	}
	return oss.str();
}

std::string MtprCoeffGroup(const MLMTPR& mtpr, int coeff_index)
{
	if (coeff_index < mtpr.species_count)
		return "shift_coeffs";

	const int scal_begin = mtpr.species_count;
	const int radial_begin = mtpr.RadialCoeffOffset();
	if (coeff_index >= scal_begin && coeff_index < radial_begin)
		return "scal_coeffs";

	const int radial_end = radial_begin + mtpr.radial_func_count * mtpr.RadialCoeffBlockSize();
	if (coeff_index >= radial_begin && coeff_index < radial_end)
		return "radial_coeffs";

	if (mtpr.HasEnvGate()) {
		const int env_begin = mtpr.EnvGateCoeffOffset();
		const int env_end = env_begin + mtpr.EnvGateCoeffCount();
		if (coeff_index >= env_begin && coeff_index < env_begin + mtpr.species_count)
			return "env_gate_lambda_raw";
		if (coeff_index >= env_begin && coeff_index < env_end)
			return "env_gate_log_density";
	}

	return "linear_coeffs";
}

class MTPRGradientCheckTrainer : public MTPR_trainer
{
public:
	using MTPR_trainer::MTPR_trainer;

	bool CheckActiveGradients(MLMTPR& mtpr,
	                          std::vector<Configuration>& training_set,
	                          const std::vector<int>& active_coeff_indices,
	                          double relative_step,
	                          double rel_tolerance,
	                          double abs_tolerance,
	                          int max_report)
	{
		struct Result {
			int coeff_index = -1;
			std::string group;
			double value = 0.0;
			double analytic = 0.0;
			double finite_diff = 0.0;
			double abs_error = 0.0;
			double rel_error = 0.0;
			bool passed = false;
		};
		struct GroupStats {
			int count = 0;
			int fail_count = 0;
			double max_abs_error = 0.0;
			double max_rel_error = 0.0;
		};

		if (training_set.empty())
			ERROR("check-mtpr-grad requires at least one configuration.");
		if (relative_step <= 0.0)
			ERROR("check-mtpr-grad requires a positive --step value.");

		SetCollectErrorMetrics(false);
		CalcObjectiveFunctionGrad(training_set);
		const double loss0 = loss_;
		std::vector<double> analytic_grad = loss_grad_;
		std::vector<Result> failures;
		std::map<std::string, GroupStats> group_stats;

		Result max_abs_result;
		Result max_rel_result;
		bool have_result = false;

		for (int coeff_index : active_coeff_indices) {
			const double original = mtpr.Coeff()[coeff_index];
			const double delta = relative_step * std::max(1.0, std::abs(original));
			mtpr.Coeff()[coeff_index] = original + delta;
			const double loss_plus = ObjectiveFunction(training_set);
			mtpr.Coeff()[coeff_index] = original - delta;
			const double loss_minus = ObjectiveFunction(training_set);
			mtpr.Coeff()[coeff_index] = original;

			Result result;
			result.coeff_index = coeff_index;
			result.group = MtprCoeffGroup(mtpr, coeff_index);
			result.value = original;
			result.analytic = analytic_grad[coeff_index];
			result.finite_diff = (loss_plus - loss_minus) / (2.0 * delta);
			result.abs_error = std::abs(result.analytic - result.finite_diff);
			const double denom = std::max(abs_tolerance,
				std::max(std::abs(result.analytic), std::abs(result.finite_diff)));
			result.rel_error = result.abs_error / denom;
			result.passed = std::isfinite(result.analytic)
			             && std::isfinite(result.finite_diff)
			             && std::isfinite(result.abs_error)
			             && std::isfinite(result.rel_error)
			             && (result.abs_error <= abs_tolerance || result.rel_error <= rel_tolerance);

			GroupStats& stats = group_stats[result.group];
			stats.count++;
			if (!result.passed) {
				stats.fail_count++;
				failures.push_back(result);
			}
			stats.max_abs_error = std::max(stats.max_abs_error, result.abs_error);
			stats.max_rel_error = std::max(stats.max_rel_error, result.rel_error);

			if (!have_result || result.abs_error > max_abs_result.abs_error)
				max_abs_result = result;
			if (!have_result || result.rel_error > max_rel_result.rel_error)
				max_rel_result = result;
			have_result = true;
		}

		std::sort(failures.begin(), failures.end(),
			[](const Result& a, const Result& b) {
				if (a.rel_error != b.rel_error)
					return a.rel_error > b.rel_error;
				return a.abs_error > b.abs_error;
			});

		std::cout << std::scientific << std::setprecision(8);
		std::cout << "check-mtpr-grad loss0 " << loss0
		          << " configs " << training_set.size()
		          << " active_coeffs " << active_coeff_indices.size()
		          << " step " << relative_step
		          << " rel_tol " << rel_tolerance
		          << " abs_tol " << abs_tolerance
		          << std::endl;
		for (const auto& item : group_stats) {
			std::cout << "group " << item.first
			          << " count " << item.second.count
			          << " failures " << item.second.fail_count
			          << " max_abs_err " << item.second.max_abs_error
			          << " max_rel_err " << item.second.max_rel_error
			          << std::endl;
		}
		if (have_result) {
			std::cout << "max_abs coeff " << max_abs_result.coeff_index
			          << " group " << max_abs_result.group
			          << " value " << max_abs_result.value
			          << " analytic " << max_abs_result.analytic
			          << " finite_diff " << max_abs_result.finite_diff
			          << " abs_err " << max_abs_result.abs_error
			          << " rel_err " << max_abs_result.rel_error
			          << std::endl;
			std::cout << "max_rel coeff " << max_rel_result.coeff_index
			          << " group " << max_rel_result.group
			          << " value " << max_rel_result.value
			          << " analytic " << max_rel_result.analytic
			          << " finite_diff " << max_rel_result.finite_diff
			          << " abs_err " << max_rel_result.abs_error
			          << " rel_err " << max_rel_result.rel_error
			          << std::endl;
		}

		if (!failures.empty()) {
			const int report_count = std::min(max_report, static_cast<int>(failures.size()));
			std::cout << "failed_coefficients " << failures.size() << std::endl;
			for (int i = 0; i < report_count; ++i) {
				const Result& result = failures[i];
				std::cout << "fail coeff " << result.coeff_index
				          << " group " << result.group
				          << " value " << result.value
				          << " analytic " << result.analytic
				          << " finite_diff " << result.finite_diff
				          << " abs_err " << result.abs_error
				          << " rel_err " << result.rel_error
				          << std::endl;
			}
		}

		return failures.empty();
	}
};

}

// does a number of developer unit tests
// returns true if all tests are successful
// otherwise returns false and stops further tests
bool self_test_dev()
{
	ofstream logstream("temp/log");
	SetStreamForOutput(&logstream);

	int mpi_rank = 0;
	int mpi_size = 1;

#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif // MLIP_MPI

#ifndef MLIP_DEBUG
	if (mpi_rank == 0) {
		std::cout << "Note: self-test is running without #define MLIP_DEBUG;\n"
			<< "      build with -DMLIP_DEBUG and run if troubles encountered" << std::endl;
	}
#endif

	if (mpi_size == 1) {
		if (mpi_rank == 0) cout << "Serial pub tests:" << endl;
		if (!RunAllTests(false)) return false;
		if (mpi_rank == 0) cout << "Serial dev tests:" << endl;
		if (!RunAllTestsDev(false)) return false;
	}
#ifdef MLIP_MPI
	if (mpi_rank == 0) cout << "MPI pub tests (" << mpi_size << " cores):" << endl;
	if (!RunAllTests(true)) return false;
	if (mpi_rank == 0) cout << "MPI dev tests (" << mpi_size << " cores):" << endl;
	if (!RunAllTestsDev(true)) return false;
#endif // MLIP_MPI

	logstream.close();
	return true;
}

bool DevCommands(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	bool is_command_found = false;
	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	BEGIN_COMMAND("self-test-dev",
		"performs a number of unit tests",
		"mlp-sus2 self-test-dev\n"
	) {
		if(!self_test_dev()) exit(1);
	} END_COMMAND;

	BEGIN_COMMAND("prune-model",
		"writes a species-pruned SUS2 model",
		"mlp-sus2 prune-model input.mtp output.mtp --species=2,4,6\n"
		"  Keeps the selected old species, preserves shared parameters, and remaps\n"
		"  them in the output model as 0,1,2,... in the order provided.\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 prune-model: 2 arguments are required\n";
			return 1;
		}
		if (opts["species"] == "") {
			std::cout << "mlp-sus2 prune-model: --species=<comma-separated indices> is required\n";
			return 1;
		}

		const std::vector<int> old_species_indices = ParseSpeciesIndexList(opts["species"]);
		if (mpi_rank == 0) {
			MLMTPR mtpr(args[0]);
			if (!mtpr.HasCompleteParameters())
				ERROR("prune-model requires a complete trained model with shift/scal/radial/linear coefficients.");
			const int old_species_count = mtpr.species_count;
			mtpr.PruneSpecies(old_species_indices);
			mtpr.Save(args[1]);
			std::cout << "Pruned model written to " << args[1]
			          << " species_count " << old_species_count
			          << " -> " << mtpr.species_count
			          << " mapping: " << FormatSpeciesMapping(old_species_indices)
			          << std::endl;
		}
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	} END_COMMAND;

	BEGIN_COMMAND("check-mtpr-grad",
		"finite-difference checks active SUS2 MTPR training gradients",
		"mlp-sus2 check-mtpr-grad pot.mtp train.cfg [options]\n"
		"  Checks every active coefficient used by BFGS, excluding redundant stored\n"
		"  radial species rows and optionally excluding scal_coeffs with --fine-tune.\n"
		"  Options:\n"
		"  --energy-weight=<double>: default=1\n"
		"  --force-weight=<double>: default=0.01\n"
		"  --stress-weight=<double>: default=0.001\n"
		"  --std-weight=<double>: default=0.2\n"
		"  --stdd-weight=<double>: default=0.00001\n"
		"  --scale-by-force=<int>: default=0\n"
		"  --weighting=<string>: default=vibrations\n"
		"  --step=<double>: relative finite-difference step, default=1e-6\n"
		"  --rel-tol=<double>: default=1e-3\n"
		"  --abs-tol=<double>: default=1e-6\n"
		"  --max-configs=<int>: default=1\n"
		"  --max-report=<int>: default=30\n"
		"  --fine-tune: check the same active space as fine-tune BFGS\n"
	) {
		if (args.size() != 2) {
			std::cout << "mlp-sus2 check-mtpr-grad: 2 arguments are required\n";
			return 1;
		}
		if (mpi_rank == 0) {
			double weight_energy = 1.0;
			if (opts["energy-weight"] != "")
				weight_energy = std::stod(opts["energy-weight"]);
			double weight_force = 0.01;
			if (opts["force-weight"] != "")
				weight_force = std::stod(opts["force-weight"]);
			double weight_stress = 0.001;
			if (opts["stress-weight"] != "")
				weight_stress = std::stod(opts["stress-weight"]);
			double std_weight = 0.2;
			if (opts["std-weight"] != "")
				std_weight = std::stod(opts["std-weight"]);
			double stdd_weight = 0.00001;
			if (opts["stdd-weight"] != "")
				stdd_weight = std::stod(opts["stdd-weight"]);
			int scale_by_force = 0;
			if (opts["scale-by-force"] != "")
				scale_by_force = std::stoi(opts["scale-by-force"]);
			double relative_step = 1e-6;
			if (opts["step"] != "")
				relative_step = std::stod(opts["step"]);
			double rel_tolerance = 1e-3;
			if (opts["rel-tol"] != "")
				rel_tolerance = std::stod(opts["rel-tol"]);
			double abs_tolerance = 1e-6;
			if (opts["abs-tol"] != "")
				abs_tolerance = std::stod(opts["abs-tol"]);
			int max_configs = 1;
			if (opts["max-configs"] != "")
				max_configs = std::stoi(opts["max-configs"]);
			int max_report = 30;
			if (opts["max-report"] != "")
				max_report = std::stoi(opts["max-report"]);
			if (max_configs <= 0)
				ERROR("--max-configs must be positive.");
			if (max_report < 0)
				ERROR("--max-report must be non-negative.");

			MLMTPR mtpr(args[0]);
			std::vector<Configuration> training_set;
			std::ifstream ifs(args[1], std::ios::binary);
			Configuration cfg;
			while (static_cast<int>(training_set.size()) < max_configs && cfg.Load(ifs))
				training_set.push_back(cfg);
			if (training_set.empty())
				ERROR("No configurations were loaded for check-mtpr-grad.");

			MTPRGradientCheckTrainer trainer(&mtpr,
				weight_energy,
				weight_force,
				weight_stress,
				0.0,
				1.0e-6,
				"",
				scale_by_force,
				0);
			trainer.std_scaling = std_weight;
			trainer.stdd_scaling = stdd_weight;
			if (opts["weighting"] != "")
				trainer.weighting = opts["weighting"];

			std::vector<int> active_coeff_indices;
			const bool fine_tune = opts["fine-tune"] != "";
			mtpr.BuildActiveCoeffIndices(active_coeff_indices, fine_tune);
			if (!trainer.CheckActiveGradients(mtpr,
			                                  training_set,
			                                  active_coeff_indices,
			                                  relative_step,
			                                  rel_tolerance,
			                                  abs_tolerance,
			                                  max_report))
				exit(2);
		}
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	} END_COMMAND;

	BEGIN_COMMAND("select-add",
		"actively selects configurations to be added to the current training set",
		"mlp-sus2 select-add pot.mtp train.cfg new.cfg diff.cfg:\n"
		"actively selects configurations from new.cfg and save those\n"
		"that need to be added to train.cfg to diff.cfg\n"
		"  Options:\n"
		"  --init-threshold=<num>: set the initial threshold to num, default=1e-5\n"
		"  --select-threshold=<num>: set the select threshold to num, default=1.1\n"
		"  --swap-threshold=<num>: set the swap threshold to num, default=1.0000001\n"
//		"  --energy-weight=<num>: set the weight for energy equation, default=1\n"
//		"  --force-weight=<num>: set the weight for force equations, default=0\n"
//		"  --stress-weight=<num>: set the weight for stress equations, default=0\n"
//		"  --nbh-weight=<num>: set the weight for site energy equations, default=0\n"
		"  --als-filename=<filename>: active learning state (ALS) filename\n"
		"  --selected-filename=<filename>: file with selected configurations\n"
		"  --selection-limit=<num>: swap limit for multiple selection, default=0 (disabled)\n"
		"  --weighting=<string>: way of weighting configurations with different number of atoms,\n"
                "                        default=vibrations, other=molecules, structures.\n"
		) {

		if (args.size() != 4) {
			std::cout << "\tError: 4 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string train_filename = args[1];
		const string new_cfg_filename = args[2];
		const string diff_filename = args[3];

		int selection_limit=0;						//limits the number of swaps in MaxVol

		cout << "Potential from " << mtp_filename
			<< ", traning set: " << train_filename
			<< ", add from set: " << new_cfg_filename
			<< endl;
		MLMTPR mtpr(mtp_filename);

		double init_threshold = 1e-5;
		if (opts["init-threshold"] != "")
			init_threshold = std::stod(opts["init-threshold"]);
		double select_threshold = 1.1;
		if (opts["select-threshold"] != "")
			select_threshold = std::stod(opts["select-threshold"]);
		double swap_threshold = 1.0000001;
		if (opts["swap-threshold"] != "")
			swap_threshold = std::stod(opts["swap-threshold"]);
		double nbh_cmpnts_weight = 0;
		if (opts["nbh-weight"] != "")
			nbh_cmpnts_weight = std::stod(opts["nbh-weight"]);
		double ene_cmpnts_weight = 1;
		if (opts["energy-weight"] != "")
			ene_cmpnts_weight = std::stod(opts["energy-weight"]);
		double frc_cmpnts_weight = 0;
		if (opts["force-weight"] != "")
			frc_cmpnts_weight = std::stod(opts["force-weight"]);
		double str_cmpnts_weight = 0;
		if (opts["stress-weight"] != "")
			str_cmpnts_weight = std::stod(opts["stress-weight"]);
		string als_filename = "state.als";
		if (opts["mvs-filename"] != "")
			als_filename = opts["mvs-filename"];
		if (opts["als-filename"] != "")
			als_filename = opts["als-filename"];
		string selected_filename = "selected.cfg";
		if (opts["selected-filename"] != "")
			selected_filename = opts["selected-filename"];
		if (opts["selection-limit"] != "")
			selection_limit = std::stoi(opts["selection-limit"]);
	
		MaxvolSelection selector(&mtpr, init_threshold, swap_threshold, swap_threshold,
						nbh_cmpnts_weight, ene_cmpnts_weight, frc_cmpnts_weight, str_cmpnts_weight);

		if (opts["weighting"] != "")
			selector.weighting = opts["weighting"];

		int count = 0;

		{
			cout << "loading training set... " << std::flush;

			ifstream ifs(train_filename, std::ios::binary);
			Configuration cfg;
			while (cfg.Load(ifs)) {
				cfg.features["ID"] = to_string(-1);
				selector.AddForSelection(cfg);
				count ++;
			}

			cout << "done" << endl;

		}
		cout << count << " configurations found in the training set\n" << std::flush;
		selector.Select();
		for (Configuration& x : selector.selected_cfgs)
			x.features["ID"] = "-1";

		selector.threshold_select = select_threshold;
	
		count = 1;

		ifstream ifs(new_cfg_filename, std::ios::binary);
		vector<Configuration> new_cfg_set;
		for (Configuration cfg; cfg.Load(ifs); count++) {
			cfg.features["ID"] = to_string(count);
			new_cfg_set.push_back(cfg);
			selector.AddForSelection(cfg);
		}

		if (selection_limit==0)
			selector.Select();
		else
			{
			selector.Select(selection_limit);
			cout << "Swap limit = " << selection_limit << endl;
			}

		{
			int count = 0;
			for (Configuration& cfg : selector.selected_cfgs)
				if (cfg.size() > 0) count++;

			cout << count << " configurations selected from both sets\n" << std::flush;
		}
		cout << "new configuration count = " << new_cfg_set.size() << endl;

		

		ofstream ofs(diff_filename, ios::binary);
		vector<int> valid_to_train;
		std::set<int> unique_cfg;
		count = 0;
		for (Configuration& x : selector.selected_cfgs) {
			if (stoi(x.features["ID"]) > 0 && unique_cfg.find(x.id()) == unique_cfg.end()) {
				valid_to_train.push_back(stoi(x.features["ID"]));
				x.features.erase("ID");
				x.Save(ofs);
				unique_cfg.insert(x.id());
				count++;
			}
		}
		ofs.close();

		cout << "TS increased by " << count << " configs" << endl;
		
		//further selection till selection limit
		int delta = selection_limit - count;
		count = 0;
		vector<double> grades;
		vector<int> inds;
		if (delta>99990) //disabled further selection
		{
			for (int j=0; j< new_cfg_set.size();j++)
			{
				if (unique_cfg.find(new_cfg_set[j].id()) == unique_cfg.end())
				{
					double gr = selector.Grade(new_cfg_set[j]);

					if (count == 0)
					{
						grades.push_back(gr);
						inds.push_back(j);
					}
					for (int i = 0; i < min(delta, count); i++)
					{

						if (grades[i] <= gr) 
						{
							grades.insert(grades.begin() + i, gr);
							inds.insert(inds.begin() + i, j);
							break;
						}
						else if (i == count - 1)
						{
							grades.push_back(gr);
							inds.push_back(j);
							break;
						}
					}
					count++;
				}
			}
		cout << "TS increased by " << delta << " configs" << endl;
		}
		
		selector.Save(als_filename);
		selector.SaveSelected(selected_filename);
		ofs.open(diff_filename,ios::app);
		for (int i = 0; i < delta; i++)
			new_cfg_set[inds[i]].Save(ofs);

		ofs.close();
	} END_COMMAND;

	BEGIN_COMMAND("calc-grade",
		"calculates and saves maxvol grades of input configurations",
		"mlp-sus2 calc-grade pot.mtp train.cfg in.cfg out.cfg:\n"
		"actively selects from train.cfg, generates the ALS file from train.cfg, and\n"
		"calculates maxvol grades of configurations located in in.cfg\n"
		"and writes them to out.cfg\n"
		"  Options:\n"
		"  --init-threshold=<num>: set the initial threshold to 1+num, default=1e-5\n"
		"  --select-threshold=<num>: set the select threshold to num, default=1.1\n"
		"  --swap-threshold=<num>: set the swap threshold to num, default=1.0000001\n"
//		"  --energy-weight=<num>: set the weight for energy equation, default=1\n"
//		"  --force-weight=<num>: set the weight for force equations, default=0\n"
//		"  --stress-weight=<num>: set the weight for stress equations, default=0\n"
//		"  --nbh-weight=<num>: set the weight for site energy equations, default=0\n"
		"  --als-filename=<filename>: active learning state (ALS) filename\n"
		) {

		if (args.size() != 4) {
			std::cout << "\tError: 4 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string train_filename = args[1];
		const string input_filename = args[2];
		const string output_filename = args[3];

		cout << "Potential from " << mtp_filename
			<< ", train: " << train_filename
			<< ", input: " << input_filename
			<< endl;
		MLMTPR mtpr(mtp_filename);

		double init_threshold = 1e-5;
		if (opts["init-threshold"] != "")
			init_threshold = std::stod(opts["init-threshold"]);
		double select_threshold = 1.1;
		if (opts["select-threshold"] != "")
			select_threshold = std::stod(opts["select-threshold"]);
		double swap_threshold = 1.0000001;
		if (opts["swap-threshold"] != "")
			swap_threshold = std::stod(opts["swap-threshold"]);
		double nbh_cmpnts_weight = 0;
		if (opts["nbh-weight"] != "")
			nbh_cmpnts_weight = std::stod(opts["nbh-weight"]);
		double ene_cmpnts_weight = 1;
		if (opts["energy-weight"] != "")
			ene_cmpnts_weight = std::stod(opts["energy-weight"]);
		double frc_cmpnts_weight = 0;
		if (opts["force-weight"] != "")
			frc_cmpnts_weight = std::stod(opts["force-weight"]);
		double str_cmpnts_weight = 0;
		if (opts["stress-weight"] != "")
			str_cmpnts_weight = std::stod(opts["stress-weight"]);
		string als_filename = "state.als";
		if (opts["mvs-filename"] != "")
			als_filename = opts["mvs-filename"];		
		if (opts["als-filename"] != "")
			als_filename = opts["als-filename"];		

		MaxvolSelection selector(&mtpr, init_threshold, select_threshold, swap_threshold, 
						nbh_cmpnts_weight, ene_cmpnts_weight, frc_cmpnts_weight, str_cmpnts_weight);

		ifstream ifs_train(train_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs_train)) {
			selector.AddForSelection(cfg);
		}
		selector.Select();

		ifs_train.close();
		selector.Save(als_filename);

		ifstream ifs_input(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		while (cfg.Load(ifs_input)) {
			selector.Grade(cfg);
			cfg.Save(ofs);	
		}
		ifs_input.close();
		ofs.close();

	} END_COMMAND;


	BEGIN_COMMAND("calc-descriptors",
		"calculates descriptors in each neighborhood of configurations",
		"mlp-sus2 calc-descriptors pot.mtp in.cfg out.cfg:\n"
		"calculates descriptors in each neighborhood of configurations from in.cfg and\n"
		"writes configurations with calculated descriptors to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);	

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcDescriptors(cfg, ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_COMMAND("calc-partialE",
		"calculates body-order E of each atom",
		"mlp-sus2 calc-partialE pot.mtp in.cfg out.cfg:\n"
		"calculates body-order E of atoms from in.cfg and\n"
		"writes results to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);	

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcpartialE(cfg, ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_COMMAND("calc-efs",
		"calculates energies, forces, and stresses (efs) of configurations",
		"mlp-sus2 calc-efs pot.mtp in.cfg out.cfg:\n"
		"calculates energies, forces, and stresses (efs) of configurations from in.cfg and\n"
		"writes configurations with calculated efs to out.cfg\n"
		) {

		if (args.size() != 3) {
			std::cout << "\tError: 3 arguments required\n";
			return 1;
		}

		const string mtp_filename = args[0];
		const string input_filename = args[1];
		const string output_filename = args[2];

		MLMTPR mtpr(mtp_filename);	

		ifstream ifs(input_filename, std::ios::binary);
		ofstream ofs(output_filename, std::ios::binary);
		Configuration cfg;
		while (cfg.Load(ifs)) {
			mtpr.CalcEFS(cfg);
			cfg.Save(ofs);
		}

		ifs.close();
		ofs.close();

	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("invert-stress",
		"changes sign of stress in a configuration database",
		"mlp-sus2 invert-stress db.cfg:\n"
		"changes sign of stress in all configurations of db.cfg"
	) {

		if (args.size() != 1) {
			std::cout << "\tError: 1 arguments required\n";
			return 1;
		}

		const string cfg_filename = args[0];

		auto cfgs = LoadCfgs(cfg_filename);
		
		{ ofstream ofs(cfg_filename); }

		int counter=0;
		for (auto cfg : cfgs)
		{
			cfg.stresses *= -1;
			cfg.AppendToFile(cfg_filename);
			cout << ++counter << endl;
		}

	} END_COMMAND;

	return is_command_found;
}

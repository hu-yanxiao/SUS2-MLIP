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

using namespace std;

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

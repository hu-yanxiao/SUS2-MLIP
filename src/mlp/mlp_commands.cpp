/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <sstream>
#ifdef MLIP_MPI
#	include <mpi.h>
#endif
#include "mlp.h"
#ifdef MLIP_DEV
#	include "../../dev_src/mlp/mtpr_train.h"
#	include "../../dev_src/mlp/dev_self_test.h"
#endif
#include "../../src/common/stdafx.h"
#include "../../src/mlip_wrapper.h"
#include "../../src/drivers/basic_drivers.h"
#include "../../src/drivers/relaxation.h"
#include "../../src/mlp/self_test.h"
#include "../../src/mlp/train.h"
#include "../../src/mlp/calc_errors.h"

const char version_str[] = "2.1.1";

using namespace std;


// does a number of unit tests (not dev)
// returns true if all tests are successful
// otherwise returns false and stops further tests
bool self_test()
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
		if(!RunAllTests(false)) return false;
	}
#ifdef MLIP_MPI
	if (mpi_rank == 0) cout << "MPI pub tests (" << mpi_size << " cores):" << endl;
	if(!RunAllTests(true)) return false;
#endif // MLIP_MPI

	logstream.close();
	return true;
}

bool Commands(const string& command, vector<string>& args, map<string, string>& opts)
{
	bool is_command_found = false;
	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	if (command == "list" || command == "help") {
		if(command == "list" || args.size() == 0) is_command_found = true;
		if (mpi_rank == 0) {
			cout << "mlp "
#ifdef MLIP_MPI
				<< "mpi version " << version_str << " (" << mpi_size << " cores),\n"
#else
				<< "serial version " << version_str << ",\n"
#endif
				<< "(C) Alexander V. Shapeev, Evgeny V. Podryabinkin, Konstantin Gubaev, Ivan S. Novikov (Skoltech).\n";
			if (command == "help" && args.size() == 0)
				cout << USAGE;
			if (command == "list")
				cout << "List of available commands:\n";
		}
	}

	BEGIN_UNDOCUMENTED_COMMAND("run",
		"reads configurations from a database and processes them",
		"mlp run settings-file [options]:\n"
		"reads configurations from a database and processes them according to settings-file.\n"
		"  Options can be given in any order. Options include:\n"
		"  --filename=<str>: Read configurations from <str>\n"
		"  --limit=<num>: maximum number of configurations to read\n"
		"Works only in a serial (non-mpi) way.\n"
		) {

		if (args.size() == 0) {
			cout << "mlp run: minimum 1 argument required\n";
			return 1;
		}

		ios::sync_with_stdio(false);

		MLIP_Wrapper mlip_wrapper(LoadSettings(args[0]));
		Settings driver_settings(opts);
		CfgReader cfg_reader(&mlip_wrapper, driver_settings);
		cfg_reader.Run();
	} END_COMMAND;

	BEGIN_COMMAND("relax",
		"relaxes the configuration(s)",
		"mlp relax settings-file [options]:\n"
		"settings file should contain settings for relaxation and for mlip regime.\n"
		"  Options can be given in any order. Options include:\n"
		"  --pressure=<num>: external pressure (in GPa)\n"
		"  --iteration-limit=<num>: maximum number of iterations\n"
		"  --min-dist=<num>: terminate relaxation if atoms come closer than <num>\n"
		"  --force-tolerance=<num>: relaxes until forces are less than <num>(eV/Angstr.)\n"
		"        Zero <num> disables atom relaxation (fixes atom fractional coordinates)\n"
		"  --stress-tolerance=<num>: relaxes until stresses are smaller than <num>(GPa)\n"
		"        Zero <num> disables lattice relaxation\n"
		"  --max-step=<num>: Maximal allowed displacement of atoms and lattice vectors\n"
		"        (in Angstroms)\n"
		"  --min-step=<num>: Minimal displacement of atoms and lattice vectors (Angstr.)\n"
		"        If all actual displacements are smaller then the relaxation stops.\n"
		"  --bfgs-wolfe_c1\n"
		"  --bfgs-wolfe_c2\n"
		"  --cfg-filename=<str>: Read initial configurations from <str>\n"
		"  --save-relaxed=<str>: Save the relaxed configurations to <str>\n"
		"  --save-unrelaxed=<str>: If relaxation failed, save the configuration to <str>\n"
		"  --log=<str>: Write relaxation log to <str>\n"
	) {

		if (args.size() == 0) {
			cout << "mlp relax: minimum 1 argument required\n";
			return 1;
		}

		ios::sync_with_stdio(false);

		MLIP_Wrapper mlip_wrapper(LoadSettings(args[0]));
		Message("Driver: internal relaxation");
		Settings driver_settings(opts);
		Relaxation rlx(&mlip_wrapper, driver_settings);
		// TODO: load one-by-one (what if there are too many configurations?)
		// TODO: save into appropriate files

		vector<Configuration> cfgs;
		
		int total_cfgs = 0;

#ifdef MLIP_MPI
		cfgs = LoadCfgs(opts["cfg-filename"],HUGE_INT,mpi_rank,mpi_size);
		int m = (int)cfgs.size();
		MPI_Reduce(&m, &total_cfgs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD); // This avoids bugs in wrongly configuted environments
#else
		cfgs = LoadCfgs(opts["cfg-filename"]);
#endif	

		

		if (mpi_rank == 0)
			cout << "Total " << total_cfgs << " configurations are to be relaxed\n" << endl;

		Configuration cfg_orig;								//to save this cfg if failed to relax
		int count = 0;
		bool error;														//was the relaxation successful or not

		ofstream relaxed, unrelaxed;

		if (opts["save-relaxed"] != "")
			relaxed.open(opts["save-relaxed"] + "_" + to_string(mpi_rank));

		if (opts["save-unrelaxed"] != "")
			unrelaxed.open(opts["save-unrelaxed"] + "_" + to_string(mpi_rank));

		for (auto& cfg : cfgs)
		{
				error = false;
				cfg_orig = cfg;
				rlx.cfg = cfg;

				try { rlx.Run(); }
				catch (MlipException& e) {
					Warning("Relaxation failed: " + e.message);
					error = true;
				}

				if (!error) {
					if (opts["save-relaxed"] != "")
						rlx.cfg.Save(relaxed);
				}
				else {
					if (opts["save-unrelaxed"] != "")
						cfg_orig.Save(unrelaxed);
				}

			
			count++;
		}

		if (opts["save-relaxed"] != "")
			relaxed.close();
		if (opts["save-unrelaxed"] != "")
			unrelaxed.close();
	} END_COMMAND;

	//BEGIN_COMMAND("read-from-db",
	//	"reads the configurations from a database",
	//	"mlp read-from-db settings-file [options]:\n"
	//	"settings file should contain settings for relaxation and for mlip regime.\n"
	//	"Works only in a serial (non-mpi) way.\n"
	//	) {

	//	if (args.size() == 0) {
	//		cout << "mlp run: minimum 1 argument required\n";
	//		return 1;
	//	}

	//	ios::sync_with_stdio(false);
	//	ControlUnit2* shell = new ControlUnit2(args[0], opts);
	//	Message("Driver: database reader");
	//	CfgReader* p_driver = new CfgReader(shell->p_wrp, shell->settings);
	//	p_driver->Run();
	//	delete p_driver;
	//	delete shell;
	//} END_COMMAND;
	
	BEGIN_UNDOCUMENTED_COMMAND("make-cell-square",
		"makes the unit cell as square as possible by replicating it",
		"mlp make-cell-square in.cfg out.cfg\n"
		"  Options:\n"
		"  -min-atoms=<int>: minimal number of atoms (default: 32)\n"
		"  -max-atoms=<int>: maximal number of atoms (default: 64)\n"
	) {
		int min_atoms=32, max_atoms=64;
		if (opts["min-atoms"] != "") {
			try { min_atoms = stoi(opts["min-atoms"]); }
			catch (invalid_argument) {
				cerr << (string)"mlp error: " + opts["min-atoms"] + " is not an integer\n";
			}
		}
		if (opts["max-atoms"] != "") {
			try { max_atoms = stoi(opts["max-atoms"]); }
			catch (invalid_argument) {
				cerr << (string)"mlp error: " + opts["max-atoms"] + " is not an integer\n";
			}
		}

		Configuration cfg;
		ifstream ifs(args[0], ios::binary);
		ofstream ofs(args[1], ios::binary);
		while(cfg.Load(ifs))
		{
			int cfg_size = cfg.size();
			const int M = 4;
			Matrix3 A, B;
			Matrix3 L = cfg.lattice * cfg.lattice.transpose();
			Matrix3 X;
			double score = 1e99;
			for (A[0][0] = -M; A[0][0] <= M; A[0][0]++)
				for (A[0][1] = -M; A[0][1] <= M; A[0][1]++)
					for (A[0][2] = -M; A[0][2] <= M; A[0][2]++)
						for (A[1][0] = -M; A[1][0] <= M; A[1][0]++)
							for (A[1][1] = -M; A[1][1] <= M; A[1][1]++)
								for (A[1][2] = -M; A[1][2] <= M; A[1][2]++)
									for (A[2][0] = -M; A[2][0] <= M; A[2][0]++)
										for (A[2][1] = -M; A[2][1] <= M; A[2][1]++)
											for (A[2][2] = -M; A[2][2] <= M; A[2][2]++) {
												if (fabs(A.det()) * cfg_size < min_atoms) continue;
												if (fabs(A.det()) * cfg_size > max_atoms) continue;
												X = A*L*A.transpose();
												double shift = (X[0][0] + X[1][1] + X[2][2]) / 3;
												X[0][0] -= shift;
												X[1][1] -= shift;
												X[2][2] -= shift;
												double curr_score = X.NormFrobeniusSq();
												if (curr_score < score) {
													score = curr_score;
													B = A;
												}
											}
			for (int a = 0; a < 3; a++) {
				for (int b = 0; b < 3; b++)
					cout << (int)B[a][b] << " ";
				cout << "\n";
			}
			cout << B.det() << "\n";
			cfg.ReplicateUnitCell(templateMatrix3<int>((int)B[0][0], (int)B[0][1], (int)B[0][2], (int)B[1][0], (int)B[1][1], (int)B[1][2], (int)B[2][0], (int)B[2][1], (int)B[2][2]));
			cfg.CorrectSupercell();
			cfg.Save(ofs);
		}
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("replicate-cell",
		"replicates unit cell",
		"mlp replicate-cell in.cfg out.cfg nx ny nz:\n"
		"or"
		"mlp replicate-cell in.cfg out.cfg nxx nxy nxz nyx nyy nyz nzx nzy nzz:\n"
		"Replicate the supercells by nx*ny*nz.\n"
		"Alternatively, applies the 3x3 integer matrix to the lattice\n"
	) {
		if (args.size() == 5) {
			int nx, ny, nz;
			try { nx = stoi(args[2]); }
			catch (invalid_argument) {
				cerr << (string)"mlp error: " + args[2] + " is not an integer\n";
			}
			try { ny = stoi(args[3]); }
			catch (invalid_argument) {
				cerr << (string)"mlp error: " + args[3] + " is not an integer\n";
			}
			try { nz = stoi(args[4]); }
			catch (invalid_argument) {
				cerr << (string)"mlp error: " + args[4] + " is not an integer\n";
			}

			Configuration cfg;
			ifstream ifs(args[0], ios::binary);
			ofstream ofs(args[1], ios::binary);
			while (cfg.Load(ifs)) {
				cfg.ReplicateUnitCell(nx, ny, nz);
				cfg.Save(ofs);
			}
		} else if (args.size() == 11) {
			int args_int[3][3];
			for (int i = 0; i < 9; i++) {
				try { args_int[i/3][i%3] = stoi(args[i+2]); }
				catch (invalid_argument) {
					cerr << (string)"mlp error: " + args[i+2] + " is not an integer\n";
				}
			}
			templateMatrix3<int> A(args_int);

			Configuration cfg;
			ifstream ifs(args[0], ios::binary);
			ofstream ofs(args[1], ios::binary);
			while (cfg.Load(ifs)) {
				cfg.ReplicateUnitCell(A);
				cfg.Save(ofs);
			}
		}
	} END_COMMAND;

	BEGIN_COMMAND("convert-cfg",
		"convert a file with configurations from one format to another",
		"mlp convert-cfg [options] inputfilename outputfilename\n"
		"  input filename sould always be entered before output filename\n"
		"  Options can be given in any order. Options include:\n"
		"  --input-format=<format>: format of the input file\n"
		"  --output-format=<format>: format of the output file\n"
		"  --append:      opens output file in append mode\n"
		"  --last:        ignores all configurations in the input file except the last one\n"
		"                 (useful with relaxation)\n"
		"  --fix-lattice: creates an equivalent configuration by moving the atoms into\n"
		"                 the supercell (if they are outside) and tries to make the\n"
		"                 lattice as much orthogonal as possible and lower triangular.\n"
		"\n"
		"  <format> can be:\n"
		"  txt (default): mlip textual format\n"
		"  bin:           mlip binary format\n"
		"  vasp-outcar:   only as input; VASP versions 5.3.5 and 5.4.1 were tested\n"
		"  vasp-poscar:   only as output. When writing multiple configurations,\n"
		"                 POSCAR0, POSCAR1, etc. are created\n"
		"  lammps-datafile: only as output. Can be read by read_data from lammps.\n"
		"                 Multiple configurations are saved to several files.\n"
		) {
		// additional formats (so far, an undocumented feature):
                // json
		if (opts["input-format"] == "") opts["input-format"] = "txt";
		if (opts["output-format"] == "") opts["output-format"] = "txt";

		if (opts["output-format"] == "vasp-outcar")
			ERROR("Format " + opts["output-format"] + " is not allowed for output");
		if (opts["input-format"] == "vasp-poscar" || opts["input-format"] == "lammps-datafile" || opts["input-format"] == "json")
			ERROR("Format " + opts["input-format"] + " is not allowed for input");

		const bool possibly_many_output_files = (opts["output-format"] == "vasp-poscar")
			|| (opts["output-format"] == "lammps-datafile");

		ifstream ifs(args[0], ios::binary);
		if (ifs.fail()) ERROR("cannot open " + args[0] + " for reading");

		ofstream ofs;
		if (!possibly_many_output_files) {
			if (opts["append"] == "") {
				ofs.open(args[1], ios::binary);
				if (ofs.fail()) ERROR("cannot open " + args[1] + " for writing");
			} else {
				ofs.open(args[1], ios::binary | ios::app);
				if (ofs.fail()) ERROR("cannot open " + args[1] + " for appending");
			}
		}

		vector<Configuration> db;
		int count = 0;
		bool read_success = true;

		// block that reads configurations into the database
		if (opts["input-format"] == "vasp-outcar") {
			ifs.close();
			LoadDynamicsFromOUTCAR(db, args[0]);
		}

		Configuration cfg_last; // used to differentiate between single-configuration and multiple-configuration inputs

		while (read_success) {
			Configuration cfg;

			// Read cfg
			if (db.size() == 0) {
				if (opts["input-format"] == "txt" || opts["input-format"] == "bin")
					// our format
					read_success = cfg.Load(ifs);
				else
					ERROR("unknown file format");
			} else {
				read_success = (count < db.size());
				if (read_success)
					cfg = db[count];
			}
			if (!read_success) break;

			if (opts["fix-lattice"] != "") cfg.CorrectSupercell();

			// Save cfg
			if (opts["last"] == "") {
				if (opts["output-format"] == "txt")
					cfg.Save(ofs);
				if (opts["output-format"] == "bin")
					cfg.SaveBin(ofs);
				if (opts["output-format"] == "vasp-poscar") {
					if (count != 0) {
						if (count == 1) cfg_last.WriteVaspPOSCAR(args[1] + "0");
						cfg.WriteVaspPOSCAR(args[1] + to_string(count));
					}
				}
				if (opts["output-format"] == "lammps-datafile") {
					if (count != 0) {
						if (count == 1) cfg_last.WriteLammpsDatafile(args[1] + "0");
						cfg.WriteLammpsDatafile(args[1] + to_string(count));
					}
				}
				if (opts["output-format"] == "json") {
					if (count != 0) {
						if (count == 1) {
							ofs << "[\n";
							cfg_last.WriteJsonCfg(ofs);
						}
						ofs << ",\n";
						cfg.WriteJsonCfg(ofs);
					}
				}
			}

			cfg_last = cfg;
			count++;
		}
		if (opts["output-format"] == "json" && count >= 2)
			ofs << "\n]\n";
		if (opts["last"] == "") {
			if (opts["output-format"] == "vasp-poscar" && count == 1)
				cfg_last.WriteVaspPOSCAR(args[1]);
			if (opts["output-format"] == "lammps-datafile" && count == 1)
				cfg_last.WriteLammpsDatafile(args[1]);
			if (opts["output-format"] == "json" && count == 1)
				cfg_last.WriteJsonCfg(ofs);
		} else {
			// --last option given
			if (opts["output-format"] == "txt")
				cfg_last.Save(ofs);
			if (opts["output-format"] == "bin")
				cfg_last.SaveBin(ofs);
			if (opts["output-format"] == "vasp-poscar")
				cfg_last.WriteVaspPOSCAR(args[1]);
			if (opts["output-format"] == "lammps-datafile")
				cfg_last.WriteLammpsDatafile(args[1]);
			if (opts["output-format"] == "json")
				cfg_last.WriteJsonCfg(ofs);
		}

		std::cout << "Processed " << count << " configurations" << std::endl;
	} END_COMMAND;

	BEGIN_COMMAND("train",
		"fits an MTP",
		"mlp train potential.mtp train_set.cfg [options]:\n"
		"  trains potential.mtp on the training set from train_set.cfg\n"
		"  Options include:\n"
		"    --energy-weight=<double>: weight of energies in the fitting. Default=1\n"
		"    --force-weight=<double>: weight of forces in the fitting. Default=0.01\n"
		"    --stress-weight=<double>: weight of stresses in the fitting. Default=0.001\n"
		"    --scale-by-force=<double>: Default=0. If >0 then configurations near equilibrium\n"
		"                               (with roughtly force < <double>) get more weight. \n"
		"    --valid-cfgs=<string>: filename with configuration to validate\n"
		"    --max-iter=<int>: maximal number of iterations. Default=1000\n"
		"    --curr-pot-name=<string>: save potential on each iteration if <string> not empty.\n"
		"    --trained-pot-name=<string>: filename for trained potential. Default=Trained.mtp_\n"
                "    --distribution-weight=<double>: weight of distribution in fitting. Default=1\n"
		"    --bfgs-conv-tol=<double>: stop if error dropped by a factor smaller than this\n"
		"                              over 50 BFGS iterations. Default=1e-3\n"	
		"    --weighting=<string>: how to weight configuration wtih different sizes\n"
		"        relative to each other. Default=vibrations. Other=molecules, structures.\n"
		"    --init-params=<string>: how to initialize parameters if a potential was not\n"
		"        pre-fitted. Default is random. Other is same - this is when interaction\n"
		"        of all species is the same (more accurate fit, but longer optimization)\n"
		"    --skip-preinit: skip the 75 iterations done when parameters are not given\n"
		"    --update-mindist: updating the mindist parameter with actual \n"
		"                      minimal interatomic distance in the training set\n"
	) {

		if (args.size() < 2) {
			cout << "mlp train: at least 2 arguments are required\n";
			return 1;
		}
                
                //double weight_std = 1.0;
                //if(opts["distribution-weight"] != "") 
                      //  weight_energy = stod(opts["distribution-weight"]);

		double weight_energy = 1.0;
		if(opts["energy-weight"] != "") 
			weight_energy = stod(opts["energy-weight"]);
//                double weight_std = 1.0;
//                if(opts["std-weight"] != "")
//                        weight_std = stod(opts["std-weight"]);
		double weight_force = 0.01;
		if(opts["force-weight"] != "")
			weight_force = stod(opts["force-weight"]);

		double weight_stress = 0.001;
		if(opts["stress-weight"] != "") 
			weight_stress = stod(opts["stress-weight"]);

		double scale_by_force = 0.0;
		if(opts["scale-by-force"] != "") 
			scale_by_force = stod(opts["scale-by-force"]);

#ifdef MLIP_DEV
		Train_MTPR(args,opts);
#else
		std::cout << "This command is supported in DEV mode only" << endl;

		//Train(args,opts);
		//if (stoi(vecstr[vecstr.size()-1]) == 1 && opts["max-iter"] == "")
		//	Train(args[0], args[1], weight_energy, weight_force, weight_stress, scale_by_force, mtrxfnm);
#endif
	} END_COMMAND;                      

	BEGIN_COMMAND("calc-errors",
		"compares EFS calculated by MTP with EFS from the database",
		"mlp calc-errors pot.mtp db.cfg:\n"
		"calculates errors of \"pot.mtp\" on the database \"db.cfg\"\n"
	) {

		if (args.size() != 2) {
			cout << "mlp calc-errors: 2 arguments required\n";
			return 1;
		}

#ifndef MLIP_DEV
		//CalcErrors(args, opts);
		std::cout << "This command is supported in DEV mode only" << endl;
#else
		opts["max-iter"] = "0"; 
		opts["trained-pot-name"] = "";
		Train_MTPR(args,opts); //this will just calculate errors with MTPR
#endif

	} END_COMMAND;

	BEGIN_COMMAND("self-test",
		"performs a number of unit tests",
		"mlp self-test\n"
		) {
		if (!self_test()) exit(1);
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("add-from-outcar",
		"reads one configuration from the OUTCAR(s)",
		"mlp add-from-outcar db outcar [outcar2 outcar3 ...]:\n"
		"reads one configuration from each of the outcars\n"
		"and APPENDs to the database db\n"
		) {

		if (args.size() < 2) {
			cout << "mlp add-from-outcar: minimum 2 arguments required\n";
			return 1;
		}

		ofstream ofs(args[0], std::ios::app|std::ios::binary);
		Configuration cfg;
		for (int i = 1; i < args.size(); i++) {
			cfg.LoadFromOUTCAR(args[i]);
			cfg.Save(ofs);
		}
		ofs.close();
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("add-from-outcar-dyn",
		"reads dynamics from the OUTCAR(s)",
		"mlp add-from-outcar-dyn db outcar [outcar2 outcar3 ...]:\n"
		"reads dynamics configuration from each of the outcars\n"
		"each configuration from the dynamics is APPENDed to the database\n"
		) {

		if (args.size() < 2) {
			cout << "mlp add-from-outcar-dyn: minimum 2 arguments required\n";
			return 1;
		}

		ofstream ofs(args[0], ios::app|ios::binary);
		for (int i = 1; i < args.size(); i++) {
			vector<Configuration> db;
			if (!LoadDynamicsFromOUTCAR(db, args[i]))
				cerr << "Warning: OUTCAR is broken" << endl;
			for (Configuration cfg : db)
				cfg.Save(ofs);
		}
		ofs.close();
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("add-from-outcar-last",
		"reads the dynamics from OUTCAR(s) and gets the last configuration from each OUTCAR",
		"mlp add-from-outcar-last db outcar [outcar2 outcar3 ...]:\n"
		"reads the dynamics from OUTCAR(s) and gets the last configuration from each OUTCAR.\n"
		"APPENDs all last configurations to the database db\n"
		) {

		if (args.size() < 2) {
			std::cout << "mlp add-from-outcar: minimum 2 arguments required\n";
			return 1;
		}

		ofstream ofs(args[0], ios::app|ios::binary);
		Configuration cfg;
		for (int i = 1; i < args.size(); i++) {
			if (!cfg.LoadLastFromOUTCAR(args[i]))
				cerr << "Warning: OUTCAR is broken" << endl;
			cfg.Save(ofs);
		}
		ofs.close();
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("to-poscar",
		"reads one configuration from a database and exports to poscar",
		"mlp to-poscar db [poscar-file]:\n"
		"reads the configuration from db and exports to poscar-file\n"
		"if there is more than one configuration, exports all to\n"
		"poscar-file1, poscar-file2, ...\n"
		"if poscar-file is not given then poscar-file=POSCAR"
		) {

		if (args.size() != 1 && args.size() != 2) {
			cout << "mlp to-poscar: 1 or 2 arguments required\n";
			return 1;
		}

		string poscar_filename;
		if (args.size() == 1) poscar_filename = "POSCAR";
		else poscar_filename = args[1];
		
		ifstream ifs(args[0], ios::binary);
		vector<Configuration> db;
		for (Configuration cfg; cfg.Load(ifs); db.push_back(cfg));
		ifs.close();

		if (db.size() == 0)
			cerr << "Error: the database " << args[0] << " is empty.\n";
		else if(db.size() == 1)
			db[0].WriteVaspPOSCAR(poscar_filename);
		else
			for (int i = 0; i < (int)db.size(); i++)
				db[i].WriteVaspPOSCAR(poscar_filename + to_string(i));
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("to-bin",
		"converts a database to the binary format",
		"mlp to-bin db.cfgs db.bin.cfgs:\n"
		"Reads the database db.cfgs and saves in a binary format to db.bin.cfgs\n"
		) {

		if (args.size() < 3) {
			cout << "mlp to-bin: minimum 2 arguments required\n";
			return 1;
		}

		ifstream ifs(args[0], ios::binary);
		ofstream ofs(args[2], ios::binary);
		Configuration cfg;
		int count;
		for (count = 0; cfg.Load(ifs); count++)
			cfg.SaveBin(ofs);
		ofs.close();
		cout << count << " configurations processed.\n";
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("from-bin",
		"converts a database from the binary format",
		"mlp from-bin db.bin.cfgs db.cfgs:\n"
		"Reads the binary database db.bin.cfgs and saves to db.cfgs\n"
		) {

		if (args.size() < 3) {
			cout << "mlp to-bin: minimum 2 arguments required\n";
			return 1;
		}

		ifstream ifs(args[0], ios::binary);
		ofstream ofs(args[2], ios::binary);
		Configuration cfg;
		int count;

		for (count = 0; cfg.Load(ifs); count++)
			cfg.Save(ofs);

		ofs.close();
		cout << count << " configurations processed.\n";
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("fix-lattice",
		"puts the atoms into supercell (if they are outside) and makes lattice as much orthogonal as possible and lower triangular",
		"mlp fix-lattice in.cfg out.cfg:\n"
		) {

		if (args.size() < 2) {
			cout << "mlp to-bin: minimum 2 arguments required\n";
			return 1;
		}

		ifstream ifs(args[0], ios::binary);
		ofstream ofs(args[2], ios::binary);
		Configuration cfg;
		int count;
		for (count = 0; cfg.Load(ifs); count++) {
			cfg.CorrectSupercell();
			cfg.Save(ofs);
		}
		ofs.close();
		cout << count << " configurations processed.\n";
	} END_COMMAND;
	
	BEGIN_UNDOCUMENTED_COMMAND("compress-extend",
		"compresses/extends input configuration",
		"mlp compress-extend input.cfg deformed.cfg:\n"
		) {	
		
		if (args.size() != 2) {
			cout << "mlp calc-errors: 2 arguments required\n";
			return 1;
		}
		
                Configuration cfg_init, cfg;
                ifstream ifs(args[0]);
                cfg_init.Load(ifs);
                Matrix3 mapping_init(1,0,0,0,1,0,0,0,1);
                Matrix3 mapping;
                ofstream ofs(args[1]);
        
                for (int i = -10; i <= 10; i++) {
                    mapping = mapping_init;
                    cfg = cfg_init;
                    for (int a = 0; a < 3; a++) 
                        mapping[a][a] += i*0.01;
                    cfg.Deform(mapping);
                    cfg.has_energy(false);
                    cfg.has_stresses(false);
                    cfg.has_forces(false);
                    cfg.Save(ofs);
                }
		
	} END_COMMAND;

	BEGIN_COMMAND("mindist",
		"reads a cfg file and saves it with added mindist feature",
        "mlp mindist db.cfg [Options]\n"
        "  Options can be given in any order. Options include:\n"
        "  --no-overwrite: do not change db.cfg, only print global mindist\n"
        ) {

		if (args.size() != 1) {
			cout << "mlp mindist: 1 arguments required\n";
			return 1;
		}
        bool no_overwrite = opts.count("no-overwrite") > 0;
        if (no_overwrite) {
            double global_md = 9e99;
            Configuration cfg;
            ifstream ifs(args[0], ios::binary);
            while (cfg.Load(ifs))
                global_md = std::min(global_md, cfg.MinDist());
            cout << "Global mindist: " << global_md << endl;
        }
        else {
            vector<Configuration> db;
            {
                Configuration cfg;
                ifstream ifs(args[0], ios::binary);
                while (cfg.Load(ifs))
                    db.push_back(cfg);
                ifs.close();
            }
            {
                Configuration cfg;
                ofstream ofs(args[0], ios::binary);
                double global_md = 9e99;
                for (int i = 0; i < db.size(); i++) {
                    double md = db[i].MinDist();
                    db[i].features["mindist"] = to_string(md);
                    db[i].Save(ofs);
                    global_md = std::min(global_md, md);
                }
                cout << "Global mindist: " << global_md << endl;
                ofs.close();
            }
        }
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("subsample",
		"subsamples a database (reduces size by skiping every N configurations)",
		"mlip-subsample db_in.cfg db_out.cfg skipN\n"
		) {

		if (args.size() != 3) {
			cout << "mlp subsample: 3 arguments required\n";
			return 1;
		}

		ifstream ifs(args[0], ios::binary);
		ofstream ofs(args[1], ios::binary);
		Configuration cfg;
		int skipN = stoi(args[2]);
		for (int i = 0; cfg.Load(ifs); i++) {
			if (i % skipN == 0)
				cfg.Save(ofs);
		}
	} END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("fix-cfg",
		"Fixes a database by reading and writing it again",
		"mlp fix-db db.cfg [options]\n"
		"  Options:\n"
		"  --no-loss: writes a file with a very high precision\n"
	) {
		unsigned int flag = 0U;
		if (opts["no-loss"] != "")
			flag |= Configuration::SAVE_NO_LOSS;

		std::vector<Configuration> db;
		{
			Configuration cfg;
			std::ifstream ifs(args[0], std::ios::binary);
			while (cfg.Load(ifs)) {
					db.push_back(cfg);
			}
			ifs.close();
		}
		{
			Configuration cfg;
			std::ofstream ofs(args[0], std::ios::binary);
			for (int i = 0; i < db.size(); i++) {
				db[i].Save(ofs, flag);
			}
			ofs.close();
		}
	}
	END_COMMAND;

	BEGIN_UNDOCUMENTED_COMMAND("filter-nonconv",
		"removes configurations with feature[EFS_by] containing not_converged",
		"mlp filter-nonconv db.cfg\n"
		) {
		if (args.size() != 1) {
		std::cerr << "Error: 1 argument required" << std::endl;
		std::cerr << "Usage: mlp filter-nonconv db.cfg" << std::endl;
		return 1;
	}
	std::vector<Configuration> db;
	{
		Configuration cfg;
		std::ifstream ifs(args[0], std::ios::binary);
		while (cfg.Load(ifs)) {
			if (cfg.features["EFS_by"].find("not_converged") == std::string::npos)
				db.push_back(cfg);
		}
		ifs.close();
	}
	{
		Configuration cfg;
		std::ofstream ofs(args[0], std::ios::binary);
		for (int i = 0; i < db.size(); i++) {
			db[i].Save(ofs);
		}
		ofs.close();
	}
	} END_COMMAND;


	return is_command_found;
}

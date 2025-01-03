/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov, Konstantin Gubaev
 */

#ifndef MLIP_UTILS_SELF_TEST_H
#define MLIP_UTILS_SELF_TEST_H

//	Uncomment for generation of reference results
//#define GEN_TESTS

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else 
#include <cblas.h>
#endif

#include "../drivers/relaxation.h"
#include "../mlip_wrapper.h"

using namespace std;


#define TEST(name) \
test_count++; \
curr_test_name = name; \
if((mpi_rank==0) && (mpi_size>0) && (!is_parallel)) try { \
std::cout << "  Test " << (test_count < 10 ? " " : "") << test_count << "  ...   (" << std::string(curr_test_name).substr(0,60) << ")";

#define PAR_TEST(name) \
test_count++; \
curr_test_name = name; \
if((mpi_rank==0) && is_parallel) std::cout << "  Test " << (test_count < 10 ? " " : "") << test_count << "  ...   (" << std::string(curr_test_name).substr(0,60) << ")"; \
if(is_parallel) try {

#define END_TEST if(mpi_rank==0) std::cout << "\r  Test " << (test_count < 10 ? " " : "") << test_count << " passed (" << std::string(curr_test_name).substr(0,60) << ")" << std::endl;} \
catch(const MlipException& e) { std::cerr << "\nERROR: MLIP Exception caught:\n" << e.What(); return false; }

#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << "\nTest " << test_count << " (" << curr_test_name << ") failed" << std::endl; \
if(!str.empty()) std::cout << "Error message: " << str << std::endl; \
return false;}

#ifdef MLIP_NO_SELFTEST
bool RunAllTests(bool is_parallel) { return true; }
#else
bool RunAllTests(bool is_parallel)
{
	const string PATH = "";

	int mpi_rank=0;
	int mpi_size=1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif // MLIP_MPI

	int test_count = 0;
	std::string curr_test_name;
{} // this tricks MSVS not to create an extra tab
//

TEST("configurations reading") {
	int count = 0;
	int status = 0;

	// read the database
	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	Configuration cfg;
	for (count = 0; (status = cfg.Load(ifs)); count++);
	if ((status != false) || (count != 7)) FAIL()
} END_TEST;

TEST("Binary and normal configurations reading") {
	int count = 0;
	int status = 0;

	Configuration cfg;
	vector<Configuration> set_bin;
	vector<Configuration> set;

	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	std::ofstream ofs("temp/configurations_test1.bin.cfgs", ios::binary);

	for (int i = 0; cfg.Load(ifs); i++)
		cfg.SaveBin(ofs);

	ifs.close();
	ofs.close();

	// read the database
	std::ifstream ifs1("temp/configurations_test1.bin.cfgs", ios::binary);
	std::ifstream ifs2("configurations/reading_test.cfg", ios::binary);
	std::ifstream ifs3("configurations/binary_sample.cfg", ios::binary);

	for (; (status = cfg.Load(ifs1)); count++) {
		set_bin.push_back(cfg);
	}
	for (; (status = cfg.Load(ifs2)); count++) {
		set.push_back(cfg);
	}
	for (; (status = cfg.Load(ifs3)); count++);

	bool flag = true;
	for (int i = 0; i < set.size(); i++) {
		if (set[i] != set_bin[i]) {
			flag = false;
			break;
		}
	}

	if ((status != false) || (count != 15) || !flag) FAIL()

		ifs1.close();
	ifs2.close();
	ifs3.close();

} END_TEST;

TEST("Loading configuration database to a vector") {
	vector<Configuration> cfgs(LoadCfgs("configurations/reading_test.cfg"));
	vector<Configuration> cfgs2 = LoadCfgs("configurations/reading_test.cfg");
	Configuration cfg;
	ifstream ifs("configurations/reading_test.cfg", ios::binary);
	for (int i = 0; cfg.Load(ifs); i++)
		if (cfg != cfgs[i] || cfg != cfgs2[i])
			FAIL();

} END_TEST;

TEST("Binary configurations file format and old format r/w") {
	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	std::ofstream ofs_bin("temp/test1.bin.cfgs", std::ios::binary);
	Configuration cfg;

	for (int count = 0; cfg.Load(ifs); count++)
		cfg.SaveBin(ofs_bin);

	ifs.close();
	ofs_bin.close();

	ifs.open("configurations/reading_test.cfg", ios::binary);
	std::ifstream ifs_bin("temp/test1.bin.cfgs", std::ios::binary);
	Configuration cfgb;

	while (cfgb.Load(ifs_bin)) {
		cfg.Load(ifs);
		if (cfg.size() != cfgb.size())
			FAIL();

		if (!(cfg.lattice == cfgb.lattice))
			FAIL();

		for (int i = 0; i < cfg.size(); i++)
			if (!(cfg.pos(i) == cfgb.pos(i)))
				FAIL();

		if ((cfg.has_forces() != cfgb.has_forces()) ||
			(cfg.has_energy() != cfgb.has_energy()) ||
			(cfg.has_stresses() != cfgb.has_stresses()) ||
			(cfg.has_site_energies() != cfgb.has_site_energies()))
			FAIL();

		if (cfg.has_energy())
			if (cfg.energy != cfgb.energy)
				FAIL();

		if (cfg.has_forces())
			for (int i = 0; i < cfg.size(); i++)
				if (!(cfg.force(i) == cfgb.force(i)))
					FAIL();

		if (cfg.has_stresses())
			if (!(cfg.stresses == cfgb.stresses))
				FAIL();

		if (cfg.has_site_energies())
			for (int i = 0; i < cfg.size(); i++)
				if (cfg.site_energy(i) != cfgb.site_energy(i))
					FAIL();
	}

	ifs.close();
	ifs_bin.close();
} END_TEST;

TEST("New format configurations reading/writing test") {
	int count = 0;
	int status = 0;

	// read the database
	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	Configuration cfg;
	vector<Configuration> cfgvec;
	ofstream ofs("temp/cfg_rw_test.cfgs");
	for (count = 0; (status = cfg.Load(ifs)); count++) {
		cfgvec.push_back(cfg);
		cfg.Save(ofs, Configuration::SAVE_NO_LOSS);
	}
	if ((status != false) || (count != 7)) FAIL();
	ifs.close();
	ofs.close();

	ofs.open("temp/cfg_rw_test2.cfgs", ios::binary);
	ifs.open("temp/cfg_rw_test.cfgs", ios::binary);
	for (count = 0; (status = cfg.Load(ifs)); count++) {
		if (cfg != cfgvec[count])
			FAIL();
		cfg.Save(ofs);
	}
	if ((status != false) || (count != 7)) FAIL();
	ofs.close();
	ifs.close();

} END_TEST;

TEST("Identical configurations reading test") {
	int count = 0;
	bool status = 0;

	for (int i = 1; i <= 3; i++) {
		// read the database
		std::ifstream ifs("configurations/identical/all_identical" + to_string(i) + ".cfg", std::ios::binary);
		Configuration cfg_init, cfg;
		status = cfg_init.Load(ifs);
		if (status != true) FAIL();
		for (count = 1; (status = cfg.Load(ifs)); count++)
			if (cfg != cfg_init) FAIL();
		ifs.close();
	}

} END_TEST;

TEST("Testing Configuration periodic extension") {
	std::ifstream ifs("configurations/period_ext_test.cfg", std::ios::binary);
	MTP mtp("mlips/fitted.mtp");
	for (int i = 0; i < 3; i++) {
		Configuration cfg1, cfg2;
		cfg1.Load(ifs);
		cfg2.Load(ifs);
		mtp.CalcEFS(cfg1);
		mtp.CalcEFS(cfg2);
		if (cfg1.stresses != cfg2.stresses) FAIL();
	}

} END_TEST;

TEST("Testing reading configuration with fractional coordinates") {
	// read the database
	std::ifstream ifs("configurations/reading_test.cfg", std::ios::binary);
	Configuration cfg1, cfg2;
	cfg1.Load(ifs);
	ifs.close();
	ofstream ofs(PATH + "temp/frac_coord.cfg");
	cfg1.Save(ofs, Configuration::SAVE_DIRECT_COORDS | Configuration::SAVE_NO_LOSS);
	ofs.close();
	ifs.open(PATH + "temp/frac_coord.cfg", std::ios::binary);
	cfg2.Load(ifs);
	for (int i=0; i<cfg1.size(); i++) 
		if ((cfg1.pos(i) - cfg2.pos(i)).Norm() > 1.0e-13)
			FAIL();

} END_TEST;

TEST("Testing Configuration::features") {
	std::ofstream ofs;
	Configuration cfg2;

	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	Configuration cfg;
	if (!cfg.Load(ifs)) FAIL()
		ifs.close();
	cfg.features["test_feature1"] = "ABC";
	cfg.features["test feature2"] = "DEF";
	cfg.features["test_feature3"] = "line with spaces";
	cfg.features["test_feature4"] = "line with spaces and\nnewline";
	cfg.features["test feature5"] = "line with \" \" - space";
	cfg.features["test_feature5"] = "line with spaces and newline at end\n";
	cfg.features["\n"] = "\n   mo\re \trash i\n fea\tu\res \n";
//	cfg.features[" "] = "\n";
//	cfg.features[" "] = "";

	ofs.open("temp/temp.cfgs", ios::binary);
	cfg.Save(ofs);
	ofs.close();
	ifs.open("temp/temp.cfgs", ios::binary);
	if (!cfg2.Load(ifs)) FAIL();
	ifs.close();
	if (cfg.features != cfg2.features) FAIL();

	ofs.open("temp/temp.cfgs", std::ios::binary);
	cfg.SaveBin(ofs);
	ofs.close();
	ifs.open("temp/temp.cfgs", std::ios::binary);
	if (!cfg2.Load(ifs)) FAIL();
	ifs.close();
	if (cfg.features != cfg2.features) FAIL();
} END_TEST;

TEST("VASP OUTCAR iterations count") {

	Configuration cfg;
	cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_converged");

	if (cfg.features["EFS_by"] != "VASP")
		FAIL();

	cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_not_converged");

	if (cfg.features["EFS_by"] != "VASP_not_converged")
		FAIL();

} END_TEST;

TEST("BLAS_test") {
	double A[6] = { 1.0,2.0,1.0,-3.0,4.0,-1.0 };
	double B[6] = { 1.0,2.0,1.0,-3.0,4.0,-1.0 };
	double C[9] = { .5,.5,.5,.5,.5,.5,.5,.5,.5 };
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3, 2, C, 3);

	if (fabs(C[0] - 11) > 1e-6 || fabs(C[1] - -9) > 1e-6 || fabs(C[2] - 5) > 1e-6 || fabs(C[3] - -9) > 1e-6 || 
	fabs(C[4] - 21) > 1e-6 || fabs(C[5] - -1) > 1e-6 || fabs(C[6] - 5) > 1e-6 || fabs(C[7] - -1) > 1e-6 || fabs(C[8] - 3) > 1e-6)
		FAIL();
} END_TEST;

TEST("Errorneus configurations reading test") {
	// read the database
	for (int i=1; i<=9; i++) {
		std::ifstream ifs((std::string)"configurations/errorneous/" + to_string(i) + ".cfg");
		if (!ifs.is_open()) FAIL();
		Configuration cfg;
		bool is_valid = true;
		try { cfg.Load(ifs); }
		catch (const MlipException&) {
			is_valid = false;
		}
		if (is_valid) FAIL();
		ifs.close();
	}

} END_TEST;

TEST("VASP loading") {
	{
#ifdef GEN_TESTS
		Configuration cfg;
		std::ofstream ofs;

		cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_usual");
		ofs.open("configurations/VASP/cfg_usual");
		cfg.Save(ofs);
		ofs.close();

		cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_no_stresses");
		ofs.open("configurations/VASP/cfg_no_stresses");
		cfg.Save(ofs);
		ofs.close();
#endif
	}
	Configuration cfg1, cfg2;
	std::ifstream ifs;

	cfg1.LoadFromOUTCAR("configurations/VASP/OUTCAR_usual");
	ifs.open("configurations/VASP/cfg_usual"); cfg2.Load(ifs); ifs.close();
	if (cfg1 != cfg2) FAIL();

	cfg1.LoadFromOUTCAR("configurations/VASP/OUTCAR_no_stresses");
	ifs.open("configurations/VASP/cfg_no_stresses"); cfg2.Load(ifs); ifs.close();
	if (cfg1 != cfg2) FAIL();

	std::vector<Configuration> db;
	if (!LoadDynamicsFromOUTCAR(db, "configurations/VASP/OUTCAR_relax"))
		FAIL();

	if (!cfg1.LoadLastFromOUTCAR("configurations/VASP/OUTCAR_relax"))
		FAIL("Valid OUTCAR dynamics read, but returned false");
	if (cfg1 != db.back())
		FAIL();

	if (LoadDynamicsFromOUTCAR(db, "configurations/VASP/OUTCAR_broken"))
		FAIL("Broken OUTCAR read, but returned true");

} END_TEST;

TEST("MTP file format v1.1.0 test") {
	try {
		MTP mtp102("mlips/fitted.mtp");
		mtp102.Save("temp/MTPv1.1.0test.mtp");
		MTP mtp103("temp/MTPv1.1.0test.mtp");

		Configuration cfg1;
		Configuration cfg2;
		ifstream ifs("configurations/reading_test.cfg", ios::binary);
		cfg1.Load(ifs);
		cfg2 = cfg1;
		mtp103.CalcEFS(cfg1);
		mtp102.CalcEFS(cfg2);

		if (cfg1.energy != cfg2.energy)
			FAIL();
		for (int i=0; i<cfg1.size(); i++)
			if (cfg1.force(i) != cfg2.force(i))
				FAIL();
		if (cfg1.stresses[0][0] != cfg2.stresses[0][0] ||
			cfg1.stresses[1][1] != cfg2.stresses[1][1] ||
			cfg1.stresses[2][2] != cfg2.stresses[2][2] ||
			cfg1.stresses[0][1] != cfg2.stresses[0][1] ||
			cfg1.stresses[0][2] != cfg2.stresses[0][2] ||
			cfg1.stresses[1][2] != cfg2.stresses[1][2])
			FAIL();
	}
	catch (...)
		FAIL();

} END_TEST;

TEST("MTP file reading 1") {
	MTP basis("mlips/lightweight.mtp");
} END_TEST;

TEST("MTP file reading 2") {
	MTP mtp("mlips/lightweight.mtp");
	for (double i : mtp.regress_coeffs)
		if (i != 0.0) 
			FAIL()
} END_TEST;

TEST("MTP file reading 3") {
	MTP mtp("mlips/fitted.mtp");
	double res=0;
	for (double i : mtp.regress_coeffs)
		res += i;
	if (res == 0.0)
		FAIL()
} END_TEST;

TEST("MTP learning") {
	MTP mtp("mlips/unfitted.mtp");
	LinearRegression regr(&mtp, 500, 1, 10);
	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	Configuration cfg;

	vector<Configuration> training_set;
	for (int count = 0; cfg.Load(ifs); count++)
		training_set.push_back(cfg);
	regr.Train(training_set);
	ifs.close();

#ifdef GEN_TESTS
	mtp.Save("mlips/fitted.mtp");
#endif

	// the MTP that is already trained
	MTP mtp_ready("mlips/fitted.mtp");
	ifs.open("configurations/reading_test.cfg", ios::binary);
	for (int count=0; cfg.Load(ifs); count++) {
		Configuration cfg2(cfg);

		// check computing energy with trained vs stored coefficients
		mtp.CalcEFS(cfg);
		mtp_ready.CalcEFS(cfg2);
		if (abs(cfg2.energy - cfg.energy) > 1e-6) {
			std::cerr << "training produced wrong coefficients (error " << abs(cfg2.energy - cfg.energy) << ")";
			FAIL()
		}
		//std::cout << cfg2.energy << " " << cfg.energy << std::endl;

		// check that CalcEFS and calc_E give the same energy
		mtp_ready.CalcEFS(cfg);
		mtp_ready.CalcE(cfg2);
		if (abs(cfg2.energy - cfg.energy) > 1e-6) {
			std::cerr << "CalcEFS differs from calc_E (error " << abs(cfg2.energy - cfg.energy) << ")";
			FAIL()
		}

		// check the optimized (backpropagation) procedure
		mtp.CalcEFSDebug(cfg);
		mtp.CalcEFS(cfg2);
		for (int i=0; i<cfg.size(); i++)
			if (distance(cfg2.force(i), cfg.force(i)) > 1e-6) {
				std::cerr << "CalcEFS differs from CalcEFS_opt (force error " << distance(cfg2.force(i), cfg.force(i)) << ")";
				FAIL()
			}

		if (norm(cfg2.stresses - cfg.stresses) > 1e-6) {
			std::cerr << "CalcEFS differs from CalcEFS_opt (stress error " << norm(cfg2.stresses - cfg.stresses) << ")";
			FAIL()
		}
	}
} END_TEST;

/*PAR_TEST("parallel trainer test")
{
	{
		MTP mtp("paralel_fitting_test.mtp");
		mtp.Save("temp/paralel_fitting_test.mtp");
	}
	MTP mtp("temp/paralel_fitting_test.mtp");
	LinearRegression regr(&mtp, 500, 1, 10);
	auto cfgs = LoadCfgs("configurations/reading_test.cfg");
	regr.Train(cfgs);

	train("temp/paralel_fitting_test.mtp", "configurations/reading_test.cfg", 500,1,10,0, "", nullptr);

	for (auto& cfg : cfgs)
		mtp.CalcEFS(cfg);

	MTP mtpp("temp/paralel_fitting_test.mtp");

	ErrorMonitor errmon;
	for (auto& cfg : cfgs)
	{
		Configuration cfgg(cfg);
		errmon.compare(cfg, cfgg);
		if (errmon.ene_cfg.delta > 1.0e-13 ||
			errmon.frc_cfg.max.delta > 1.0e-12 ||
			errmon.str_all.max.delta > 1.0e-13)
			FAIL( + to_string(errmon.ene_cfg.delta) + " " +
				to_string(errmon.frc_cfg.max.delta) + " " +
				to_string(errmon.str_all.max.delta));
	}

} END_TEST;*/
	return true;
}

#endif //  MLIP_NO_SELFTEST

#endif // SELF_TEST

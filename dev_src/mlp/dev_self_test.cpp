/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *   This file contributors: Ivan Novikov, Konstantin Gubaev
 */

//	Uncomment for generation of reference results
// #define GEN_TESTS

#ifdef MLIP_MPI
#include <mpi.h>
#endif

#include <fstream>
#include <iostream>
#include <ctime>

#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else 
#include <cblas.h>
#endif

#include "../sample_potentials.h"
#include "../mtpr.h"
//#include "../mtpr2.h"
#include "../eam.h"
#include "../../src/drivers/relaxation.h"
//#include "../taper.h"
//#include "../coul_qeq.h"
//#include "../mult_trainer.h"
#include "../mtpr_trainer.h"
//#include "../mtpr_trainer2.h"
#include "../../src/mlip_wrapper.h"
#include "../../src/mlp/self_test.h"
#include "../sw_basis.h"

//#ifndef MLIP_NOEWALD
//#	include "../mlip_plus_qeq.h"
//#	include "../mtpr_plus_qeq.h"
//#endif

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
bool RunAllTestsDev(bool is_parallel) { return true; }
#else
#pragma GCC optimize ("O1")
bool RunAllTestsDev(bool is_parallel)
{
	const string PATH = "../../dev_test/";

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

TEST("EAMSimple CalcEnergyGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double diffstep = 1e-5;
	double relErrMax = 0.0, absErrMax = 0.0;
	EAMSimple eam(8, 5, 1.9);
	std::vector<double> energy_grad(eam.CoeffCount());

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam.Coeff()[i] = 20.0 / (i + 1);
		eam.Coeff()[i] = 0.1*sin(i);
	}

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	for (int i = 0; cfg.Load(ifs); i++) {
		FillWithZero(energy_grad);
		eam.CalcEnergyGrad(cfg, energy_grad);
		for (int j = 0; j < eam.CoeffCount(); j++) {
			eam.Coeff()[j] -= diffstep;
			eam.CalcEFS(cfg); double energy_m = cfg.energy;
			eam.Coeff()[j] += 2 * diffstep;
			eam.CalcEFS(cfg); double energy_p = cfg.energy;
			eam.Coeff()[j] -= diffstep;
			double cdiff = (energy_p - energy_m) / (2 * diffstep);
			//std::cout << energy_p << " " << energy_m << std::endl;
			//std::cout << energy_grad[j] << " " << cdiff  << " "  << fabs(energy_grad[j]-cdiff) << " ";
			//std::cout << fabs((energy_grad[j]-cdiff)/cdiff) << std::endl;
			double absErr = fabs(energy_grad[j] - cdiff);
			double relErr = fabs((energy_grad[j] - cdiff) / cdiff);
			if (relErr > relErrMax) relErrMax = relErr;
			if (absErr > absErrMax) absErrMax = absErr;
			//std::cout << cdiffForce << " " << cfg.force(i)[l] << std::endl;
		}
	}

	if (relErrMax > 0.02) {
		std::cout << "relErrMax is " << relErrMax;
		FAIL()
	}

	ifs.close();
} END_TEST;

TEST("EAMSimple CalcEFS check with finite differences") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-4;
	double control_delta = 1E-3;
	EAMSimple eam(4, 4, 1.6);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam.Coeff()[i] = 1.0 / pow(2, i);
		//eam.Coeff()[i] = 0.25/(i+1);
		eam.Coeff()[i] = 0.1*sin(i + 1);
	}

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	cfg.Load(ifs);
	if (!eam.CheckEFSConsistency_debug(cfg, displacement, control_delta))
		FAIL()

	ifs.close();
} END_TEST;

TEST("EAMSimple AddLossGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 0.001;
	EAMSimple eam(8, 5, 1.6);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam.Coeff()[i] = 1.0 / pow(2, i);
		eam.Coeff()[i] = 2.1*sin(i + 1);
	}

	NonLinearRegression dtr(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

	//ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	for (int i=0; cfg.Load(ifs); i++) {
		if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
			FAIL();
	}

	ifs.close();

} END_TEST;

TEST("EAMMult CalcEFS check with finite differences") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 1E-3;
	EAMMult eam(4, 4, 2, 1.6);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam.Coeff()[i] = 1.0 / pow(2, i);
		//eam.Coeff()[i] = 0.25/(i+1);
		eam.Coeff()[i] = 2.1*sin(i + 1);
	}

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	cfg.Load(ifs);
	if (!eam.CheckEFSConsistency_debug(cfg, displacement, control_delta))
		FAIL()

	ifs.close();
} END_TEST;

TEST("EAMMult AddLossGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-6;
	double control_delta = 0.001;
	EAMMult eam(8, 5, 2, 1.6);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam.Coeff()[i] = 1.0 / pow(2, i);
		eam.Coeff()[i] = 1.1*sin(i + 1);
	}

	NonLinearRegression dtr(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

	//ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	//vector<Configuration> train_set;
	//for (int i = 0; cfg.Load(ifs); i++) {
	//	train_set.push_back(cfg);
	//}

	//unsigned int start_time = clock();
	//	dtr.CalcObjectiveFunctionGrad(train_set);
	//unsigned int end_time = clock();
	//cout << "LossGrad time = " << (end_time - start_time) / 1E+6 << endl;

	//unsigned int start_time2 = clock();
	//	dtr.ObjectiveFunction(train_set);
	//unsigned int end_time2 = clock();
	//cout << "Loss time = " << (end_time2 - start_time2) / 1E+6 << endl;

	//exit(1);

	for (int i=0; cfg.Load(ifs); i++) {
		// FD test
		if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
			FAIL()
	}

	ifs.close();

} END_TEST;

/*TEST("Stillinger-Weber CalcEFS check with finite differences") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 1E-3;
	StillingerWeberRadialBasis sw(PATH+"unlearned.sw");

	for (int i = 0; i < sw.CoeffCount(); i++) {
		sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
		//cout << sw.Coeff()[i] << " ";
	}
	//cout << endl;

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	cfg.Load(ifs);
	if (!sw.CheckEFSConsistency_debug(cfg, displacement, control_delta))
		FAIL()

	ifs.close();
	
} END_TEST;

TEST("Stillinger-Weber CalcEFSvsCalcEFSGrad EFS comparison") {
	Configuration cfg;
	double control_delta = 1E-3;
	StillingerWeberRadialBasis sw(PATH+"unlearned.sw");
	double absEnErr, absForcesErr, absStressesErr;

	for (int i = 0; i < sw.CoeffCount(); i++) {
		sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
		//cout << sw.Coeff()[i] << " ";
	}
	//cout << endl;

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	cfg.Load(ifs);

	Configuration cfg1 = cfg;
	Configuration cfg2 = cfg;	

	vector<double> energy_grad;
	Array3D forces_grad;
	Array3D stress_grad;

	sw.CalcEFS(cfg1);
	sw.CalcEFSGrads(cfg2, energy_grad, forces_grad, stress_grad);

	absEnErr = fabs(cfg1.energy-cfg2.energy); 

	if (absEnErr > control_delta) FAIL();

	absForcesErr = 0;
	for (int i = 0; i < cfg.size(); i++) {
		for (int l = 0; l < 3; l++) {
			double curr_err = fabs(cfg1.force(i)[l]-cfg2.force(i)[l]);
			if (curr_err > absForcesErr) absForcesErr = curr_err;
		}
	}

	if (absForcesErr > control_delta) FAIL();

	absStressesErr = 0;
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			double curr_err = fabs(cfg1.stresses[a][b]-cfg2.stresses[a][b]);
			if (curr_err > absStressesErr) absStressesErr = curr_err;
		}
	}

	if (absForcesErr > control_delta) FAIL();

	ifs.close();
	
} END_TEST;

TEST("Stillinger-Weber CalcEFSGrad check with central differences") {

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);
	
	Configuration cfg;
	cfg.Load(ifs);
	Configuration cfg1 = cfg;
	Configuration cfg2 = cfg;

	StillingerWeberRadialBasis sw(PATH+"unlearned.sw");

	for (int i = 0; i < sw.CoeffCount(); i++) {
		sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
		//cout << sw.Coeff()[i] << " ";
	}
	//cout << endl;

	vector<double> energy_grad;
	Array3D forces_grad;
	Array3D stress_grad;

	vector<double> energy_grad_dummy;
	Array3D forces_grad_dummy;
	Array3D stress_grad_dummy;

	sw.CalcEFSGrads(cfg, energy_grad, forces_grad, stress_grad);

	double delta = 1e-5;
	double control_delta = 1E-3;
	int natom = cfg.size();
	Array2D forces_p;
	forces_p.resize(natom, 3);
	Array2D forces_m;
	forces_m.resize(natom, 3);
	vector<double> stress_p(9);
	vector<double> stress_m(9);

	double relEnErr = 0;
	double relForcesErr = 0; 
	double relStressesErr = 0;

	for (int i = 0; i < sw.CoeffCount(); i++) {
		//cout << "coeff = " << i << endl;
		sw.Coeff()[i] += delta;
		sw.CalcEFS(cfg1);
		sw.Coeff()[i] -= 2*delta;
		sw.CalcEFS(cfg2);
		sw.Coeff()[i] += delta;
		//cout << "energy:" << endl;
		double cdiff = (cfg1.energy - cfg2.energy) / (2*delta);
		double curr_err = fabs((cdiff - energy_grad[i])/cdiff);
		if (curr_err > relEnErr) relEnErr = curr_err;	
		//if (curr_err > 1e-5) 
		//	cout << cdiff << " " << energy_grad[i] << endl;

		//cout << "forces:" << endl;
		for (int j = 0; j < cfg.size(); j++) {
			for (int l = 0; l < 3; l++) {
				//cout << cfg1.force(j, l) << " " << cfg2.force(j, l) << " ";
				double cdiff = (cfg1.force(j, l) - cfg2.force(j, l)) / (2*delta);
				double curr_err = fabs((cdiff - forces_grad(j, i, l))/cdiff);
				if (curr_err > relForcesErr) relForcesErr = curr_err;
				//if (curr_err > 1e-5)
				//	cout << cdiff << " " << forces_grad(j, i, l) << endl;
				//cout << j << " " << l << endl;
			}
		}

		//cout << "stresses:" << endl;
		for (int a = 0; a < 3; a++) {
			for (int b = 0; b < 3; b++) {
				double cdiff = (cfg1.stresses[a][b] - cfg2.stresses[a][b]) / (2*delta);
				double curr_err = fabs((cdiff - stress_grad(a, b, i))/cdiff);
				if (curr_err > relStressesErr) relStressesErr = curr_err;
				//if (curr_err > 1e-5)
				//	cout << cdiff << " " << stress_grad(a, b, i) << endl;
			}
		}

	}

	//cout << relEnErr << endl;
	//cout << relForcesErr << endl;
	//cout << relStressesErr << endl;

	if (relEnErr > control_delta) FAIL();
	if (relForcesErr > control_delta) FAIL();
	if (relStressesErr > control_delta) FAIL();

} END_TEST;

TEST("Stillinger-Weber CalcEnergyGradvsCalcEFSGrad energy grad comparison") {
	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);
	
	Configuration cfg;
	cfg.Load(ifs);

	StillingerWeberRadialBasis sw(PATH+"unlearned.sw");

	for (int i = 0; i < sw.CoeffCount(); i++) {
		sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
		//cout << sw.Coeff()[i] << " ";
	}
	//cout << endl;

	vector<double> energy_grad1;
	Array3D forces_grad;
	Array3D stress_grad;

	vector<double> energy_grad2;

	sw.CalcEFSGrads(cfg, energy_grad1, forces_grad, stress_grad);
	sw.CalcEnergyGrad(cfg, energy_grad2);

	double control_delta = 1e-4;
	double absEnErr = 0;

	for (int i = 0; i < sw.CoeffCount(); i++) {
		if (fabs(energy_grad1[i]-energy_grad2[i]) > absEnErr) {
			absEnErr = fabs(energy_grad1[i]-energy_grad2[i]);
		}
	}

	//cout << absEnErr << endl;

	if (absEnErr > control_delta) FAIL();

	
} END_TEST;*/

TEST("Lennard-Jones CalcEFS check with finite differences") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 1E-3;
	LennardJones lj(PATH+"unlearned_lj_lin.mlip", 1.6, 5);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		//lj.Coeff()[i] = 0.25/(i+1);
		lj.Coeff()[i] = 12.1*sin(i + 1);
	}

	ifstream ifs(PATH+"Al_Ni4cdiffs.cfgs", ios::binary);

	cfg.Load(ifs);
	if (!lj.CheckEFSConsistency_debug(cfg, displacement, control_delta))
		FAIL()

	ifs.close();

} END_TEST;

TEST("Lennard-Jones AddLossGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 0.001;
	LennardJones lj(PATH+"unlearned_lj_lin.mlip", 1.6, 5);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		lj.Coeff()[i] = 121.1*(i + 1);
	}
	//cout << endl;

	NonLinearRegression dtr(&lj, 1.0, 1.0, 0.1, 0.0, 1.0);

	//NonLinearRegression dtr(&lj, 1.0, 1.0, 0.1, 0.0, 1.0);

	//ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	for (int i = 0; cfg.Load(ifs); i++) {
		// FD test
		if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
			FAIL()
	}

	ifs.close();

} END_TEST;

TEST("NonlinearLennard-Jones CalcEFS check with finite differences") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 1E-3;
	NonlinearLennardJones lj(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		//lj.Coeff()[i] = 0.25/(i+1);
		lj.Coeff()[i] = pow(fabs(1.5*sin(i + 1)),0.25);
	}

	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	cfg.Load(ifs);
	if (!lj.CheckEFSConsistency_debug(cfg, displacement, control_delta))
		FAIL()

	ifs.close();

} END_TEST;

TEST("NonlinearLennard-Jones AddLossGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 0.001;
	NonlinearLennardJones lj(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		lj.Coeff()[i] = fabs(2.5*sin(i + 1));
	}

	NonLinearRegression dtr(&lj, 1000.0, 1.0, 0.1, 0.0, 1.0);

	//ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	for (int i = 0; cfg.Load(ifs); i++) {
		// FD test
		if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
			FAIL()
	}

	ifs.close();

} END_TEST;

TEST("PairPot AddLossGrad test by finite difference") {
	Configuration cfg, cfg1, cfg2;
	double displacement = 1e-5;
	double control_delta = 0.001;
	PairPotentialNonlinear pair(8, 1.6);

	NonLinearRegression dtr(&pair, 1.0, 1.0, 10.0, 0.0, 1.0);

	for (int i = 0; i < pair.CoeffCount(); i++) {
		//pair.Coeff()[i] = 1.0 / pow(2, i);
		pair.Coeff()[i] = 1*sin(i + 1);
	}

	//ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	for (int i = 0; cfg.Load(ifs); i++) {
		// FD test
		if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
			FAIL()
	}

	ifs.close();

} END_TEST;

TEST("MTPR CalcEFS (check forces correctness by using of central differences)") {
	//Configuration cfg, cfg1, cfg2;

	//double diffstep = 1E-4;
	//double relErrMax = 0.0, absErrMax = 0.0;
	//double frob_norm2 = 0.0, forces_prec2 = 0.0;

	//ifstream ifs(PATH+"2comp.cfgs", ios::binary);

	//MLMTPR pot("learned.mtp");

	//for (int n = 0; cfg.load(ifs); n++) {
	//pot.CalcEFS(cfg);

	//for (int i = 0; i < cfg.size(); i++) {
	//for (int l = 0; l < 3; l++) {
	//cfg1 = cfg;
	//cfg2 = cfg;
	//cfg1.pos(i)[l] += diffstep;
	//cfg2.pos(i)[l] -= diffstep;
	//pot.CalcEFS(cfg1);
	//pot.CalcEFS(cfg2);
	//double cdiffForce = (cfg2.energy - cfg1.energy) / (2 * diffstep);
	//double absErr = fabs(cfg.force(i)[l] - cdiffForce);
	//double relErr = fabs((cfg.force(i)[l] - cdiffForce) / cdiffForce);
	//if (relErr > relErrMax) relErrMax = relErr;
	//if (absErr > absErrMax) absErrMax = absErr;
	//}
	//}
	//if (relErrMax > 1E-2) {
	//cout << relErrMax << endl;
	//FAIL()
	//}
	//if (absErrMax > 1E-2) {
	//cout << absErrMax << endl;
	//FAIL()
	//}
	//}

	//ifs.close();

	Configuration cfg;
	double diffstep = 1E-4;

	ifstream ifs(PATH + "2comp.cfgs", ios::binary);
	cfg.Load(ifs);

	MLMTPR pot(PATH+"learned.mtp");

	//cout << pot.CoeffCount() << " " << pot.alpha_count-1 << endl;

	if (!pot.CheckEFSConsistency_debug(cfg, diffstep))
		FAIL();

} END_TEST;

/*TEST("MTPR2 CalcEFS (check forces correctness by using of central differences)") {

	Configuration cfg;
	double diffstep = 1E-4;

	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);
	cfg.Load(ifs);

	MLMTPR2 pot(PATH+"unlearned.mtp");

	//cout << pot.CoeffCount() << " " << (pot.alpha_count-1)*pot.species_count << endl;

	for (int i = 0; i < pot.CoeffCount(); i++)
		pot.Coeff()[i] = 3e-1*rand() / RAND_MAX;

	if (!pot.CheckEFSConsistency_debug(cfg, diffstep))
		FAIL();

} END_TEST;

TEST("MTPR2 loss gradient test") {

	MLMTPR2 mtpr(PATH+"unlearned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++) {
		mtpr.Coeff()[i] = 1e-1*rand() / RAND_MAX;
		//cout << mtpr.Coeff()[i] << " ";
	}
	//cout << endl;

	//cout << mtpr.CoeffCount() << " " << mtpr.alpha_count << endl;

	MTPR_trainer2 nlr(&mtpr, 1, 1, 0);
	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);
	Configuration cfg;
	int fail = 0;

	for (int i = 0; cfg.Load(ifs); i++)
		if (!nlr.CheckLossConsistency_debug(cfg, 1e-6)) {
			fail = 1;
			break;
		}
	ifs.close();

	if (fail > 0) {
		std::cerr << "The gradient is wrong!";
		FAIL()
	}

	//exit(1);

} END_TEST;

TEST("MTPR+CoulQEq CalcEFS (check forces correctness with cdiffs)") {
	Configuration cfg, cfg1, cfg2;
	double diffstep = 1e-6;
	double relErrMax = 0.0, absErrMax = 0.0;
	double r_cut = 5.0;
	double C = 1.0;

	int n_at_types = 2;
	vector<double> coeffs(2 * n_at_types);

	for (int i = 0; i < coeffs.size(); i++)
		coeffs[i] = 0.5*(i + 1);

	Taper taper(r_cut);
	taper.Init();

	CoulQEq qeq(C, r_cut, n_at_types, &taper, &coeffs[0]);

	MLMTPR mtpr(PATH+"learned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 1e-2*rand() / RAND_MAX;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 10000.0;
	double wgt_eqtn_stress = 10.0;
	double opt_weight_charges = 10000.0;
	double wgt_eqtn_constr = 1.0;

	MultTrainer trainer(&mtpr, &qeq, &taper, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_charges, wgt_eqtn_constr);

	ifstream ifs(PATH+"NaCl8at.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs); i++)
		trainer.AddForTrain(cfg);

	trainer.GenerateCoulHess();
	for (int k = 0; k < trainer.train_set.size(); k++)
		trainer.GenerateQEqHess(k);

	Configuration cfg_qeq = cfg;
	Configuration cfg_mtpr = cfg;

	cfg.has_forces(true);
	cfg.has_stresses(true);

	qeq.CalcEFSquick(cfg_qeq, trainer.QEqHess[0]);
	mtpr.CalcEFS(cfg_mtpr);
	trainer.SummarizeEFSQ(cfg_qeq, cfg_mtpr, cfg);

	//cout << endl;

	for (int i = 0; i < cfg.size(); i++) {
		//cout << i << endl;
		for (int l = 0; l < 3; l++) {
			Configuration cfg1_qeq = cfg;
			Configuration cfg2_qeq = cfg;
			cfg1_qeq.pos(i)[l] += diffstep;
			cfg2_qeq.pos(i)[l] -= diffstep;
			qeq.CalcEFSquick(cfg1_qeq, trainer.QEqHess[0]);
			qeq.CalcEFSquick(cfg2_qeq, trainer.QEqHess[0]);
			Configuration cfg1_mtpr = cfg;
			Configuration cfg2_mtpr = cfg;
			cfg1_mtpr.pos(i)[l] += diffstep;
			cfg2_mtpr.pos(i)[l] -= diffstep;
			mtpr.CalcEFS(cfg1_mtpr);
			mtpr.CalcEFS(cfg2_mtpr);
			trainer.SummarizeEFSQ(cfg1_qeq, cfg1_mtpr, cfg1);
			trainer.SummarizeEFSQ(cfg2_qeq, cfg2_mtpr, cfg2);
			double cdiffForce = (cfg2.energy - cfg1.energy) / (2 * diffstep);
			//if (fabs(cfg.force(i)[l]) < 1e-6) continue;
			double absErr = fabs(cfg.force(i)[l] - cdiffForce);
			double relErr = fabs((cfg.force(i)[l] - cdiffForce) / cdiffForce);
			if (relErr > relErrMax) relErrMax = relErr;
			if (absErr > absErrMax) absErrMax = absErr;
			//std::cout << cdiffForce << " " << cfg.force(i)[l] << " " << absErr << std::endl;
		}
	}
	if (relErrMax > 3E-2) {
		cout << relErrMax << endl;
		FAIL()
	}
	if (absErrMax > 3E-2) {
		cout << absErrMax << endl;
		FAIL()
	}

	ifs.close();
} END_TEST;

TEST("MTPR+CoulQEq AddLossGrad test by finite difference") {
	Configuration cfg;
	double diffstep = 1e-4;
	double r_cut = 5.0;
	int n_at_types = 2;

	vector<double> coeffs(2 * n_at_types);

	for (int i = 0; i < coeffs.size(); i++)
		coeffs[i] = 2.5*(i + 1);

	Taper taper(r_cut);
	taper.Init();

	CoulQEq qeq(1.0, r_cut, n_at_types, &taper, &coeffs[0]);

	MLMTPR mtpr(PATH+"learned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 1e-2*rand() / RAND_MAX;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 100.0;
	double wgt_eqtn_stress = 10.0;
	double opt_weight_charges = 100.0;
	double wgt_eqtn_constr = 1.0;

	MultTrainer trainer(&mtpr, &qeq, &taper, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_charges, wgt_eqtn_constr);

	//ifstream ifs("NaCl_small.cfgs", ios::binary);
	ifstream ifs(PATH+"NaClVasp300.cfgs", ios::binary);
	//ifstream ifs("CSH_DFT2.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs); i++)
		trainer.AddForTrain(cfg);

	trainer.GenerateCoulHess();

	vector<double> cdiff(qeq.CoeffCount()+mtpr.CoeffCount());
	FillWithZero(cdiff);

	//cout << endl;
	//cout << qeq.CoeffCount() << " " << mtpr.CoeffCount() << endl;

	trainer.CalcObjectiveFunctionGrad();

	for (int j = 0; j < mtpr.CoeffCount(); j++) {
		mtpr.Coeff()[j] += diffstep;
		double lossp = trainer.ObjectiveFunction();
		mtpr.Coeff()[j] -= 2 * diffstep;
		double lossm = trainer.ObjectiveFunction();
		mtpr.Coeff()[j] += diffstep;
		cdiff[j] += (lossp - lossm) / (2 * diffstep);
	}

	for (int j = 0; j < qeq.CoeffCount(); j++) {
		qeq.Coeff()[j] += diffstep;
		double lossp = trainer.ObjectiveFunction();
		qeq.Coeff()[j] -= 2 * diffstep;
		double lossm = trainer.ObjectiveFunction();
		qeq.Coeff()[j] += diffstep;
		cdiff[j+mtpr.CoeffCount()] += (lossp - lossm) / (2 * diffstep);
	}

	double control_delta = 1E-3;

	if (!trainer.CheckLossGrad(control_delta, cdiff))
		FAIL()

	ifs.close();
} END_TEST;*/

TEST("Linesearch") {
	{
		// minimize f(x) = -x - x^2 + x^3
		const double alpha_0[] = { 0.12,0.234,0.33,0.44,0.55,0.66,0.77,0.88,2.1,1000,1e30 };
		const int alpha_0_count = sizeof(alpha_0) / sizeof(double);
		for (int i = 0; i < alpha_0_count; i++) {
			Linesearch ls;
			ls.Reset(alpha_0[i]);
			int count;
			for (count = 0; count < 30; count++) {
				double x = ls.x();
				ls.Iterate(-x - x*x + x*x*x, -1 - 2 * x + 3 * x*x);
				if (std::abs(x - 1) < 1e-5) break;
			}
			if (count == 30) ERROR("no convergence after 30 iterations, alpha_0 = " + to_string(alpha_0[i]));
		}
	}
	{
		class Linesearch_mod : public Linesearch {
		public:
			void SetX(double _x) { curr_x = _x; }
		};

		// minimize f(x) = x/2-sin(x)
		Linesearch_mod ls;
		ls.Reset(10.0);
		{
			double x = ls.x();
			ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
		}
		{
			double x = ls.x();
			ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
		}
		ls.SetX(6.28);

		int count;
		for (count = 0; count < 30; count++) {
			double x = ls.x();
			ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
			if (std::abs(x - 1.04719755) < 1e-5) break;
		}
		if (count == 30) ERROR("no convergence after 30 iterations on Linesearch_mod");
	}
	{
		// minimize f(x) = x/2-sin(x)
		const double alpha_0[] = { 6.28,0.12,3.14, 1000, 1e30 };
		const int alpha_0_count = sizeof(alpha_0) / sizeof(double);
		for (int i = 0; i < alpha_0_count; i++) {
			Linesearch ls;
			ls.Reset(alpha_0[i]);
			int count;
			for (count = 0; count < 300; count++) {
				double x = ls.x();
				ls.Iterate(x/2 - sin(x), 0.5 - cos(x));
				if (std::abs(x - 1.04719755) < 1e-5) break;
			}
			if (count == 300) ERROR("no convergence after 300 iterations, alpha_0 = " + to_string(alpha_0[i]));
		}
	}
} END_TEST;

TEST("checks minimal distance in configurations") {
	int count = 0;
	int status = 0;

	// read the database
	std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
	Configuration cfg;
	for (count = 0; (status = cfg.Load(ifs)); count++) {
		Neighborhoods nbhs(cfg, 6.0);
		double min_dist = 123e123;
		for (int i = 0; i < (int)nbhs.size(); i++)
			for (int j = 0; j < (int)nbhs[i].dists.size(); j++)
				min_dist = __min(min_dist, nbhs[i].dists[j]);
		double cfg_min_dist = cfg.MinDist();
		if (fabs(min_dist - cfg_min_dist) > 1e-6)
			FAIL()
	}
} END_TEST;

TEST("Supercell rotation test") {
	ifstream ifs(PATH+"4relax.cfg", ios::binary);
	Configuration cfg;
	cfg.Load(ifs);
	Matrix3 rot_mat(-0.686602, -0.340459, -0.642391, 
					-0.126594, -0.814094, 0.566767,
					-0.715927, 0.470466, 0.515858	);
	MTP mtp(PATH+"MTP4relax.mtp");
	cfg.Deform(rot_mat);
	mtp.CalcEFS(cfg);
	cfg.CorrectSupercell();
	Configuration cfg2(cfg);
	mtp.CalcEFS(cfg2);

	for (int i=0; i<cfg.size(); i++)
		if ((cfg2.force(i) - cfg.force(i)).Norm() > 1.0e-7)
		//{
		//	cout << cfg2.force(i, 0) << ' ' << cfg2.force(i, 1) << ' ' << cfg2.force(i, 2) << endl;
		//	cout << cfg.force(i, 0) << ' ' << cfg.force(i, 1) << ' ' << cfg.force(i, 2) << endl;
		//}
				FAIL();

	if ((cfg2.stresses - cfg.stresses).NormFrobeniusSq() > 1.0e-7)
		FAIL();
} END_TEST;

TEST("Relaxation driver test") {
	ifstream ifs(PATH+"4relax.cfg", ios::binary);
	Configuration cfg;
	cfg.Load(ifs);
	MTP mtp(PATH+"MTP4relax.mtp");
	Relaxation rlx(&mtp, cfg);
	rlx.tol_force = 0.0001;
	rlx.tol_stress = 0.001;
	rlx.itr_limit = 500;
	//rlx.p_log = &cout;

	rlx.Run();
	cfg = rlx.cfg;

	//cout<< cfg.energy << ' ' << rlx.step;
	double mf = 0;
	for (int i=0; i<cfg.size(); i++)
		mf = __max(mf, cfg.force(i).Norm());

	if (fabs(cfg.energy + 25.339994) > 0.0001)
		ERROR("Energy of the relaxed configuration is " + to_string(cfg.energy) +
		" but should be 25.34.");
	if (mf > rlx.tol_force)
		ERROR("Max force is "+to_string(mf)+" (too big)");
	if (cfg.stresses.MaxAbs() > rlx.tol_stress)
		ERROR("Max stress is " + to_string(cfg.stresses.MaxAbs()) + " (too big)");
} END_TEST;

TEST("Robustness of relaxation with changing potential") {
	ifstream ifs(PATH+"4relax.cfg", ios::binary);
	Configuration cfg;
	cfg.Load(ifs);
	MTP mtp(PATH+"MTP4relax.mtp");
	
	class CrazyPotential : public AnyPotential
	{
	public: 
		int call_cntr = 0;
		AnyPotential* ppot;
		CrazyPotential(AnyPotential* _pot) : ppot(_pot) {};
		void CalcEFS(Configuration& cfg) override
		{
			ppot->CalcEFS(cfg);
			if(call_cntr<100) cfg.energy += 100 * (call_cntr++ % 7);
		}
	};
	CrazyPotential crazy_pot(&mtp);

	Relaxation rlx(&crazy_pot, cfg);
	rlx.tol_force = 0.0001;
	rlx.tol_stress = 0.001;
	rlx.itr_limit = 500;

	rlx.Run();
	cfg = rlx.cfg;

	while (cfg.energy > 100)
		cfg.energy -= 100;
	//cout<< cfg.energy << ' ' << rlx.step;

	double mf = 0;
	for (int i=0; i<cfg.size(); i++)
		mf = __max(mf, cfg.force(i).Norm());
	if (fabs(cfg.energy + 25.339994) > 0.0001)
		ERROR("Energy of the relaxed configuration is " + to_string(cfg.energy) +
			" but should be 25.34.");
	if (mf > rlx.tol_force)
		ERROR("Max force is " + to_string(mf) + " (too big)");
	if (cfg.stresses.MaxAbs() > rlx.tol_stress)
		ERROR("Max stress is " + to_string(cfg.stresses.MaxAbs()) + " (too big)");
} END_TEST;

TEST("Another MTP learning + ErrorMonitor") {
	MTP mtp(PATH+"MTP2_61.params.mtp");
	LinearRegression regr(&mtp, 500, 1, 10);
	std::ifstream ifs(PATH+"Si1.proc", ios::binary);
	Configuration cfg;
	ErrorMonitor errmon;

	vector<Configuration> training_set;
	for (int count = 0; cfg.Load(ifs); count++)
		training_set.push_back(cfg);
	regr.Train(training_set);
	ifs.close();

	// the MTP that is already trained
	MTP mtp_ready(PATH+"MTP_testsuite_test.params.mtp");
	ifs.open(PATH+"Si2.proc", ios::binary);
	for (int count = 0; cfg.Load(ifs); count++) {
		Configuration cfg2(cfg);
		mtp.CalcEFS(cfg);
		errmon.collect(cfg2, cfg);
	}
	ifs.close();

#ifdef GEN_TESTS
	std::ofstream ofs(PATH + "test_errlearning.txt");
	ofs << errmon.cfg_cntr << std::endl;
	ofs << errmon.ene_aveabs() << std::endl;
	ofs << errmon.ene_averel() << std::endl;
	ofs << errmon.epa_aveabs() << std::endl;
	ofs << errmon.epa_averel() << std::endl;
	ofs << errmon.frc_aveabs() << std::endl;
	ofs << errmon.frc_rmsrel() << std::endl;
	ofs << errmon.str_rmsrel() << std::endl;
	ofs.close();
#endif

	ifs.open(PATH + "test_errlearning.txt");
	double foo;
	int bar;
	ifs >> bar;
	if (bar != errmon.cfg_cntr)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.ene_aveabs()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.ene_averel()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.epa_aveabs()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.epa_averel()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.frc_aveabs()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.frc_rmsrel()) > 1.0e-6)
		FAIL()
		ifs >> foo;
	if (fabs(foo - errmon.str_rmsrel()) > 1.0e-6)
		FAIL()
		ifs.close();
} END_TEST;

TEST("MTP fitting on insufficient data") {
	MTP mtp(PATH+"MTP100_3.mtp");
	auto cfgs = LoadCfgs("configurations/reading_test.cfg");
	LinearRegression lr(&mtp);
	ErrorMonitor errmon;

	lr.AddToSLAE(cfgs[2]);
	lr.Train();
	Configuration cfg(cfgs[2]);
	mtp.CalcEFS(cfg);
	errmon.collect(cfgs[2], cfg);
	if (errmon.epa_all.max.delta > 10e-10 ||
		errmon.frc_all.max.delta > 10e-10 ||
		errmon.vir_all.max.delta > 10e-10)
		FAIL();
	//errmon.report(&cout);

	for (int i=0; i<5; i++)
	{
		auto ccfg(cfgs[i]);
		lr.AddToSLAE(ccfg);
		lr.Train();
		cfg = ccfg;
		mtp.CalcEFS(cfg);
		errmon.collect(ccfg, cfg);
		//errmon.report(&cout);
		if (errmon.epa_all.max.delta > 10e-8 ||
			errmon.frc_all.max.delta > 10e-7 ||
			errmon.vir_all.max.delta > 10e-5)
			FAIL();
	}
} END_TEST;

TEST("MaxVol initialization and selection") {
	MaxVol maxvol(10, 1.0e-7);
	maxvol.B.resize(10, 10);

	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++)
			maxvol.B(i, j) = i + j;
	for (int i = 0; i < 10; i++) {
		double norm_vect_row_sq = 0.0;
		for (int j = 0; j < 10; j++)
			norm_vect_row_sq += pow(maxvol.B(i, j), 2);
		for (int j = 0; j < 10; j++)
			maxvol.B(i, j) /= sqrt(norm_vect_row_sq);
	}

	maxvol.MaximizeVol(1.0, 5);
	if (maxvol.InitedRowsCount() != 2)
		FAIL()

		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				maxvol.B(i, j) = pow(i + 1, j);
	for (int i = 0; i < 10; i++) {
		double norm_vect_row_sq = 0.0;
		for (int j = 0; j < 10; j++)
			norm_vect_row_sq += pow(maxvol.B(i, j), 2);
		for (int j = 0; j < 10; j++)
			maxvol.B(i, j) /= sqrt(norm_vect_row_sq);
	}

	maxvol.MaximizeVol(1.0, 10);
	if (maxvol.InitedRowsCount() != maxvol.n - 1)
		FAIL();

	double fingerprint = 0.0;
	for (int i = 0; i < 10; i++)
		fingerprint += 10 * maxvol.A(i, i);
	if (fabs(fingerprint - 10.6768) > 1.0e-4)
		FAIL()

		for (int i = 0; i < 10; i++)
			for (int j = 0; j < 10; j++)
				maxvol.B(i, j) = (i == j) ? 1 : 0;

	maxvol.MaximizeVol(1.0, 10);

	vector<double> rsum(10);
	vector<double> csum(10);
	for (int i = 0; i < 10; i++)
		for (int j = 0; j < 10; j++) {
			rsum[i] += maxvol.A(i, j);
			csum[j] += maxvol.A(i, j);
		}
	for (int i = 0; i < 10; i++)
		if (fabs(rsum.at(i) - 1.0) > 1e-6 || fabs(csum.at(i) - 1.0) > 1e-6)
			FAIL()
} END_TEST;

TEST("MaxVol evaluation") {
	MTP mtp("mlips/unfitted.mtp");
	VoidPotential pot;
	MaxvolSelection mlip(&mtp);
	
	mlip.MV_nbh_cmpnts_weight = 0.25;
	mlip.MV_ene_cmpnts_weight = 0.25;
	mlip.MV_frc_cmpnts_weight = 0.25;
	mlip.MV_str_cmpnts_weight = 0.25;
	std::vector<Configuration> cfgs;
	Configuration cfg;
	std::ifstream ifs(PATH+"Ti4MVtest.proc3", ios::binary);
	while (cfg.Load(ifs))
		mlip.Select(cfg);
	ifs.close();

	ifs.open("configurations/reading_test.cfg", ios::binary);
	while (cfg.Load(ifs))
		cfgs.push_back(cfg);
	ifs.close();

#ifdef GEN_TESTS
	std::ofstream ofs(PATH+"test_MV_evaluation.txt");
	ofs.precision(15);

	mlip.MV_nbh_cmpnts_weight = 1.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < cfgs.size(); i++)
		ofs << mlip.Grade(cfgs[i]) << std::endl;

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 1.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < cfgs.size(); i++)
		ofs << mlip.Grade(cfgs[i]) << std::endl;

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 1.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < cfgs.size(); i++)
		ofs << mlip.Grade(cfgs[i]) << std::endl;

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 1.0;
	for (int i = 0; i < cfgs.size(); i++)
		ofs << mlip.Grade(cfgs[i]) << std::endl;

	ofs.close();
#endif

	ifs.open(PATH+"test_MV_evaluation.txt", ios::binary);

	mlip.MV_nbh_cmpnts_weight = 1.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < (int)cfgs.size(); i++) {
		double grade;
		ifs >> grade;
		if (fabs(grade - mlip.Grade(cfgs[i])) > 1.0e-7)
			FAIL()
	}

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 1.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < (int)cfgs.size(); i++) {
		double grade;
		ifs >> grade;
		if (fabs(grade - mlip.Grade(cfgs[i])) > 1.0e-7)
			FAIL()
	}

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 1.0;
	mlip.MV_str_cmpnts_weight = 0.0;
	for (int i = 0; i < (int)cfgs.size(); i++) {
		double grade;
		ifs >> grade;
		if (fabs(grade - mlip.Grade(cfgs[i])) > 1.0e-7)
			FAIL()
	}

	mlip.MV_nbh_cmpnts_weight = 0.0;
	mlip.MV_ene_cmpnts_weight = 0.0;
	mlip.MV_frc_cmpnts_weight = 0.0;
	mlip.MV_str_cmpnts_weight = 1.0;
	for (int i = 0; i < (int)cfgs.size(); i++) {
		double grade;
		ifs >> grade;
		if (fabs(grade - mlip.Grade(cfgs[i])) > 1.0e-7)
			FAIL()
	}

	ifs.close();
} END_TEST;

TEST("Configurations motion while MaxVol selection") {
	VoidPotential pot;
	double MV_trshld = 1.1;

	MTP mtp("mlips/unfitted.mtp");
	LinearRegression linreg(&mtp);
	MaxvolSelection MLIP(&mtp, 1.0e-7, MV_trshld, 1+1.0e-7);

	std::ifstream ifs(PATH+"Ti4MVtest.proc3", ios::binary);
	Configuration cfg;

	int start_id = cfg.id();

	std::vector<int> swap_cnt_tracker;

	std::vector<Configuration>& selected = MLIP.selected_cfgs;
	for (int count = 0; cfg.Load(ifs); count++)
		swap_cnt_tracker.push_back(MLIP.Select(cfg));

	ifs.close();

#ifdef GEN_TESTS
	{
		std::ofstream ofs(PATH+"test_MV_swaps.txt");
		for (int i = 0; i < swap_cnt_tracker.size(); i++)
			ofs << swap_cnt_tracker[i] << '\n';
		ofs.close();
		ofs.open(PATH+"test_MV_selected.txt");
		for (Configuration& p_cfg : selected)
			ofs << p_cfg.id() - start_id << '\n';
		ofs.close();
	}
#endif

	ifs.open(PATH+"test_MV_swaps.txt", ios::binary);
	for (int i = 0; i < (int)swap_cnt_tracker.size(); i++) {
		int foo;
		ifs >> foo;
		if (foo != swap_cnt_tracker[i])
			FAIL()
	}
	ifs.close();
	ifs.open(PATH+"test_MV_selected.txt", ios::binary);
	for (Configuration& cfgg : selected) {
		int foo;
		ifs >> foo;
		if (cfgg.id() - start_id != foo)
			FAIL()
	}
	ifs.close();
} END_TEST;

TEST("Maxvol selection UniqueCfgCnt() and swap_limit tests") {
	MTP mtp(PATH+"MTP61.mtp");
	MaxvolSelection mv(&mtp, 0.00001, 1.00001, 1.00001, 1, 0, 0, 0);
	if (mv.UniqueCfgCount() != 0)
		FAIL();
	mv.LoadSelected(PATH+"Li.cfg");
	if (mv.UniqueCfgCount() != 1)
		FAIL();
	ifstream ifs(PATH + "B.cfgs", ios::binary);
	Configuration cfg;
	while (cfg.Load(ifs))
	{
		int prev = mv.UniqueCfgCount();
		mv.Select(cfg);
		if (mv.UniqueCfgCount() - prev > 1)
			FAIL();
	}
} END_TEST;

TEST("Multiple selection test#1") {
	MTP mtp(PATH+"MTP9.mtp");

	ifstream ifs(PATH + "B.cfgs", ios::binary);
	vector<Configuration> cfgs;
	int cntr = 0;
	for (Configuration cfg; cfg.Load(ifs) && cntr<20; cntr++)
		cfgs.push_back(cfg);

	MaxvolSelection mv_single(&mtp, 0.0000005, 1.0000005, 1.0000005, 0, 1, 0, 0);
	bool swaps = true;
	while (swaps)
	{
		swaps = false;
		for (Configuration& cfg : cfgs)
			swaps |= (mv_single.Select(cfg)>0);
	}

	MaxvolSelection mv_mult(&mtp, 0.0000005, 1.0000005, 1.0000005, 0, 1, 0, 0);
	for (Configuration& cfg : cfgs)
		mv_mult.AddForSelection(cfg);
	mv_mult.Select();

	if (mv_single.UniqueCfgCount() != mv_mult.UniqueCfgCount())
		FAIL();

	multiset<int> s1, s2;

	for (Configuration& cfg : mv_single.selected_cfgs)
		s1.insert(stoi(cfg.features["number"]));
	for (Configuration& cfg2 : mv_mult.selected_cfgs)
		s2.insert(stoi(cfg2.features["number"]));

	if (s1 != s2)
		FAIL();

} END_TEST;

TEST("Multiple selection test#2") {
	MTP mtp(PATH+"MTP9.mtp");

	ifstream ifs(PATH + "B.cfgs", ios::binary);
	vector<Configuration> cfgs;
	int cntr = 0;
	for (Configuration cfg; cfg.Load(ifs) && cntr<20; cntr++)
		cfgs.push_back(cfg);

	MaxvolSelection mv_single(&mtp, 0.000001, 1.000001, 1.000001, 1, 0, 0, 0);
	for (Configuration& cfg : cfgs)
		mv_single.Select(cfg);

	MaxvolSelection mv_mult(&mtp, 0.000001, 1.000001, 1.000001, 1, 0, 0, 0);
	for (Configuration& cfg : cfgs)
	{
		mv_mult.AddForSelection(cfg);
		mv_mult.Select();
	}

	if (mv_single.UniqueCfgCount() != mv_mult.UniqueCfgCount())
		FAIL();

	multiset<int> s1, s2;

	for (Configuration& cfg : mv_single.selected_cfgs)
		s1.insert(stoi(cfg.features["number"]));
	for (Configuration& cfg2 : mv_mult.selected_cfgs)
		s2.insert(stoi(cfg2.features["number"]));

	if (s1 != s2)
		FAIL();

	s1.clear(); s2.clear();

	for (Configuration& cfg : cfgs)
		mv_mult.AddForSelection(cfg);
	mv_mult.Select();

	bool swaps = true;
	while (swaps)
	{
		swaps = false;
		for (Configuration& cfg : cfgs)
			swaps |= (mv_single.Select(cfg)>0);
	}

	for (Configuration& cfg : mv_single.selected_cfgs)
		s1.insert(stoi(cfg.features["number"]));
	for (Configuration& cfg2 : mv_mult.selected_cfgs)
		s2.insert(stoi(cfg2.features["number"]));

	if (s1 != s2)
		FAIL();

} END_TEST;

TEST("MaxVol Saving/Loading selection test#1") {
	MTP mtp(PATH+"MTP61.mtp");

	ifstream ifs(PATH + "B.cfgs", ios::binary);
	vector<Configuration> cfgs;
	int cntr = 0;
	for (Configuration cfg; cfg.Load(ifs) && cntr<2; cntr++)
		cfgs.push_back(cfg);

	MaxvolSelection mv(&mtp, 0.0000001, 1.0000001, 1.0000001, 1, 0, 0, 0);
	for (Configuration& cfg : cfgs)
		mv.AddForSelection(cfg);
	mv.Select();

	MTP foo(PATH+"MTP9.mtp");
	MaxvolSelection mv_check(&foo, 0.0000001, 1.0000001, 1.0000001, 16, 45, 123, 543);
	try
	{
		mv.Save(PATH+"temp/maxvol_state.mvs");
		mv_check.Load(PATH+"temp/maxvol_state.mvs");
	}
	catch (...)
		FAIL();

	for (int i=0; i<mv.selected_cfgs.size(); i++)
		if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
			FAIL();

	if (mv.UniqueCfgCount() != mv_check.UniqueCfgCount())
		FAIL();

	if (mv.maxvol.A.size1 != mv_check.maxvol.A.size1 ||
		mv.maxvol.A.size1 != mv_check.maxvol.A.size2)
		FAIL();

	for (int i=0; i<mv.maxvol.A.size1*mv.maxvol.A.size2; i++)
		if (*(&mv.maxvol.A(0, 0) + i) != *(&mv_check.maxvol.A(0, 0) + i))
			FAIL();

	for (Configuration cfg; cfg.Load(ifs); cntr++)
	{
		mv.AddForSelection(cfg);
		mv_check.AddForSelection(cfg);
	}
	mv.Select();
	mv_check.Select();

	for (int i=0; i<mv.selected_cfgs.size(); i++)
		if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
			FAIL();

} END_TEST;

//TEST("MaxVol Saving/Loading selection test#2") {
//	MLMTPR mtp(PATH+"PdAg_20g_str.mtp");
//
//	ifstream ifs(PATH+"20g_Selected.cfgs", ios::binary);
//	vector<Configuration> cfgs;
//	int cntr = 0;
//	for (Configuration cfg; cfg.Load(ifs) && cntr<2; cntr++)
//		cfgs.push_back(cfg);
//
//	MaxvolSelection mv(&mtp, 0.0000001, 0.1, 0.0000001, 0, 1, 0, 0);
//	for (Configuration& cfg : cfgs)
//		mv.AddForSelection(cfg);
//	mv.Select();
//
//	MLMTPR foo(PATH+"PdAg_20g_str.mtp");
//	MaxvolSelection mv_check(&foo, 0.0000001, 0.0000001, 0.0000001, 16, 45, 123, 543);
//	try
//	{
//		mv.Save("temp/maxvol_state.mvs");
//		mv_check.Load("temp/maxvol_state.mvs");
//	}
//	catch (...)
//		FAIL();
//
//	for (int i=0; i<mv.selected_cfgs.size(); i++)
//		if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
//			FAIL();
//
//	if (mv.UniqueCfgCount() != mv_check.UniqueCfgCount())
//		FAIL();
//
//	if (mv.maxvol.A.size1 != mv_check.maxvol.A.size1 ||
//		mv.maxvol.A.size1 != mv_check.maxvol.A.size2)
//		FAIL();
//
//	for (int i=0; i<mv.maxvol.A.size1*mv.maxvol.A.size2; i++)
//		if (*(&mv.maxvol.A(0, 0) + i) != *(&mv_check.maxvol.A(0, 0) + i))
//			FAIL();
//
//	for (Configuration cfg; cfg.Load(ifs); cntr++)
//	{
//		mv.AddForSelection(cfg);
//		mv_check.AddForSelection(cfg);
//	}
//	mv.Select();
//	mv_check.Select();
//
//	for (int i=0; i<mv.selected_cfgs.size(); i++)
//		if (mv.selected_cfgs[i] != mv_check.selected_cfgs[i])
//			FAIL();
//
//} END_TEST;

TEST("MaxVol learning on the fly") {
	VoidPotential pot;
	double MV_trshld = 1+1.0e-7;

	MTP mtp("mlips/unfitted.mtp");
	LinearRegression regr(&mtp, 500, 1, 10);
	MaxvolSelection al(&mtp, 1.0e-7, MV_trshld, MV_trshld, 0, 1, 0, 0);
	LOTF MLIP(&pot, &al, &regr, true);

	std::ifstream ifs(PATH+"Ti4MVtest.proc3", ios::binary);
	Configuration cfg;
	Configuration cfg4valid;
	ErrorMonitor err_mon;

	for (int count = 0; count < 10; count++) {
		if (!cfg.Load(ifs))
			FAIL();

		al.Select(cfg);
	}

	regr.Train(al.selected_cfgs);

	for (int count = 10; cfg.Load(ifs); count++) {
		cfg4valid = cfg;
		MLIP.CalcEFS(cfg);
		err_mon.collect(cfg4valid, cfg);
	}

	ifs.close();

#ifdef GEN_TESTS
	{
		std::ofstream ofs(PATH+"learn_on_the_fly.txt");
		ofs << err_mon.ene_aveabs() << '\n'
			<< err_mon.frc_rmsrel() << '\n'
			<< err_mon.str_rmsrel() << '\n';
		ofs.close();
	}
#endif

	ifs.open(PATH+"learn_on_the_fly.txt");

	double ref;
	ifs >> ref;
	if (abs(err_mon.ene_aveabs() - ref) > 2e-5)
		FAIL()
	ifs >> ref;
	if (abs(err_mon.frc_rmsrel() - ref) > 2e-5)
		FAIL()
	ifs >> ref;
	if (abs(err_mon.str_rmsrel() - ref) > 2e-5)
		FAIL()
	ifs >> ref;

	ifs.close();
} END_TEST;

TEST("check radial basis derivatives with finite differences") {
	double diffstep = 1e-4;
	int basis_function_count = 10;
	double r_min = 1.6;
	double r_cut = 5.2;
	double r = 3.5;

	RadialBasis_Shapeev bas(r_min, r_cut, basis_function_count);
	bas.RB_Calc(r);

	RadialBasis_Shapeev bas_p(r_min, r_cut, basis_function_count);
	bas_p.RB_Calc(r + diffstep);

	RadialBasis_Shapeev bas_m(r_min, r_cut, basis_function_count);
	bas_m.RB_Calc(r - diffstep);
	for (int i = 0; i < basis_function_count; i++) {
		double cdiff = (bas_p.rb_vals[i] - bas_m.rb_vals[i]) / (2 * diffstep);
		if (fabs((cdiff - bas.rb_ders[i]) / cdiff) > 2e-4) {
			std::cout << fabs((cdiff - bas.rb_ders[i]) / cdiff) << std::endl;
			FAIL()
		}
		//std::cout << cdiff << " " << bas.ders[i] << std::endl;
	}
} END_TEST;

/*#ifndef MLIP_NOEWALD
TEST("Check Ewald summation") {
	vector<string> fnm(5);
	vector<double> madelung(5);

	fnm[0] = "ZnSblendetwo.cfgs";
	fnm[1] = "CsCl.cfgs";
	fnm[2] = "NaCl8ideal.cfgs";
	fnm[3] = "CaF2.cfgs";
	fnm[4] = "ZnSwurtizetwo.cfgs";

	madelung[0] = 1.638;
	madelung[1] = 1.763;
	madelung[2] = 1.748;
	madelung[3] = 1.680;
	madelung[4] = 1.641;

	double relErrMax = 0;

	for (int k = 0; k < fnm.size(); k++) {
		//cout << k << endl;

		ifstream ifs(PATH+fnm[k], ios::binary);
		Configuration cfg;
		cfg.Load(ifs);

		double coulomb_constant = 14.4; //for eV-angstroms

		//double angstrom_to_bohr = 1.88973;
		//double coulomb_constant = 1; //for hartree-bohrs

		//for (int i = 0; i < cfg.size(); i++)
		//	for (int l = 0; l < 3; l++)
		//		cfg.pos(i)[l] *= angstrom_to_bohr;

		//for (int a = 0; a < 3; a++)
		//	for (int l = 0; l < 3; l++)
		//		cfg.lattice[a][l] *= angstrom_to_bohr;

		SelfEnergy se(PATH+"unlearned2.qeq");

		QEq qeq(&se);

		vector<double> xred(3*cfg.size());
		vector<double> ew_forces_red(3*cfg.size());
		vector<double> ew_stress(6);
		vector<double> charges(cfg.size());
		vector<double> gmet(9);
		vector<double> rmet(9);
		vector<double> gprimd(9);
		vector<double> rprimd(9);
		double ucvol;

		qeq.calc_lattice_params_for_ewald(cfg, &xred[0], &gprimd[0], &rprimd[0], &gmet[0], &rmet[0], ucvol);

		for (int i = 0; i < cfg.size(); i++) 
			charges[i] = cfg.charges(i);

		//Vector3 * pos;

		//pos = new Vector3 [cfg.size()];

		//for (int i = 0; i < cfg.size(); i++) {
		//	pos[i] = cfg.pos(i) * cfg.lattice;
		//	cout << pos[i][0] << " " << pos[i][1] << " " << pos[i][2] << endl;
		//}

		double * QEq_en_ch_ders;
		vector<double> Ew_forces_ch_ders(3*cfg.size()*cfg.size());
		vector<double> Ew_stress_ch_ders(6*cfg.size());
		vector<double> QEqHessk(cfg.size()*cfg.size());
		double eew;

		FillWithZero(QEqHessk);

		int key = 0; //key = 0 ewald energy only, key = 3 - ewald energy with forces, stresses, 
			     //key = 4 - ewald energy with forces and grads
		int natom = cfg.size();
		ewald_sum_(&natom, &ucvol, &xred[0], &rmet[0], &gmet[0], &rprimd[0], &gprimd[0], &eew, 
		QEq_en_ch_ders, &ew_forces_red[0], &ew_stress[0], &Ew_forces_ch_ders[0], &Ew_stress_ch_ders[0],
		&QEqHessk[0], &charges[0], &key);

		double my_madelung = 2 * eew * cfg.MinDist() / (coulomb_constant * charges[0] * charges[cfg.size()/2] * cfg.size());
		//cout << "my_madelung = " << my_madelung << ", eew = " << eew << endl;

		if (fabs(my_madelung - madelung[k])/madelung[k] > relErrMax) relErrMax = fabs(my_madelung - madelung[k])/madelung[k];

	}

	if (relErrMax > 1E-3) {
		cout << relErrMax << endl;
		FAIL()
	}

} END_TEST

TEST("Check Ewald ders w.r.t. charges") {

	//ifstream ifs("NaCl8at.cfgs", ios::binary);
	ifstream ifs(PATH+"LiTi2O4_01_train.cfgs", ios::binary);
	Configuration cfg;
	cfg.Load(ifs);

	SelfEnergy se(PATH+"unlearned3.qeq");

	QEq qeq(&se);

	vector<double> xred(3*cfg.size());
	vector<double> ew_forces_red(3*cfg.size());
	vector<double> ew_stress(6);
	vector<double> charges(cfg.size());
	vector<double> gmet(9);
	vector<double> rmet(9);
	vector<double> gprimd(9);
	vector<double> rprimd(9);
	double ucvol;

	qeq.calc_lattice_params_for_ewald(cfg, &xred[0], &gprimd[0], &rprimd[0], &gmet[0], &rmet[0], ucvol);

	for (int i = 0; i < cfg.size(); i++) {
		//charges[i] = cfg.charges(i);
		charges[i] = -1*sin(2*i-cfg.size()+1);
	}

	double * QEq_en_ch_ders;
	vector<double> Ew_forces_ch_ders(3*cfg.size()*cfg.size());
	vector<double> Ew_stress_ch_ders(6*cfg.size());
	vector<double> QEqHessk(cfg.size()*cfg.size());
	double eew;

	FillWithZero(QEqHessk);

	int key = 4; //key = 4 - ewald energy, forces and stresses grads
	int natom = cfg.size();
	ewald_sum_(&natom, &ucvol, &xred[0], &rmet[0], &gmet[0], &rprimd[0], &gprimd[0], &eew, 
	QEq_en_ch_ders, &ew_forces_red[0], &ew_stress[0], &Ew_forces_ch_ders[0], &Ew_stress_ch_ders[0],
	&QEqHessk[0], &charges[0], &key);

	key = 3; //key = 3 - ewald energy, forces, stresses
	double diffstep = 1E-4;
	double absErrForces = 0;
	double absErrStresses = 0;
	vector<double> ew_stress_p(6);
	vector<double> ew_stress_m(6);
	vector<double> ew_forces_red_p(3*cfg.size());
	vector<double> ew_forces_red_m(3*cfg.size());
	for (int i = 0; i < cfg.size(); i++) {
		charges[i] += diffstep;
		ewald_sum_(&natom, &ucvol, &xred[0], &rmet[0], &gmet[0], &rprimd[0], &gprimd[0], &eew, 
		QEq_en_ch_ders, &ew_forces_red_p[0], &ew_stress_p[0], &Ew_forces_ch_ders[0], &Ew_stress_ch_ders[0],
		&QEqHessk[0], &charges[0], &key);
		charges[i] -= 2*diffstep;
		ewald_sum_(&natom, &ucvol, &xred[0], &rmet[0], &gmet[0], &rprimd[0], &gprimd[0], &eew, 
		QEq_en_ch_ders, &ew_forces_red_m[0], &ew_stress_m[0], &Ew_forces_ch_ders[0], &Ew_stress_ch_ders[0],
		&QEqHessk[0], &charges[0], &key);
		//cout << "forces ders:" << endl;
		for (int j = 0; j < cfg.size(); j++) {
			for (int l = 0; l < 3; l++) {
				double force_ch_der = (ew_forces_red_p[j*3+l] - ew_forces_red_m[j*3+l]) / (2*diffstep);
				double absErr = fabs(Ew_forces_ch_ders[j*natom*3+i*3+l] - force_ch_der);
				if (absErr > absErrForces) absErrForces = absErr;
				//cout << Ew_forces_ch_ders[j*natom*3+i*3+l] << " " << force_ch_der << endl;
			}
		}
		//cout << "stresses ders:" << endl;
		for (int l = 0; l < 6; l++) {
			double stress_ch_der = (ew_stress_p[l] - ew_stress_m[l]) / (2*diffstep);
			double absErr = fabs(Ew_stress_ch_ders[l*natom+i]-stress_ch_der);
			if (absErr > absErrStresses) absErrStresses = absErr;
			//cout << Ew_stress_ch_ders[l*natom+i] << " " << stress_ch_der << " ";
			//cout << fabs((Ew_stress_ch_ders[l*natom+i]-stress_ch_der) / stress_ch_der) << endl;
		}
		charges[i] += diffstep;
	}

	if (absErrForces > 1E-4) {
		cout << absErrForces << endl;
		FAIL()
	}

	if (absErrStresses > 1E-4) {
		cout << absErrStresses << endl;
		FAIL()
	}

} END_TEST

TEST("QEq OptimizeChargesGrad check with finite differences") {
	Configuration cfg;
	double absErrMax = 0.0;

	MLMTPR mtpr(PATH+"unlearned3.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 0;

	SelfEnergy se(PATH+"unlearned3.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++) {
		if ((i+1) % 3 == 0) se.Coeff()[i] = sqrt(3.5*(i+1));
		else se.Coeff()[i] = 3.5*(i+1);
		//cout << se.Coeff()[i] << endl; 
	}

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 1.0;
	double wgt_eqtn_stress = 1.0;
	double opt_weight_charges = 1.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges); 

	ifstream ifs1(PATH+"LiTi2O4_01_train.cfgs", ios::binary);
	//ifstream ifs1("NaCl_small.cfgs", ios::binary);
	//cfg.Load(ifs1);
	//ifstream ifs1("NaClVasp300.cfgs", ios::binary);
	//for (int ii = 0; cfg_dummy.Load(ifs1); ii++) {
	//	if (ii == 90) cfg = cfg_dummy;
	//}
	//ifstream ifs1("CSHfirst.cfgs", ios::binary);
	vector<Configuration> train_set;
	cfg.Load(ifs1);
	train_set.push_back(cfg);
	ifs1.close();

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mtpr_plus_qeq.GenerateQEqHess(train_set[k], k);

	Array2D charges_grad;
	charges_grad.resize(cfg.size(), se.CoeffCount());
	charges_grad.set(0);

	Array2D charges_grad_diff;
	charges_grad_diff.resize(cfg.size(), se.CoeffCount());
	charges_grad_diff.set(0);

	vector<double> charges_p(cfg.size()), charges_m(cfg.size()), charges_opt(cfg.size());
	FillWithZero(charges_p);
	FillWithZero(charges_m);
	FillWithZero(charges_opt);
	double diffstep = 1e-4;

	//for (int i = 0; i < cfg.size()*cfg.size(); i++)
	//	cout << mtpr_plus_qeq.QEqHess[0][i] << " ";
	//cout << endl;

	cfg.has_charges(true);

	qeq.OptimizeCharges(cfg, charges_opt, mtpr_plus_qeq.QEqHess[0]);
	for (int i = 0; i < cfg.size(); i++) 
		cfg.charges(i) = charges_opt[i];

	qeq.OptimizeChargesGrad(cfg, charges_grad, mtpr_plus_qeq.QEqHess[0]);	

	for (int idx = 0; idx < se.CoeffCount(); idx++) {
		//cout << "plus, idx = " << idx << endl;
		se.Coeff()[idx] += diffstep;

		//cout << "hessian: " << endl;
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		//for (int i = 0; i < cfg.size(); i++) 
		//	cout << mtpr_plus_qeq.QEqHess[0][i*cfg.size()+i] << " ";
		//cout << endl;  

		qeq.OptimizeCharges(cfg, charges_p, mtpr_plus_qeq.QEqHess[0]);

		//cout << "minus, idx = " << idx << endl;
		se.Coeff()[idx] -= 2*diffstep;

		//cout << "hessian: " << endl;
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		//for (int i = 0; i < cfg.size(); i++) 
		//	cout << mtpr_plus_qeq.QEqHess[0][i*cfg.size()+i] << " ";
		//cout << endl;  

		qeq.OptimizeCharges(cfg, charges_m, mtpr_plus_qeq.QEqHess[0]);

		for (int i = 0; i < cfg.size(); i++) 
			charges_grad_diff(i, idx) = (charges_p[i]-charges_m[i])/(2*diffstep);

		se.Coeff()[idx] += diffstep;
	}

	for (int i = 0; i < cfg.size(); i++) {	
		for (int idx = 0; idx < se.CoeffCount(); idx++) {
			//cout << charges_grad_diff(i, idx) << " " << charges_grad(i, idx) << " ";
			//if (fabs(charges_grad(i, idx)) > 1e-10) cout << charges_grad_diff(i, idx) / charges_grad(i, idx) << endl;
			//else cout << endl;
			if (fabs(charges_grad_diff(i, idx)-charges_grad(i, idx)) > absErrMax) absErrMax = fabs(charges_grad_diff(i, idx)-charges_grad(i, idx)); 
		}
	}

	if (absErrMax > 1E-3) {
		cout << "absErrMax = ";
		cout << absErrMax << endl;
		FAIL()
	}

} END_TEST

TEST("QEq CalcEGrad check with finite differences") {
	Configuration cfg;
	double relErrMax = 0.0, absErrMax = 0.0;
	double relErrForceMax = 0.0, absErrForceMax = 0;

	MLMTPR mtpr(PATH+"unlearned3.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 0;

	SelfEnergy se(PATH+"unlearned3.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++)
		se.Coeff()[i] = 3.5*(i+1);

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 0.0;
	double wgt_eqtn_stress = 0.0;
	double opt_weight_charges = 0.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges); 

	ifstream ifs1(PATH+"LiTi2O4_01_train.cfgs", ios::binary);
	//ifstream ifs1("NaCl_small.cfgs", ios::binary);
	//cfg.Load(ifs1);
	//ifstream ifs1("NaClVasp300.cfgs", ios::binary);
	//for (int ii = 0; cfg_dummy.Load(ifs1); ii++) {
	//	if (ii == 90) cfg = cfg_dummy;
	//}
	//ifstream ifs1("CSHfirst.cfgs", ios::binary);

	vector<Configuration> train_set;
	cfg.Load(ifs1);
	train_set.push_back(cfg);
	ifs1.close();

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mtpr_plus_qeq.GenerateQEqHess(train_set[k], k);

	vector<double> energy_grad;
	vector<double> charges_opt(cfg.size());
	FillWithZero(charges_opt);

	cfg.has_charges(true);

	qeq.OptimizeCharges(cfg, charges_opt, mtpr_plus_qeq.QEqHess[0]);
	for (int i = 0; i < cfg.size(); i++) 
		cfg.charges(i) = charges_opt[i];

	qeq.CalcEGrad(cfg, energy_grad);

	double diffstep = 1e-4;
	for (int idx = 0; idx < se.CoeffCount(); idx++) {
		//cout << "idx = " << idx << endl;
		Configuration cfg1 = cfg;
		se.Coeff()[idx] += diffstep;
		vector<double> charges_opt1(cfg.size());
		FillWithZero(charges_opt1);
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		qeq.OptimizeCharges(cfg1, charges_opt1, mtpr_plus_qeq.QEqHess[0]);
		for (int ii = 0; ii < cfg1.size(); ii++)
			cfg1.charges(ii) = charges_opt1[ii];
		qeq.CalcE(cfg1);
		Configuration cfg2 = cfg;
		se.Coeff()[idx] -= 2*diffstep;
		vector<double> charges_opt2(cfg.size());
		FillWithZero(charges_opt2);
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		qeq.OptimizeCharges(cfg2, charges_opt2, mtpr_plus_qeq.QEqHess[0]);
		for (int ii = 0; ii < cfg2.size(); ii++)
			cfg2.charges(ii) = charges_opt2[ii];
		qeq.CalcE(cfg2);
		double energy_grad_cdiff = (cfg1.energy - cfg2.energy) / (2 * diffstep);
		//cout << energy_grad[idx] << " " << energy_grad_cdiff << endl;
		double absErrForce = fabs(energy_grad[idx] - energy_grad_cdiff);
		double relErrForce = fabs((energy_grad[idx] - energy_grad_cdiff) / energy_grad_cdiff);
		if (relErrForce > relErrForceMax) relErrForceMax = relErrForce;
		if (absErrForce > absErrForceMax) absErrForceMax = absErrForce;
		se.Coeff()[idx] += diffstep;
	}

	if (absErrForceMax > 2E-3) {
		cout << "absErrForceMax = ";
		cout << absErrForceMax << endl;
		FAIL()
	}

} END_TEST

TEST("QEq CalcEFSGrad check with finite differences") {
	Configuration cfg;

	MLMTPR mtpr(PATH+"unlearned3.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 0 * rand() / RAND_MAX;;

	SelfEnergy se(PATH+"unlearned3.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++) {
		//qeq.Coeff()[i] = 3.5*(i+1);
		if ((i+1) % 3 == 0) se.Coeff()[i] = 0.7*(5 + rand() % 10);
		else {
			int random = rand() % 100;
			se.Coeff()[i] = 0.8*(-50 + random);
		} 
	}

	for (int i = 0; i < se.CoeffCount(); i++) {
		se.Coeff()[i] *= -1;
		//cout << qeq.Coeff()[i] << " ";
	}
	//cout << endl;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 0.0;
	double wgt_eqtn_stress = 0.0;
	double opt_weight_charges = 0.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges); 

	ifstream ifs1(PATH+"LiTi2O4_01_train.cfgs", ios::binary);
	//ifstream ifs1("NaCl_small.cfgs", ios::binary);
	//cfg.Load(ifs1);
	//ifstream ifs1("NaClVasp300.cfgs", ios::binary);
	//for (int ii = 0; cfg_dummy.Load(ifs1); ii++) {
	//	if (ii == 90) cfg = cfg_dummy;
	//}
	//ifstream ifs1("CSHfirst.cfgs", ios::binary);

	vector<Configuration> train_set;
	cfg.Load(ifs1);
	train_set.push_back(cfg);
	ifs1.close();

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mtpr_plus_qeq.GenerateQEqHess(train_set[k], k);

	vector<double> energy_grad;
	Array3D forces_grad;
	Array2D stress_grad;
	Array3D forces_grad_cdiff;
	Array2D stress_grad_cdiff;
	vector<double> charges_opt(cfg.size());
	FillWithZero(charges_opt);
	Array2D charges_grad;
	charges_grad.resize(cfg.size(), se.CoeffCount());
	charges_grad.set(0);

	cfg.has_charges(true);

	qeq.OptimizeCharges(cfg, charges_opt, mtpr_plus_qeq.QEqHess[0]);
	for (int i = 0; i < cfg.size(); i++) 
		cfg.charges(i) = charges_opt[i];

	//for (int i = 0; i < cfg.size(); i++)
	//	cout << cfg.charges(i) << " ";
	//cout << endl;

	qeq.OptimizeChargesGrad(cfg, charges_grad, mtpr_plus_qeq.QEqHess[0]);

	double diffstep = 1e-5;
	//vector<double> charges_p(cfg.size());
	//FillWithZero(charges_p);

	//for (int idx = 0; idx < se.CoeffCount(); idx++) {
	//	se.Coeff()[idx] += diffstep;

	//	mtpr_plus_qeq.GenerateQEqHess(0);
	//	qeq.OptimizeCharges(cfg, charges_p, mtpr_plus_qeq.QEqHess[0]);

		//for (int i = 0; i < cfg.size(); i++)
		//	charges_p[i] = cfg.charges(i);

	//	for (int i = 0; i < cfg.size(); i++) {
	//		charges_grad(i, idx) = (charges_p[i]-charges_opt[i])/diffstep;
	//	}

	//	se.Coeff()[idx] -= diffstep;
	//}

	qeq.CalcEFSGrad(cfg, energy_grad, forces_grad, stress_grad, charges_grad);
	Configuration cfg_valid = cfg;
	qeq.CalcEFS(cfg_valid);

	double absErrEnergy = fabs(cfg.energy - cfg_valid.energy);

	double absErrMaxForce = 0;
	for (int i = 0; i < cfg.size(); i++) {
		for (int l = 0; l < 3; l++) {
			double absErr = fabs(cfg.force(i)[l] - cfg_valid.force(i)[l]);
			if (absErr > absErrMaxForce) absErrMaxForce = absErr;
			//std::cout << cfg_valid.force(i)[l] << " " << cfg.force(i)[l] << " " << absErr << std::endl;
		}
	}

	double absErrMaxStress = 0;
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++) {
			double absErr = fabs(cfg.stresses[a][b] - cfg_valid.stresses[a][b]);
			if (absErr > absErrMaxForce) absErrMaxForce = absErr;
			//std::cout << cfg.stresses[a][b] << " " << cfg_valid.stresses[a][b] << " " << absErr << std::endl;
		}
	}

	double relErrEnergyGradMax = 0.0, absErrEnergyGradMax = 0;
	double relErrForceGradMax = 0.0, absErrForceGradMax = 0;
	double relErrStressGradMax = 0.0, absErrStressGradMax = 0;
	forces_grad_cdiff.resize(cfg.size(), se.CoeffCount(), 3);
	stress_grad_cdiff.resize(9, se.CoeffCount());
	for (int idx = 0; idx < se.CoeffCount(); idx++) {
		//cout << "idx = " << idx << endl;
		Configuration cfg1 = cfg;
		se.Coeff()[idx] += diffstep;
		vector<double> charges_opt1(cfg.size());
		FillWithZero(charges_opt1);
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		qeq.OptimizeCharges(cfg1, charges_opt1, mtpr_plus_qeq.QEqHess[0]);
		for (int ii = 0; ii < cfg1.size(); ii++)
			cfg1.charges(ii) = charges_opt1[ii];
		qeq.CalcEFS(cfg1);
		Configuration cfg2 = cfg;
		se.Coeff()[idx] -= 2*diffstep;
		vector<double> charges_opt2(cfg.size());
		FillWithZero(charges_opt2);
		mtpr_plus_qeq.GenerateQEqHess(train_set[0], 0);
		qeq.OptimizeCharges(cfg2, charges_opt2, mtpr_plus_qeq.QEqHess[0]);
		for (int ii = 0; ii < cfg2.size(); ii++)
			cfg2.charges(ii) = charges_opt2[ii];
		qeq.CalcEFS(cfg2);
		double energy_grad_cdiff = (cfg1.energy - cfg2.energy) / (2 * diffstep);
		//cout << "energy grad:" << endl;
		//cout << energy_grad[idx] << " " << energy_grad_cdiff << endl;
		double absErrEnergy = fabs(energy_grad[idx] - energy_grad_cdiff);
		double relErrEnergy = fabs((energy_grad[idx] - energy_grad_cdiff) / energy_grad_cdiff);
		if (relErrEnergy > relErrEnergyGradMax) relErrEnergyGradMax = relErrEnergy;
		if (absErrEnergy > absErrEnergyGradMax) absErrEnergyGradMax = absErrEnergy;

		//cout << "force grad:" << endl;
		for (int i = 0; i < cfg.size(); i++) {
			for (int l = 0; l < 3; l++) {
				forces_grad_cdiff(i, idx, l) = (cfg1.force(i)[l] - cfg2.force(i)[l]) / (2 * diffstep);
				//cout << forces_grad(i, idx, l) << " " << forces_grad_cdiff(i, idx, l) << endl;
				if (fabs(forces_grad_cdiff(i, idx, l)) > 1E-6) {
					double absErrForce = fabs(forces_grad(i, idx, l) - forces_grad_cdiff(i, idx, l));
					double relErrForce = fabs((forces_grad(i, idx, l) - forces_grad_cdiff(i, idx, l))
						/ forces_grad_cdiff(i, idx, l));
					if (relErrForce > relErrForceGradMax) relErrForceGradMax = relErrForce;
					if (absErrForce > absErrForceGradMax) absErrForceGradMax = absErrForce;
					//cout << " " << absErrForce << " " << relErrForce;
				}
				//cout << endl;
			}
		}

		//cout << "stress grad:" << endl;
		for (int a = 0; a < 3; a++) {
			for (int l = 0; l < 3; l++) {
				stress_grad_cdiff(a*3+l, idx) = (cfg1.stresses[a][l] - cfg2.stresses[a][l]) / (2 * diffstep);
				//cout << stress_grad(a*3+l, idx) << " " << stress_grad_cdiff(a*3+l, idx) << endl;
				if (fabs(stress_grad_cdiff(a*3+l, idx)) > 1E-6) {
					double absErrStress = fabs(stress_grad(a*3+l, idx) - stress_grad_cdiff(a*3+l, idx));
					double relErrStress = fabs((stress_grad(a*3+l, idx) - stress_grad_cdiff(a*3+l, idx))
						/ stress_grad_cdiff(a*3+l, idx));
					if (relErrStress > relErrStressGradMax) relErrStressGradMax = relErrStress;
					if (absErrStress > absErrStressGradMax) absErrStressGradMax = absErrStress;
					//cout << " " << absErrStress << " " << relErrStress;
				}
				//cout << endl;
			}
		}

		se.Coeff()[idx] += diffstep;
	}

	if (absErrEnergy > 1E-9) {
		cout << "absErrEnergy = ";
		cout << absErrEnergy << endl;
		FAIL()
	}

	if (absErrMaxForce > 1E-9) {
		cout << "absErrMaxForce = ";
		cout << absErrMaxForce << endl;
		FAIL()
	}

	if (absErrMaxStress > 1E-9) {
		cout << "absErrMaxStress = ";
		cout << absErrMaxStress << endl;
		FAIL()
	}

	if (absErrEnergyGradMax > 1E-3) {
		cout << "absErrEnergyGradMax = ";
		cout << absErrEnergyGradMax << endl;
		FAIL()
	}

	if (absErrForceGradMax > 1E-3) {
		cout << "absErrForceGradMax = ";
		cout << absErrForceGradMax << endl;
		FAIL()
	}

	if (absErrStressGradMax > 1E-3) {
		cout << "absErrStressGradMax = ";
		cout << absErrStressGradMax << endl;
		FAIL()
	}

} END_TEST

TEST("MTPR+QEq GenerateQEqHess check with finite differences") {
	Configuration cfg;
	double diffstep = 1e-3;
	double relErrMax = 0.0, absErrMax = 0.0;

	SelfEnergy se(PATH+"unlearned2.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++)
		se.Coeff()[i] = 4.5*(i+1);

	MLMTPR mtpr(PATH+"learned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 1e-2*rand() / RAND_MAX;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 1.0;
	double wgt_eqtn_stress = 1.0;
	double opt_weight_charges = 1.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges);

	ifstream ifs(PATH+"NaCl8at.cfgs", ios::binary);
	//ifstream ifs("CSHfirst.cfgs", ios::binary);

	vector<Configuration> train_set;

	for (int i = 0; cfg.Load(ifs); i++)
		train_set.push_back(cfg);

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);

	double dummy, dummy1, dummy2;
	Configuration cfg1, cfg2, cfg3, cfg4;

	for (int k = 0; k < train_set.size(); k++) {
		cfg = train_set[k];
		mtpr_plus_qeq.GenerateQEqHess(cfg, k);
		vector<double> charges(cfg.size());
		for (int jj = 0; jj < cfg.size(); jj++) {
			//cfg.charges(jj) = -1*sin(2*jj-cfg.size()+1);
			charges[jj] = cfg.charges(jj);
			//cout << charges[jj] << " ";
		}
		//cout << endl;

		for (int ii = 0; ii < cfg.size(); ii++)
		{
			dummy = charges[ii];
			charges[ii] = dummy+diffstep;
			cfg1 = cfg;
			cfg1.charges(ii)=charges[ii];
			qeq.CalcE(cfg1);
			charges[ii] = dummy-diffstep;
			cfg2 = cfg;
			cfg2.charges(ii)=charges[ii];
			qeq.CalcE(cfg2);
			charges[ii] = dummy;
			cfg3 = cfg;
			cfg3.charges(ii)=charges[ii];
			qeq.CalcE(cfg3);
			double AnalyticHess = mtpr_plus_qeq.QEqHess[k][ii*cfg.size()+ii];
			double CdiffHess = (cfg1.energy-2*cfg3.energy+cfg2.energy)/(diffstep*diffstep);
			//cout << CdiffHess << ", " << AnalyticHess << endl;
			double absErr = fabs(AnalyticHess - CdiffHess);
			double relErr = fabs((AnalyticHess - CdiffHess) / CdiffHess);
			if (relErr > relErrMax) relErrMax = relErr;
			if (absErr > absErrMax) absErrMax = absErr;
		}

		for (int ii = 0; ii < cfg.size(); ii++)
		{
			for (int j = 0; j < cfg.size(); j++)
			{
				if (ii != j)
				{
					dummy1 = charges[ii];
					dummy2 = charges[j];
					charges[ii] = dummy1+diffstep;
					charges[j] = dummy2+diffstep;
					cfg1 = cfg;
					cfg1.charges(ii)=charges[ii];
					cfg1.charges(j)=charges[j];
					qeq.CalcE(cfg1);
					charges[ii] = dummy1+diffstep;
					charges[j] = dummy2-diffstep;
					cfg2 = cfg;
					cfg2.charges(ii)=charges[ii];
					cfg2.charges(j)=charges[j];
					qeq.CalcE(cfg2);
					charges[ii] = dummy1-diffstep;
					charges[j] = dummy2+diffstep;
					cfg3 = cfg;
					cfg3.charges(ii)=charges[ii];
					cfg3.charges(j)=charges[j];
					qeq.CalcE(cfg3);
					charges[ii] = dummy1-diffstep;
					charges[j] = dummy2-diffstep;
					cfg4 = cfg;
					cfg4.charges(ii)=charges[ii];
					cfg4.charges(j)=charges[j];
					qeq.CalcE(cfg4);
					double AnalyticHess = mtpr_plus_qeq.QEqHess[k][ii*cfg.size()+j];
					double CdiffHess = (cfg1.energy-cfg2.energy-cfg3.energy+cfg4.energy)/(4*diffstep*diffstep);
					//cout << CdiffHess << ", " << AnalyticHess << endl;
					double absErr = fabs(AnalyticHess - CdiffHess);
					double relErr = fabs((AnalyticHess - CdiffHess) / CdiffHess);
					if (relErr > relErrMax) relErrMax = relErr;
					if (absErr > absErrMax) absErrMax = absErr;
					charges[ii] = dummy1;
					charges[j] = dummy2;
				}
			}
		}
		if (relErrMax > 1E-3) {
			cout << relErrMax << endl;
			FAIL()
		}
		if (absErrMax > 1E-3) {
			cout << absErrMax << endl;
			FAIL()
		}
	}

} END_TEST

TEST("MTPR+QEq CalcEFS (check forces correctness with cdiffs)") {
	Configuration cfg, cfg1, cfg2;
	double diffstep = 1e-5;
	double relErrMax = 0.0, absErrMax = 0.0;
	double relErrMaxStr = 0.0, absErrMaxStr = 0.0;

	SelfEnergy se(PATH+"unlearned2.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++) {
		se.Coeff()[i] = 3.5*(i+1);
		if ((i+1) % 3 == 0) se.Coeff()[i] = 0.7*(5 + rand() % 10);
		else {
			int random = rand() % 100;
			se.Coeff()[i] = 0.8*(-50 + random);
		} 
	}

	MLMTPR mtpr(PATH+"unlearned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 1.0 * rand() / RAND_MAX;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 1.0;
	double wgt_eqtn_stress = 1.0;
	double opt_weight_charges = 1.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges);

	//ifstream ifs("LiTi2O4_01_train.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl8at.cfgs", ios::binary);
	//ifstream ifs("CSHfirst.cfgs", ios::binary);

	vector<Configuration> train_set;

	for (int i = 0; cfg.Load(ifs); i++) {
		//if (i == 128) {
			Matrix3 mapping;

			mapping *= 0;

			double angle = 3.14159265358979323846 /4;
			double cos_ang = cos(angle);
			double sin_ang = sin(angle);

			mapping[0][0] = cos_ang;
			mapping[0][1] = -sin_ang;
			mapping[1][0] = sin_ang;
			mapping[1][1] = cos_ang;
			mapping[2][2] = 1;

			cfg.Deform(mapping);
			train_set.push_back(cfg);
		//}
	}

	cfg = train_set[0];

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) { 
	//		cout << mtpr_plus_qeq.train_set[0].lattice[i][j] << " ";
	//	}
	//	cout << endl;
	//}

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mtpr_plus_qeq.GenerateQEqHess(train_set[k], k);

	cfg.has_forces(true);

	mtpr_plus_qeq.CalcEFS(cfg, mtpr_plus_qeq.QEqHess[0]);

	//cout << endl;

	//for (int i = 0; i < cfg.size(); i++)
	//	cout << cfg.charges(i) <<  " ";
	//cout << endl;

	//cout << "forces:" << endl;
	for (int i = 0; i < cfg.size(); i++) {
		//cout << i << endl;
		for (int l = 0; l < 3; l++) {
			vector<double> QEqHess_p;
			vector<double> QEqHess_m;
			Configuration cfg1 = cfg;
			Configuration cfg2 = cfg;
			cfg1.pos(i)[l] += diffstep;
			cfg2.pos(i)[l] -= diffstep;
			//cout << "plus" << endl;
			mtpr_plus_qeq.GenerateQEqHess(cfg1, QEqHess_p);
			mtpr_plus_qeq.CalcEFS(cfg1, QEqHess_p);
			//cout << "minus" << endl;
			mtpr_plus_qeq.GenerateQEqHess(cfg2, QEqHess_m);
			mtpr_plus_qeq.CalcEFS(cfg2, QEqHess_m);
			//cout << cfg1.energy << " " << cfg2.energy << endl;
			double cdiffForce = (cfg2.energy - cfg1.energy) / (2 * diffstep);
			//if (fabs(cfg.force(i)[l]) < 1e-6) continue;
			double absErr = fabs(cfg.force(i)[l] - cdiffForce);
			double relErr = fabs((cfg.force(i)[l] - cdiffForce) / cdiffForce);
			if (relErr > relErrMax) relErrMax = relErr;
			if (absErr > absErrMax) absErrMax = absErr;
			//std::cout << cdiffForce << " " << cfg.force(i)[l] << " " << absErr << std::endl;
		}
	}


	//cout << "stresses:" << endl;
	Matrix3 dEdL = cfg.lattice.inverse().transpose() * cfg.stresses;
	Matrix3 fd_dEdL(0,0,0,0,0,0,0,0,0);
	for (int a = 0; a < 3; a++) 
		for (int b = 0; b < 3; b++)
		{
			Configuration cfg1 = cfg;
			Configuration cfg2 = cfg;
			vector<double> QEqHess_p;
			vector<double> QEqHess_m;

			Matrix3 deform = cfg.lattice;

			deform[a][b] += diffstep;
			cfg1.Deform(cfg1.lattice.inverse()*deform);
			mtpr_plus_qeq.GenerateQEqHess(cfg1, QEqHess_p);

			deform[a][b] -= 2*diffstep;
			cfg2.Deform(cfg2.lattice.inverse()*deform);
			mtpr_plus_qeq.GenerateQEqHess(cfg2, QEqHess_m);

			mtpr_plus_qeq.CalcEFS(cfg1, QEqHess_p);
			mtpr_plus_qeq.CalcEFS(cfg2, QEqHess_m);

			fd_dEdL[a][b] = (cfg1.energy-cfg2.energy) / (cfg1.lattice[a][b]-cfg2.lattice[a][b]);

			double absErrStr = fabs(dEdL[a][b] - fd_dEdL[a][b]);
			double relErrStr = fabs((dEdL[a][b] - fd_dEdL[a][b]) / fd_dEdL[a][b]);
			if (relErrStr > relErrMaxStr) relErrMaxStr = relErrStr;
			if (absErrStr > absErrMaxStr) absErrMaxStr = absErrStr;
			//cout << fd_dEdL[a][b] << " " << dEdL[a][b] << " " << absErrStr << endl;
		}

	if (relErrMax > 3E-2) {
		cout << relErrMax << endl;
		FAIL()
	}
	if (absErrMax > 3E-2) {
		cout << absErrMax << endl;
		FAIL()
	}

	if (relErrMax > 3E-2) {
		cout << relErrMaxStr << endl;
		FAIL()
	}
	if (absErrMax > 3E-2) {
		cout << absErrMaxStr << endl;
		FAIL()
	}

	ifs.close();

} END_TEST;

TEST("MTPR+QEq AddLossGrad test by finite difference") {
	Configuration cfg;
	double diffstep = 1e-4;
	double relErrMax = 0.0, absErrMax = 0.0;

	SelfEnergy se(PATH+"unlearned2.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++)
		se.Coeff()[i] = 2.5*(i+1);

	MLMTPR mtpr(PATH+"learned.mtp");

	for (int i = 0; i < mtpr.CoeffCount(); i++)
		mtpr.Coeff()[i] = 1e-2*rand() / RAND_MAX;

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 100.0;
	double wgt_eqtn_stress = 10.0;
	double opt_weight_charges = 100.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	MTPRplusQEQ mtpr_plus_qeq(&mtpr, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges);

	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);
	//ifstream ifs("NaClVasp300.cfgs", ios::binary);
	//ifstream ifs("CSH_DFT2.cfgs", ios::binary);
	vector<Configuration> train_set;
	for (int i = 0; cfg.Load(ifs); i++)
		train_set.push_back(cfg);

	mtpr_plus_qeq.AllocateMemoryForHess(train_set);
	mtpr_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mtpr_plus_qeq.GenerateQEqHess(train_set[k], k);

	vector<double> cdiff(mtpr.CoeffCount()+se.CoeffCount());
	FillWithZero(cdiff);

	mtpr_plus_qeq.CalcObjectiveFunctionGrad(train_set);

	for (int j = 0; j < mtpr.CoeffCount(); j++) {
		mtpr.Coeff()[j] += diffstep;
		double lossp = mtpr_plus_qeq.ObjectiveFunction(train_set);
		mtpr.Coeff()[j] -= 2 * diffstep;
		double lossm = mtpr_plus_qeq.ObjectiveFunction(train_set);
		mtpr.Coeff()[j] += diffstep;
		cdiff[j] += (lossp - lossm) / (2 * diffstep);
	}

	for (int j = mtpr.CoeffCount(); j < mtpr.CoeffCount()+se.CoeffCount(); j++) {
		se.Coeff()[j-mtpr.CoeffCount()] += diffstep;
		double lossp = mtpr_plus_qeq.ObjectiveFunction(train_set);
		se.Coeff()[j-mtpr.CoeffCount()] -= diffstep;
		double lossm = mtpr_plus_qeq.ObjectiveFunction(train_set);
		//se.Coeff()[j] += diffstep;
		cdiff[j] += (lossp - lossm) / (diffstep);
	}

	double control_delta = 1E-3;
	if (!mtpr_plus_qeq.CheckLossGrad(control_delta, cdiff))
		FAIL()

	//qeq.Save("test.qeq");

	ifs.close();

} END_TEST;

TEST("MLIP+QEq CalcEFS (check forces correctness with cdiffs)") {
	Configuration cfg, cfg1, cfg2;
	double diffstep = 1e-4;
	double relErrMax = 0.0, absErrMax = 0.0;
	double relErrMaxStr = 0.0, absErrMaxStr = 0.0;

	SelfEnergy se(PATH+"unlearned2.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++) {
		se.Coeff()[i] = 3.5*(i+1);
		if ((i+1) % 3 == 0) se.Coeff()[i] = 0.7*(5 + rand() % 10);
		else {
			int random = rand() % 100;
			se.Coeff()[i] = 0.8*(-50 + random);
		} 
	}

	NonlinearLennardJones lj(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);
	//LennardJones lj(PATH+"unlearned_lj_lin.mlip", 1.6, 5.0);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		//lj.Coeff()[i] = 0.25/(i+1);
		lj.Coeff()[i] = pow(fabs(4.5*sin(i + 1)),0.25);
	}

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 1.0;
	double wgt_eqtn_stress = 1.0;
	double opt_weight_charges = 1.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	AnyLocalMLIPplusQEQ mlip_plus_qeq(&lj, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges);

	//ifstream ifs("LiTi2O4_01_train.cfgs", ios::binary);
	ifstream ifs(PATH+"NaCl8at.cfgs", ios::binary);
	//ifstream ifs("CSHfirst.cfgs", ios::binary);

	vector<Configuration> train_set;

	for (int i = 0; cfg.Load(ifs); i++) {
		//if (i == 128) {
			Matrix3 mapping;

			mapping *= 0;

			double angle = 3.14159265358979323846 /4;
			double cos_ang = cos(angle);
			double sin_ang = sin(angle);

			mapping[0][0] = cos_ang;
			mapping[0][1] = -sin_ang;
			mapping[1][0] = sin_ang;
			mapping[1][1] = cos_ang;
			mapping[2][2] = 1;

			cfg.Deform(mapping);
			train_set.push_back(cfg);
		//}
	}

	cfg = train_set[0];

	//for (int i = 0; i < 3; i++) {
	//	for (int j = 0; j < 3; j++) { 
	//		cout << mlip_plus_qeq.train_set[0].lattice[i][j] << " ";
	//	}
	//	cout << endl;
	//}

	mlip_plus_qeq.AllocateMemoryForHess(train_set);
	mlip_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mlip_plus_qeq.GenerateQEqHess(train_set[k], k);

	cfg.has_forces(true);

	mlip_plus_qeq.CalcEFS(cfg, mlip_plus_qeq.QEqHess[0]);

	//cout << endl;

	//for (int i = 0; i < cfg.size(); i++)
	//	cout << cfg.charges(i) <<  " ";
	//cout << endl;

	//cout << "forces:" << endl;
	for (int i = 0; i < cfg.size(); i++) {
		//cout << i << endl;
		for (int l = 0; l < 3; l++) {
			vector<double> QEqHess_p;
			vector<double> QEqHess_m;
			Configuration cfg1 = cfg;
			Configuration cfg2 = cfg;
			cfg1.pos(i)[l] += diffstep;
			cfg2.pos(i)[l] -= diffstep;
			//cout << "plus" << endl;
			mlip_plus_qeq.GenerateQEqHess(cfg1, QEqHess_p);
			mlip_plus_qeq.CalcEFS(cfg1, QEqHess_p);
			//cout << "minus" << endl;
			mlip_plus_qeq.GenerateQEqHess(cfg2, QEqHess_m);
			mlip_plus_qeq.CalcEFS(cfg2, QEqHess_m);
			//cout << cfg1.energy << " " << cfg2.energy << endl;
			double cdiffForce = (cfg2.energy - cfg1.energy) / (2 * diffstep);
			//if (fabs(cfg.force(i)[l]) < 1e-6) continue;
			double absErr = fabs(cfg.force(i)[l] - cdiffForce);
			double relErr = fabs((cfg.force(i)[l] - cdiffForce) / cdiffForce);
			if (relErr > relErrMax) relErrMax = relErr;
			if (absErr > absErrMax) absErrMax = absErr;
			//std::cout << cdiffForce << " " << cfg.force(i)[l] << " " << absErr << std::endl;
		}
	}

	//cout << "stresses:" << endl;
	Matrix3 dEdL = cfg.lattice.inverse().transpose() * cfg.stresses;
	Matrix3 fd_dEdL(0,0,0,0,0,0,0,0,0);
	for (int a = 0; a < 3; a++) 
		for (int b = 0; b < 3; b++)
		{
			Configuration cfg1 = cfg;
			Configuration cfg2 = cfg;
			vector<double> QEqHess_p;
			vector<double> QEqHess_m;

			Matrix3 deform = cfg.lattice;

			deform[a][b] += diffstep;
			cfg1.Deform(cfg1.lattice.inverse()*deform);
			mlip_plus_qeq.GenerateQEqHess(cfg1, QEqHess_p);

			deform[a][b] -= 2*diffstep;
			cfg2.Deform(cfg2.lattice.inverse()*deform);
			mlip_plus_qeq.GenerateQEqHess(cfg2, QEqHess_m);

			mlip_plus_qeq.CalcEFS(cfg1, QEqHess_p);
			mlip_plus_qeq.CalcEFS(cfg2, QEqHess_m);

			fd_dEdL[a][b] = (cfg1.energy-cfg2.energy) / (cfg1.lattice[a][b]-cfg2.lattice[a][b]);

			double absErrStr = fabs(dEdL[a][b] - fd_dEdL[a][b]);
			double relErrStr = fabs((dEdL[a][b] - fd_dEdL[a][b]) / fd_dEdL[a][b]);
			if (relErrStr > relErrMaxStr) relErrMaxStr = relErrStr;
			if (absErrStr > absErrMaxStr) absErrMaxStr = absErrStr;
			//cout << fd_dEdL[a][b] << " " << dEdL[a][b] << " " << absErrStr << endl;
		}

	if (relErrMax > 3E-2) {
		cout << relErrMax << endl;
		FAIL()
	}
	if (absErrMax > 3E-2) {
		cout << absErrMax << endl;
		FAIL()
	}

	if (relErrMax > 3E-2) {
		cout << relErrMaxStr << endl;
		FAIL()
	}
	if (absErrMax > 3E-2) {
		cout << absErrMaxStr << endl;
		FAIL()
	}

	ifs.close();

} END_TEST;

TEST("MLIP+QEq AddLossGrad test by finite difference") {
	Configuration cfg;
	double diffstep = 1e-4;
	double relErrMax = 0.0, absErrMax = 0.0;

	SelfEnergy se(PATH+"unlearned2.qeq");

	QEq qeq(&se);

	for (int i = 0; i < se.CoeffCount(); i++)
		se.Coeff()[i] = 2.5*(i+1);

	NonlinearLennardJones lj(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);
	//LennardJones lj(PATH+"unlearned_lj_lin.mlip", 1.6, 5.0);

	for (int i = 0; i < lj.CoeffCount(); i++) {
		//lj.Coeff()[i] = 1.0 / pow(2, i);
		//lj.Coeff()[i] = 218.25/(i+1);
		lj.Coeff()[i] = pow(fabs(7.5*sin(i + 1)),0.25);
	}

	double wgt_eqtn_energy = 1.0;
	double wgt_eqtn_forces = 100.0;
	double wgt_eqtn_stress = 10.0;
	double opt_weight_charges = 100.0;
	double wgt_eqtn_constr = 1.0;
	double opt_weight_relfrc = 0.0;

	AnyLocalMLIPplusQEQ mlip_plus_qeq(&lj, &qeq, wgt_eqtn_energy, wgt_eqtn_forces, wgt_eqtn_stress,
		opt_weight_relfrc, wgt_eqtn_constr, opt_weight_charges);

	//ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);
	ifstream ifs(PATH+"NaClVasp300.cfgs", ios::binary);
	//ifstream ifs("CSH_DFT2.cfgs", ios::binary);
	vector<Configuration> train_set;
	for (int i = 0; cfg.Load(ifs); i++)
		if (i % 20 == 0) train_set.push_back(cfg);

	mlip_plus_qeq.AllocateMemoryForHess(train_set);
	mlip_plus_qeq.GenerateEwHess(train_set);
	for (int k = 0; k < train_set.size(); k++)
		mlip_plus_qeq.GenerateQEqHess(train_set[k], k);

	vector<double> cdiff(lj.CoeffCount()+se.CoeffCount());
	FillWithZero(cdiff);

	mlip_plus_qeq.CalcObjectiveFunctionGrad(train_set);

	for (int j = 0; j < lj.CoeffCount(); j++) {
		lj.Coeff()[j] += diffstep;
		double lossp = mlip_plus_qeq.ObjectiveFunction(train_set);
		lj.Coeff()[j] -= 2 * diffstep;
		double lossm = mlip_plus_qeq.ObjectiveFunction(train_set);
		lj.Coeff()[j] += diffstep;
		cdiff[j] += (lossp - lossm) / (2 * diffstep);
	}

	for (int j = lj.CoeffCount(); j < lj.CoeffCount()+se.CoeffCount(); j++) {
		se.Coeff()[j-lj.CoeffCount()] += diffstep;
		double lossp = mlip_plus_qeq.ObjectiveFunction(train_set);
		se.Coeff()[j-lj.CoeffCount()] -= diffstep;
		double lossm = mlip_plus_qeq.ObjectiveFunction(train_set);
		//se.Coeff()[j] += diffstep;
		cdiff[j] += (lossp - lossm) / (diffstep);
	}

	double control_delta = 1E-3;
	if (!mlip_plus_qeq.CheckLossGrad(control_delta, cdiff))
		FAIL()

	//qeq.Save("test.qeq");

	ifs.close();

} END_TEST;
#endif*/


#ifdef MLIP_MPI
PAR_TEST("training PairPotentialNonlinear") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	PairPotentialNonlinear pair_exact(6, 1.9);
	PairPotentialNonlinear pair(6, 1.9);

	for (int i = 0; i < pair.CoeffCount(); i++) {
		pair_exact.Coeff()[i] = 1.0 / pow(2, i);
		pair.Coeff()[i] = 0.25;
	}

	NonLinearRegression dtr(&pair, 1.0, 1.0, 1.0, 0.0, 1.0);

	Configuration cfg;
	vector<Configuration> training_set;
	//ifstream ifs1("Ti4MVtest.proc3", ios::binary);
	ifstream ifs1(PATH+"NaCl_eam.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs1); i++) {
		pair_exact.CalcEFS(cfg);

		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}
	ifs1.close();

	dtr.Train(training_set);

	// comparing
	double max_coeff_difference = 0;
	for (int i = 0; i < pair.CoeffCount(); i++) {
		max_coeff_difference = std::max(max_coeff_difference,
			fabs((pair.Coeff()[i] - pair_exact.Coeff()[i])));
		//std::cout << pair.Coeff()[i] << " " << pair_exact.Coeff()[i] << std::endl;
	}
	if (max_coeff_difference > 0.02) {
		std::cerr << "(error: max_coeff_difference is " << max_coeff_difference << ") ";
		FAIL()
	}
	double max_en_difference = 0;

	for (const Configuration& cfg_orig : training_set) {
		cfg = cfg_orig;
		pair.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
		//std::cout << cfg.energy << " " << pair.train_set[i].energy << " ";
		//std::cout << fabs((cfg.energy - pair.train_set[i].energy)/pair.train_set[i].energy) << std::endl;
	}
	if (max_en_difference > 0.02) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}

} END_TEST;

PAR_TEST("EAMSimple fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	EAMSimple eam_exact(8, 5, 1.6, 5.0);
	EAMSimple eam(8, 5, 1.6, 5.0);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam_exact.Coeff()[i] = 0.5 * i / eam.CoeffCount();
		//eam.Coeff()[i] = 1.0 / (i + 1);
		eam_exact.Coeff()[i] = 1.3*sin(i+1);
		eam.Coeff()[i] = 0.1;
	}

	NonLinearRegression trainer(&eam, 1000.0, 1.0, 0.1, 0.0, 1.0);

	Configuration cfg;
	vector<Configuration> training_set;
	ifstream ifs1(PATH+"NaCl_small.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs1); i++) {
		//for (int i = 0; i < 1; i++) {
		//cfg.load(ifs1);
		eam_exact.CalcEFS(cfg);
		//cout << cfg.energy << endl;
		//for (int j = 0; j < cfg.size(); j++)
		//	cout << cfg.force(j)[0] << " " << cfg.force(j)[1] << " " << cfg.force(j)[2] << endl;
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}
	ifs1.close();

	trainer.Train(training_set);

	// comparing
	//double max_coeff_difference = 0;
	//for (int i = 0; i < eam.CoeffCount(); i++) {
	//	std::cout << eam.Coeff()[i] << " " << eam_exact.Coeff()[i] << std::endl;
	//	max_coeff_difference = std::max(max_coeff_difference,
	//		fabs((eam.Coeff()[i] - eam_exact.Coeff()[i])));
	//}
	//if (max_coeff_difference > 0.02) {
	//	std::cerr << "(error: max_coeff_difference is " << max_coeff_difference << ") ";
	//	//FAIL()
	//}

	//double sum_coeffs2 = 0;
	//for (int i = 8; i < 2*8; i++) {
	//	sum_coeffs2 += eam.Coeff()[i]*eam.Coeff()[i];
	//	//std::cout << eam.Coeff()[i] << std::endl;
	//} 
	//std::cout << "sum_coeffs2 = " << sum_coeffs2 << std::endl;

	// comparing
	double max_en_difference = 0;
	for (const Configuration& cfg_orig : training_set) {
		cfg = cfg_orig;
		eam.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
		//std::cout << "energy: " << std::endl;
		//std::cout << cfg.energy << " " << cfg_orig.energy << " ";
		//std::cout << fabs((cfg.energy - cfg_orig.energy)/cfg_orig.energy) << std::endl;
		//std::cout << "forces: " << std::endl;
		//for (int ii = 0; ii < cfg_orig.size(); ii++) {
		//	std::cout << cfg.force(ii)[0] << " " << cfg_orig.force(ii)[0] << " ";
		//	std::cout << cfg.force(ii)[1] << " " << cfg_orig.force(ii)[1] << " ";
		//	std::cout << cfg.force(ii)[2] << " " << cfg_orig.force(ii)[2] << std::endl;
		//}
		//std::cout << "stresses: " << std::endl;
		//for (int a = 0; a < 3; a++)
		//	for (int b = 0; b < 3; b++)
		//		std::cout << cfg.stresses[a][b] << " " << eam.train_set[i].stresses[a][b] << std::endl;
	}
	if (max_en_difference > 0.02) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}

} END_TEST;

PAR_TEST("EAMMult fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	EAMMult eam_exact(5, 3, 2, 1.6, 5.0);
	EAMMult eam(5, 3, 2, 1.6, 5.0);

	for (int i = 0; i < eam.CoeffCount(); i++) {
		//eam_exact.Coeff()[i] = 0.5 * i / eam.CoeffCount();
		//eam.Coeff()[i] = 1.0 / (i + 1);
		eam_exact.Coeff()[i] = 0.6*sin(i+1);
		eam.Coeff()[i] = 0.1;
	}

	NonLinearRegression trainer(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

	Configuration cfg;
	vector<Configuration> training_set;
	ifstream ifs1(PATH+"NaCl_small.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs1); i++) {
		//for (int i = 0; i < 1; i++) {
		//cfg.load(ifs1);
		eam_exact.CalcEFS(cfg);
		//cout << cfg.energy << endl;
		//for (int j = 0; j < cfg.size(); j++)
		//	cout << cfg.force(j)[0] << " " << cfg.force(j)[1] << " " << cfg.force(j)[2] << endl;
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}
	ifs1.close();

	trainer.Train(training_set);

	// comparing
	double max_en_difference = 0;
	for (const Configuration& cfg_orig : training_set) {
		cfg = cfg_orig;
		eam.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
		//std::cout << "energy: " << std::endl;
		//std::cout << cfg.energy << " " << cfg_orig.energy << " ";
		//std::cout << fabs((cfg.energy - cfg_orig.energy)/cfg_orig.energy) << std::endl;
		//std::cout << "forces: " << std::endl;
		//for (int ii = 0; ii < cfg_orig.size(); ii++) {
		//	std::cout << cfg.force(ii)[0] << " " << cfg_orig.force(ii)[0] << " ";
		//	std::cout << cfg.force(ii)[1] << " " << cfg_orig.force(ii)[1] << " ";
		//	std::cout << cfg.force(ii)[2] << " " << cfg_orig.force(ii)[2] << std::endl;
		//}
		//std::cout << "stresses: " << std::endl;
		//for (int a = 0; a < 3; a++)
		//	for (int b = 0; b < 3; b++)
		//		std::cout << cfg.stresses[a][b] << " " << cfg_orig.stresses[a][b] << std::endl;
	}
	if (max_en_difference > 0.02) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}

} END_TEST;

PAR_TEST("Lennard-Jones fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	LennardJones LJexact(PATH+"unlearned_lj_lin.mlip", 1.6);
	LennardJones LJ(PATH+"unlearned_lj_lin.mlip", 1.6);

	for (int i = 0; i < LJ.CoeffCount(); i++) {
		LJexact.Coeff()[i] = sin(i) / 5;
		//LJexact.Coeff()[i] = 0.1 * i / LJ.CoeffCount();
		//LJ.Coeff()[i] = 1.0 / (i + 1);
		LJ.Coeff()[i] = 5;
	}

	NonLinearRegression trainer(&LJ, 1.0, 1.0, 1.0, 0.0, 1.0);

	vector<Configuration> training_set;
	Configuration cfg;
	ifstream ifs1(PATH+"NaCl_small.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs1); i++) {
		//for (int i = 0; i < 1; i++) {
		//cfg.load(ifs1);
		LJexact.CalcEFS(cfg);
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}
	ifs1.close();

	trainer.Train(training_set);

	// comparing
	double max_en_difference = 0;
	for (const Configuration& cfg_orig : training_set) {
		cfg = cfg_orig;
		LJ.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs(cfg.energy - cfg_orig.energy));
		//std::cout << "energy: " << std::endl;
		//std::cout << cfg.energy << " " << LJ.train_set[i].energy << " ";
		//std::cout << fabs((cfg.energy - LJ.train_set[i].energy)/LJ.train_set[i].energy) << std::endl;
		//std::cout << "forces: " << std::endl;
		//for (int ii = 0; ii < LJ.train_set[i].size(); ii++) {
		//	std::cout << cfg.force(ii)[0] << " " << LJ.train_set[i].force(ii)[0] << " ";
		//	std::cout << cfg.force(ii)[1] << " " << LJ.train_set[i].force(ii)[1] << " ";
		//	std::cout << cfg.force(ii)[2] << " " << LJ.train_set[i].force(ii)[2] << std::endl;
		//}
		//std::cout << "stresses: " << std::endl;
		//for (int a = 0; a < 3; a++)
		//	for (int b = 0; b < 3; b++)
		//		std::cout << cfg.stresses[a][b] << " " << LJ.train_set[i].stresses[a][b] << std::endl;
	}
	if (max_en_difference > 0.02) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}
} END_TEST;

PAR_TEST("NonlinearLennard-Jones fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	NonlinearLennardJones LJexact(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);
	NonlinearLennardJones LJ(PATH+"unlearned_lj_nonlin.mlip", 1.6, 5);

	for (int i = 0; i < LJ.CoeffCount(); i++) {
		LJexact.Coeff()[i] = 2*i;
		//LJexact.Coeff()[i] = 0.1 * i / LJ.CoeffCount();
		//LJ.Coeff()[i] = 1.0 / (i + 1);
		LJ.Coeff()[i] = 1.5*fabs(sin(i));
	}

	NonLinearRegression trainer(&LJ, 100.0, 1.0, 0.1, 0.0, 0.0);

	vector<Configuration> training_set;
	Configuration cfg;
	ifstream ifs1(PATH+"NaCl_small.cfgs", ios::binary);
	for (int i = 0; cfg.Load(ifs1); i++) {
		//for (int i = 0; i < 1; i++) {
		//cfg.load(ifs1);
		LJexact.CalcEFS(cfg);
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}
	ifs1.close();

	trainer.Train(training_set);

	// comparing
	double max_en_difference = 0;
	int i = 0;
	for (const Configuration& cfg_orig : training_set) {
		cfg = cfg_orig;
		LJ.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs(cfg.energy - cfg_orig.energy));
		//std::cout << "energy: " << std::endl;
		//std::cout << cfg.energy << " " << training_set[i].energy << " ";
		//std::cout << fabs((cfg.energy - training_set[i].energy)/training_set[i].energy) << std::endl;
		//std::cout << "forces: " << std::endl;
		//for (int ii = 0; ii < training_set[i].size(); ii++) {
		//	std::cout << cfg.force(ii)[0] << " " << training_set[i].force(ii)[0] << " ";
		//	std::cout << cfg.force(ii)[1] << " " << training_set[i].force(ii)[1] << " ";
		//	std::cout << cfg.force(ii)[2] << " " << training_set[i].force(ii)[2] << std::endl;
		//}
		//std::cout << "stresses: " << std::endl;
		//for (int a = 0; a < 3; a++)
		//	for (int b = 0; b < 3; b++)
		//		std::cout << cfg.stresses[a][b] << " " << training_set[i].stresses[a][b] << std::endl;
		i++;
	}
	if (max_en_difference > 0.02) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}
} END_TEST;

PAR_TEST("MTPR fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	MLMTPR mtpr_trained(PATH+"learned.mtp");

	for (int i = 0; i < mtpr_trained.CoeffCount(); i++)
		mtpr_trained.Coeff()[i] = 1e-2*rand() / RAND_MAX;

	MLMTPR mtpr_fit(PATH+"unlearned.mtp");
	//std::cout << mtpr_fit.CoeffCount() << std::endl;
	MTPR_trainer nlr(&mtpr_fit, 1, 0.1, 1e-3, 0, 1e-8);

	ifstream ifs(PATH+"NaCl_small.cfgs", ios::binary);

	Configuration cfg;
	vector<Configuration> training_set;
	for (int i = 0; cfg.Load(ifs); i++) {
		mtpr_trained.CalcEFS(cfg);
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}

	ifs.close();

	nlr.wgt_eqtn_energy = 1;
	nlr.wgt_eqtn_forces = 1;
	nlr.wgt_eqtn_constr = 1e-6;

	nlr.Train(training_set);

	// comparing
	double max_en_difference = 0;
	for (Configuration& conf : training_set) {
		cfg = conf;
		mtpr_fit.CalcEFS(cfg);
		max_en_difference = std::max(max_en_difference,
			fabs(cfg.energy - conf.energy));

	}
	if (max_en_difference > 1e-2) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}
} END_TEST;

/*PAR_TEST("MTPR2 fitting test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	MLMTPR2 mtpr_trained(PATH+"unlearned.mtp");

	for (int i = 0; i < mtpr_trained.CoeffCount(); i++)
		mtpr_trained.Coeff()[i] = 4e-2*rand() / RAND_MAX;

	MLMTPR2 mtpr_fit(PATH+"unlearned.mtp");

	for (int i = 0; i < mtpr_fit.CoeffCount(); i++)
		mtpr_fit.Coeff()[i] = sin(i);
	//std::cout << mtpr_fit.CoeffCount() << std::endl;
	MTPR_trainer2 nlr(&mtpr_fit, 1, 0.1, 1e-3, 0, 1e-8);

	ifstream ifs(PATH+"2comp.cfgs", ios::binary);

	Configuration cfg;
	vector<Configuration> training_set;
	for (int i = 0; cfg.Load(ifs); i++) {
		mtpr_trained.CalcEFS(cfg);
		if (i % mpi_size == mpi_rank) 
			training_set.push_back(cfg);
	}

	ifs.close();

	nlr.wgt_eqtn_energy = 1;
	nlr.wgt_eqtn_forces = 1;
	nlr.wgt_eqtn_constr = 1e-6;

	nlr.Train(training_set);

	// comparing
	double max_en_difference = 0;
	for (Configuration& conf : training_set) {
		cfg = conf;
		mtpr_fit.CalcEFS(cfg);
		//cout << cfg.energy << " " << conf.energy << endl;
		max_en_difference = std::max(max_en_difference,
			fabs(cfg.energy - conf.energy));

	}
	if (max_en_difference > 1e-2) {
		std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
		FAIL()
	}
} END_TEST;*/

PAR_TEST("MTPR loss gradient test") {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	MLMTPR mtpr(PATH+"learned.mtp");
	MTPR_trainer nlr(&mtpr, 1, 1, 0);
	ifstream ifs(PATH+"2comp.cfgs", ios::binary);
	Configuration cfg;
	int fail = 0;

	//if (mpi_rank == 0) cout << mtpr.CoeffCount() << " " << mtpr.alpha_count << endl;

	for (int i = 0; cfg.Load(ifs); i++)
		if (i % mpi_size == mpi_rank)
		{
			if (!nlr.CheckLossConsistency_debug(cfg, 1e-8))
				fail = 1;
		}
	ifs.close();

	int failed_any = 0;

	MPI_Reduce(&fail, &failed_any, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	if (failed_any > 0) {
		std::cerr << "Somewhere the gradient is wrong";
		FAIL()
	}



} END_TEST;
#endif
	return true;
}
#endif //MLIP_NO_SELFTEST


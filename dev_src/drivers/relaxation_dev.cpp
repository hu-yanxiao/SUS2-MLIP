/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */


#include "relaxation_dev.h"
#include <iostream>


using namespace std;


void RelaxationDev::PassEFStoBFGS()
{
	const double vol = fabs(cfg.lattice.det());
	bfgs_f = cfg.energy + pressure*vol;

	if (relax_cell) {
		const Matrix3 Stress = cfg.stresses + Matrix3::Id()*pressure*vol;
		const Matrix3 dEdL = cfg.lattice.inverse().transpose() * Stress;

		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				bfgs_g[3 * a + b] = dEdL[a][b];
	}

	if (relax_atoms) {
		for (int i = 0; i < cfg.size(); i++) {
			Vector3 frac_force = cfg.lattice * cfg.force(i);
			for (int a = 0; a < 3; a++)
				bfgs_g[pos_offset + 3 * i + a] = -frac_force[a];
		}
	}
}

void RelaxationDev::GetCfgfromBFGS()
{
	max_deformation = 0.0;
	if (relax_cell) {
		const Matrix3 deformation = cfg.lattice.inverse() * Matrix3(
			bfgs.x(0), bfgs.x(1), bfgs.x(2),
			bfgs.x(3), bfgs.x(4), bfgs.x(5),
			bfgs.x(6), bfgs.x(7), bfgs.x(8));
		max_deformation = (deformation - deformation.Id()).MaxAbs();
		if (relax_atoms) cfg.lattice *= deformation;
		else cfg.deform(deformation);
	}

	max_atom_travel = 0.0;
	if (relax_atoms) {
		const Matrix3 LT(cfg.lattice.transpose());
		for (int i = 0; i<cfg.size(); i++) {
			const Vector3 vec = LT * Vector3(bfgs.x(pos_offset + 3 * i + 0), bfgs.x(pos_offset + 3 * i + 1), bfgs.x(pos_offset + 3 * i + 2));
			max_atom_travel = __max(max_atom_travel, (cfg.pos(i)-vec).Norm());
			cfg.pos(i) = vec;
		}
	}
}

void RelaxationDev::InitBFGSWithCfg()
{
	// filling temp_x, then calling bfgs.set_x(temp_x)
	if (relax_cell) {
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				temp_x[3*a+b] = cfg.lattice[a][b];
	}

	if (relax_atoms) {
		const Matrix3 invLT(cfg.lattice.inverse().transpose());
		for (int i = 0; i<cfg.size(); i++) {
			const Vector3 frac_coord = invLT*cfg.pos(i);
			for (int a = 0; a < 3; a++)
				temp_x[pos_offset + 3 * i + a] = frac_coord[a];
		}
	}
	bfgs.Set_x(&temp_x[0], (int)temp_x.size());

	// setting the initial hessian
	const Matrix3 LLT = cfg.lattice * cfg.lattice.transpose();
	// setting force constants;
	if (relax_atoms) {
		const double force_const = 10; // typical force constant in eV/A^2
		const Matrix3 force_const_inv = (1.0 / force_const) * LLT.inverse();
		for (int i = 0; i < cfg.size(); i++) {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					bfgs.inv_hess(pos_offset + 3 * i + a, pos_offset + 3 * i + b)
					= force_const_inv[a][b];
		}
	}
	// setting the elastic constants
	if (relax_cell) {
		const double el_const = 500; // typical (slightly overestimated) el. constant in GPa
		const Matrix3 el_const_inv = (cfg.lattice.transpose() * cfg.lattice)
			* (160.21 / el_const / std::abs(cfg.lattice.det()));
		for (int i = 0; i < 3; i++) {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					bfgs.inv_hess(3 * i + a, 3 * i + b) = el_const_inv[a][b];
		}
	}
}

RelaxationDev::RelaxationDev(AnyPotential* _pot, const Configuration& _cfg, bool _relax_atoms, bool _relax_cell)
	: relax_atoms(_relax_atoms), relax_cell(_relax_cell)
{
	pos_offset = relax_cell ? 9 : 0;
	temp_x.resize(pos_offset + (relax_atoms ? 3*_cfg.size() : 0));
	bfgs_g.resize(pos_offset + (relax_atoms ? 3 * _cfg.size() : 0));
	pot = _pot;
	cfg = _cfg;
}

RelaxationDev::RelaxationDev(AnyPotential* _pot, const Configuration& _cfg, map<string, string> _settings)
{
}

RelaxationDev::~RelaxationDev()
{
}

void RelaxationDev::Run(int max_itr_cnt)
{
	cfg_count++;
	// cfg.CorrectSupercell();
	cfg.MoveAtomsIntoCell();

	const double min_step = 1e-8;

	double curr_mindist = cfg.MinDist();
	if (curr_mindist < mindist_limit + min_step) {
		Warning("Relaxation: Minimal distance in intial configuration is too short. "
			+ to_string(curr_mindist) + ". Ignoring configuration");
		cfg.energy = 9e99;
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				cfg.stresses[a][b] = 0;
		for (int i = 0; i < cfg.size(); i++)
			cfg.force(i) = Vector3(0, 0, 0);
		iterations_count = max_itr_cnt;
		return;
	}

	if (max_itr_cnt < 1)
		return;

	// initalization of BFGS argument array 'x' and 'inv_hess'
	InitBFGSWithCfg();

	// Run main loop
	for (iterations_count = 0; (iterations_count < max_itr_cnt); iterations_count++) {
		Configuration cfg_temp(cfg);
		cfg.MoveAtomsIntoCell();
		pot->CalcEFS(cfg);
		memcpy(&cfg.pos(0), &cfg_temp.pos(0), sizeof(double) * 3 * cfg.size());

		// check convergence
		bool converged = true;
		double max_force = 0;
		if (relax_atoms) {
			for (int i = 0; i < cfg.size(); i++)
				max_force = __max(max_force, cfg.force(i).Norm());
			if (max_force > tol_force) converged = false;
		}
		double max_stress = 0; // max stress in GPa
		if (relax_cell) {
			max_stress = cfg.stresses.MaxAbs() * (160.21766208 / std::abs(cfg.lattice.det()));
			if (max_stress > tol_stress) converged = false;
		}
		if (converged) break;
		//std::cout << "Relax " << cfg.features["cfg_id"]
		//	<< ": max_stress = " << max_stress
		//	<< ": grade = " << cfg.features["MV_grade"]
		//	<< ": mindist = " << cfg.MinDist()
		//	<< "\n";

		// do an iteration
		PassEFStoBFGS();
		bfgs.Iterate(bfgs_f, bfgs_g);
		GetCfgfromBFGS();
		// check maximum distance traveled
		while (cfg.MinDist() < mindist_limit
			|| max_deformation > maxstep_lattice
			|| max_atom_travel > maxstep_atoms
			) {
			bfgs.ReduceStep();
			GetCfgfromBFGS();
		}
	}
}

void RelaxationDev::PrintDebug(const char * fnm)	// for testing purposes
{
	std::ofstream ofs(fnm, std::ios::app);
	cfg.MoveAtomsIntoCell();
	for (int i = 0; i < cfg.size(); i++)
		ofs << cfg.pos(i, 0) << ' ' << cfg.pos(i, 1) << ' ' << cfg.pos(i, 2) << '\t';
	ofs << std::endl;
	ofs.close();
}

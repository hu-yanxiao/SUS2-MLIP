/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */


#include "outer_relaxation.h"
#include <iostream>


using namespace std;


void OuterNBHRelaxation::EFStoFuncGrad()
{
	const double vol = fabs(cfg.lattice.det());
	f = cfg.energy + pressure*vol;

	{
		const Matrix3 Stress = cfg.stresses + Matrix3::Id()*pressure*vol;
		const Matrix3 dEdL = cfg.lattice.inverse().transpose() * Stress;

		for (int a=0; a<3; a++)
			for (int b=0; b<3; b++)
				g[3*a + b] = dEdL[a][b];
	}

	{
		for (int i=0; i<cfg.size(); i++) 
			if (ignored_atom_inds.count(i) == 0)
			{
				Vector3 frac_force = cfg.lattice * cfg.force(i);
				for (int a=0; a<3; a++)
					g[pos_offset + 3*i+a] = -frac_force[a];
			}
	}
}

void OuterNBHRelaxation::XtoLatPos()
{
	max_deformation = 0.0;
	{
		const Matrix3 deformation = cfg.lattice.inverse() * 
									Matrix3(x[0], x[1], x[2],
											x[3], x[4], x[5],
											x[6], x[7], x[8]);
		max_deformation = (deformation - deformation.Id()).MaxAbs();
		cfg.deform(deformation);
	}

	max_atom_travel = 0.0;
	{
		const Matrix3 LT(cfg.lattice.transpose());
		for (int i=0; i<cfg.size(); i++) 
			if (ignored_atom_inds.count(i) == 0) 
			{
				const Vector3 vec = LT * Vector3(	x[pos_offset + 3*i + 0], 
													x[pos_offset + 3*i + 1], 
													x[pos_offset + 3*i + 2]	);
				max_atom_travel = __max(max_atom_travel, (cfg.pos(i)-vec).Norm());
				cfg.pos(i) = vec;
			}
	}
}

void OuterNBHRelaxation::InitBFGSWithCfg()
{
	{
		for (int a=0; a<3; a++)
			for (int b=0; b<3; b++)
				x[3*a+b] = cfg.lattice[a][b];
	}

	{
		const Matrix3 invLT(cfg.lattice.inverse().transpose());
		for (int i = 0; i<cfg.size(); i++) 
			if (ignored_atom_inds.count(i) == 0)
			{
				const Vector3 frac_coord = invLT*cfg.pos(i);
				for (int a = 0; a < 3; a++)
					x[pos_offset + 3 * i + a] = frac_coord[a];
			}
	}

	// setting the initial hessian
	const Matrix3 LLT = cfg.lattice * cfg.lattice.transpose();
	// setting force constants;
	{
		const double force_const = 10; // typical force constant in eV/A^2
		const Matrix3 force_const_inv = (1.0 / force_const) * LLT.inverse();
		for (int i=0; i<cfg.size(); i++) 
			if (ignored_atom_inds.count(i) == 0)
			{
				for (int a=0; a<3; a++)
					for (int b=0; b<3; b++)
						bfgs.inv_hess(pos_offset + 3*i + a, pos_offset + 3*i + b) =
							force_const_inv[a][b];
			}
	}
	// setting the elastic constants
	{
		const double el_const = 500; // typical (slightly overestimated) el. constant in GPa
		const Matrix3 el_const_inv = (cfg.lattice * cfg.lattice.transpose()) *
									 (160.21 / el_const / cfg.lattice.det());
		for (int i = 0; i < 3; i++) {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					bfgs.inv_hess(3 * i + a, 3 * i + b) = el_const_inv[a][b];
		}
	}
}

OuterNBHRelaxation::OuterNBHRelaxation(	AnyPotential* _pot, 
										const Configuration& _cfg, 
										int _fix_nbh_ind, 
										double cutoff )
	: fix_nbh_ind(_fix_nbh_ind)
{
	pot = _pot;
	cfg = _cfg;
	Neighborhoods nbh(cfg, cutoff);
	for (int ind : nbh[fix_nbh_ind].inds)
		ignored_atom_inds.insert(ind);
	if (cfg.size() <= nbh[fix_nbh_ind].count)
		ERROR("No atoms for relaxation");
	x.resize(cfg.size() + 9);
	cfg.CorrectSupercell();
	double curr_mindist = cfg.MinDist();
	if (curr_mindist < min_dist) 
		ERROR("Relaxation error: Minimal distance in intial configuration is too short. " +
				to_string(curr_mindist) + ". Ignoring configuration");
	InitBFGSWithCfg();
	bfgs.Set_x(x.data(), (int)x.size());
}

void OuterNBHRelaxation::Run(int max_itr_cnt)
{
	const double min_step = 1e-8;

	// Run main loop
	for (iter=0; iter<max_itr_cnt; iter++) 
	{
		pot->CalcEFS(cfg);

		
		{// check convergence
			double max_force = 0;
			{
				for (int i=0; i<cfg.size(); i++)
					if (ignored_atom_inds.count(i) == 0)
						max_force = __max(max_force, cfg.force(i).Norm());
				std::cout << max_force << ' ' << sqrt(cfg.stresses.NormFrobeniusSq()) << endl;
			}
			double max_stress = 0; // max stress in GPa
			max_stress = cfg.stresses.MaxAbs() * (160.21766208 / cfg.lattice.det());

			if (max_force < tol_force && max_stress < tol_stress)
				break;
		}

		// do an iteration
		EFStoFuncGrad();
		x = bfgs.Iterate(f,g);
		XtoLatPos();
		// check maximum distance traveled
		while (	cfg.MinDist() < min_dist ||
				max_deformation > maxstep_lattice ||
				max_atom_travel > maxstep_atoms	) 
		{
			bfgs.ReduceStep(0.5);
			XtoLatPos();
		}
	}
}

void OuterNBHRelaxation::PrintDebug(const char * fnm)	// for testing purposes
{
	std::ofstream ofs(fnm, std::ios::app);
	for (int i = 0; i < cfg.size(); i++)
		ofs << cfg.pos(i, 0) << ' ' << cfg.pos(i, 1) << ' ' << cfg.pos(i, 2) << '\t';
	ofs << std::endl;
	ofs.close();
}

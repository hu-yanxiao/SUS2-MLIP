/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "basic_potentials.h"

void AnyPotential::ResetEFS(Configuration & cfg) //!< Declaring and cleaning EFS
{
	cfg.has_energy(true);
	cfg.has_forces(true);
	cfg.has_stresses(true);
	cfg.energy = 0.0;
	memset(&cfg.force(0, 0), 0, cfg.size() * sizeof(Vector3));
	//memset(&cfg.cal_se[0], 0, cfg.size() * sizeof(std::vector<double>));
	memset(&cfg.stresses[0][0], 0, sizeof(Matrix3));
}

// Cheks equality f_i = -df/dr_i by finite differences. Returns true if deviation between FD-calculated and exact force is less than control_delta, 

bool AnyPotential::CheckEFSConsistency_debug(Configuration cfg, double displacement, double control_delta)
{
	Configuration cfg1, cfg2;
	double err_max = 0.0;

	CalcEFS(cfg);

	//std::cout << "forces: " << std::endl;
	// f_i = -dE/dr_i check
	for (int i = 0; i < cfg.size(); i++)
	{
		Vector3 fd_force;
		for (int a=0; a<3; a++)
		{
			cfg1 = cfg;
			cfg2 = cfg;
			cfg1.pos(i)[a] += displacement;
			cfg2.pos(i)[a] -= displacement;
			CalcEFS(cfg1);
			CalcEFS(cfg2);
			fd_force[a] = (cfg2.energy - cfg1.energy) / (2 * displacement);
			//if (fabs(fd_force[a] - cfg.force(i)[a]) > control_delta)
				//std::cout << fd_force[a] << " " << cfg.force(i)[a] << " " << fabs(fd_force[a] - cfg.force(i)[a]) << " " << i << " " << a << std::endl; 
		}
		double err = (cfg.force(i)-fd_force).Norm();
		err_max = __max(err, err_max);
	}

	//cout << err_max << endl;
	//if (err_max > control_delta)
	//	return false;

	// L^{-T} * stresses = dE/dL check
	//std::cout << "stresses: " << std::endl;
	Matrix3 dEdL = -cfg.lattice.inverse().transpose() * cfg.stresses;
	Matrix3 fd_dEdL(0,0,0,0,0,0,0,0,0);
	for (int a = 0; a < 3; a++) {
		for (int b = 0; b < 3; b++)
		{
			cfg1 = cfg;
			cfg2 = cfg;

			Matrix3 deform = cfg.lattice;

			deform[a][b] += displacement;
			cfg1.Deform(cfg1.lattice.inverse()*deform);

			deform[a][b] -= 2*displacement;
			cfg2.Deform(cfg2.lattice.inverse()*deform);

			CalcEFS(cfg1);
			CalcEFS(cfg2);

			fd_dEdL[a][b] = (cfg1.energy-cfg2.energy) / (cfg1.lattice[a][b]-cfg2.lattice[a][b]);
			//std::cout << dEdL[a][b] << " " << fd_dEdL[a][b] << std::endl;
		}
	}
	
	err_max = __max(sqrt((fd_dEdL - dEdL).NormFrobeniusSq()), err_max);

	return (err_max < control_delta);
}

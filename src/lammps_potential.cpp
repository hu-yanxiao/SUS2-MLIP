/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */


#include "lammps_potential.h"


using namespace std;


void LAMMPS_potential::CalcEFS(Configuration & conf)
{
	Matrix3 Q;
	Matrix3 L;
	Configuration cfg = conf;

	cfg.lattice.transpose().QRdecomp(Q, L);
	cfg.Deform(Q);

	ofstream ofs(input_filename.c_str());

	ofs.precision(16); ofs.setf(ios::scientific);

	ofs << "ITEM: TIMESTEP \n"
		<< "0\n"
		<< "ITEM: NUMBER OF ATOMS \n"
		<< cfg.size() << '\n'
		<< "ITEM: BOX BOUNDS xy xz yz pp pp pp \n"
		<< 0 <<' '<< cfg.lattice[0][0] + cfg.lattice[1][0] + cfg.lattice[2][0] <<' '<< cfg.lattice[1][0] << '\n'
		<< 0 <<' '<< cfg.lattice[1][1] + cfg.lattice[2][1] <<' '<< cfg.lattice[2][0] << '\n'
		<< 0 <<' '<< cfg.lattice[2][2] <<' '<< cfg.lattice[2][1] << '\n'
		<< "ITEM: ATOMS id x y z \n";
	for (int i=0; i<cfg.size(); i++)
		ofs << i+1 << ' '
			<< cfg.pos(i, 0) << ' '
			<< cfg.pos(i, 1) << ' '
			<< cfg.pos(i, 2) << " \n";
	ofs.close();

	// starting LAMMPS
	int res = system(("rm " + output_filename).c_str());
	res = system(start_command.c_str());
	if (res != 0)
		Message("LAMMPS exits with code " + to_string(res));

	//try
	//{
	ifstream ifs(output_filename);
	if (!ifs.is_open())
		ERROR("Can't open file " + output_filename);

	int foo;
	char buff[999];
	ifs.getline(buff, 999, '\n');
	if (strcmp(buff, "ITEM: TIMESTEP"))
		ERROR("Invalid file format. \"ITEM: TIMESTEP\" not fount");
	ifs >> foo;
	if (foo != 0)
		ERROR("TIMESTEP should be 0");
	ifs.ignore(99999, '\n');

	ifs.getline(buff, 999, '\n');
	if (strcmp(buff, "ITEM: NUMBER OF ATOMS"))
		ERROR("Invalid file format. \"ITEM: NUMBER OF ATOMS\" not fount");
	ifs >> foo;
	if (foo != cfg.size())
		ERROR("Atoms count changed after LAMMPS calculation");
	ifs.ignore(99999, '\n');

	ifs.getline(buff, 999, '\n');
	if (strcmp(buff, "ITEM: BOX BOUNDS xy xz yz pp pp pp"))
		ERROR("Invalid file format. \"ITEM: BOX BOUNDS xy xz yz pp pp pp\" not fount");
	double bar;
	ifs >> bar >> cfg.lattice[0][0] >> cfg.lattice[1][0];
	cfg.lattice[0][0] -= bar;
	ifs >> bar >> cfg.lattice[1][1] >> cfg.lattice[2][0];
	cfg.lattice[1][1] -= bar;
	ifs >> bar >> cfg.lattice[2][2] >> cfg.lattice[2][1];
	cfg.lattice[2][2] -= bar;
	cfg.lattice[0][0] -= cfg.lattice[1][0] + cfg.lattice[2][0];
	cfg.lattice[1][1] -= cfg.lattice[2][1];
	ifs.ignore(99999, '\n');

	ifs.getline(buff, 999, '\n');
	if (strcmp(buff, "ITEM: ATOMS id x y z fx fy fz "))
		ERROR("Invalid file format. \"ITEM: ATOMS id x y z fx fy fz \" not fount");
	cfg.has_forces(true);
	for (int i=0; i<cfg.size(); i++)
		ifs >> foo
		>> cfg.pos(i, 0) >> cfg.pos(i, 1) >> cfg.pos(i, 2)
		>> cfg.force(i, 0) >> cfg.force(i, 1) >> cfg.force(i, 2);
	ifs.ignore(99999, '\n');

	ifs.getline(buff, 999, '\n');
	if (strcmp(buff, "ITEM: ENERGY"))
		ERROR("Invalid file format. \"ITEM: ENERGY\" not fount");
	cfg.has_energy(true);
	ifs >> cfg.energy;
	ifs.ignore(99999, '\n');

	ifs.getline(buff, 999, '\n');
	if (!ifs.eof() && !strcmp(buff, "ITEM: STRESSES xx yy zz xy xz yz"))
	{
		cfg.has_stresses(true);
		ifs >> cfg.stresses[0][0] >> cfg.stresses[1][1] >> cfg.stresses[2][2]
			>> cfg.stresses[0][1] >> cfg.stresses[0][2] >> cfg.stresses[1][2];

		cfg.stresses[1][0] = cfg.stresses[0][1];
		cfg.stresses[2][0] = cfg.stresses[0][2];
		cfg.stresses[2][1] = cfg.stresses[1][2];

		cfg.stresses *= fabs(cfg.lattice.det()) * 0.0001 / 160.2176487;
	}

	ifs.close();
	//}
	//catch (MlipException& excp)
	//{
	//	Message(excp.What());
	//	ERROR("LAMMPS output file reading failed");
	//}

	cfg.Deform(Q.transpose());

	if (cfg.has_forces())
		for (int i=0; i<cfg.size(); i++)
			cfg.force(i) = cfg.force(i) * Q;

	if (cfg.has_stresses())
		cfg.stresses = Q.transpose() * cfg.stresses * Q;

	conf = cfg;

	conf.features["EFS_by"] = "LAMMPS";
}

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../../src/configuration.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cctype>

#ifdef MLIP_MPI
	#include <mpi.h>
#endif

using namespace std;

void LoadNextFromDump(Configuration& cfg, ifstream& ifs)
{
	string line;	

	ifs >> line;

	while (line.substr(0,21) != "ITEM: NUMBER OF ATOMS") {
		getline(ifs, line);
	}

	if (line.substr(0,21) == "ITEM: NUMBER OF ATOMS") {
		int size;
		ifs >> size;
		cfg.resize(size);
	}

	while (line.substr(0,35) != "ITEM: BOX BOUNDS xy xz yz pp pp pp") {
		getline(ifs, line);
	}

	double lo_bound[3];
	double hi_bound[3];
	double lo[3];
	double hi[3];
	double xy, xz, yz;

	if (line.substr(0,35) == "ITEM: BOX BOUNDS xy xz yz pp pp pp") {
		ifs >> lo_bound[0] >> hi_bound[0] >> xy;
		getline(ifs, line);
		ifs >> lo_bound[1] >> hi_bound[1] >> xz;
		getline(ifs, line);
		ifs >> lo_bound[2] >> hi_bound[2] >> yz;

		lo[0] = lo_bound[0] - __min(__min(0.0,xy), __min(xz,xy+xz));
		hi[0] = hi_bound[0] - __max(__max(0.0,xy), __max(xz,xy+xz));
		lo[1] = lo_bound[1] - __min(0.0,yz);
		hi[1] = hi_bound[1] - __max(0.0,yz);
		lo[2] = lo_bound[2];
		hi[2] = hi_bound[2];

		for (int a = 0; a < 3; a++)
			cfg.lattice[a][a] = hi[a] - lo[a];

		cfg.lattice[1][0] = xy;
		cfg.lattice[2][0] = xz;
		cfg.lattice[2][1] = yz;
		cfg.lattice[0][1] = 0;
		cfg.lattice[0][2] = 0;
		cfg.lattice[1][2] = 0;	

		getline(ifs, line);	
	}

	while (line.substr(0,11) != "ITEM: ATOMS") {
		getline(ifs, line);
	}

	if (line.substr(0,11) == "ITEM: ATOMS") {
		for (int i = 0; i < cfg.size(); i++) {
			int id, type;
			double pos[3];
			ifs >> id >> type >> pos[0] >> pos[1] >> pos[2];
			cfg.type(id-1) = type-1;
			for (int a = 0; a < 3; a++) {
				cfg.pos(id-1,a) = pos[a];
			}
			getline(ifs,line);
		}
	}	
} 

void LoadDatabaseFromLAMMPSMD(const char * filename1, const char * filename2, const int n_steps)
{
	ifstream ifs(filename1);
	
	try {
		if (!ifs.is_open()) throw 1;	
	}
	catch(...) {
		ERROR((std::string)"Can't open file \"" + filename1 + "\" for reading");
	}	

	ofstream ofs(filename2);

	for (int i = 0; i <= n_steps; i++) {
		Configuration cfg;
		LoadNextFromDump(cfg, ifs);
		cfg.MoveAtomsIntoCell();
		cfg.Save(ofs);
	}
}

int main(int argc, char* argv[])
{
	try {

		MPI_Init(&argc, &argv);
		int mpi_rank;
		int mpi_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

		Configuration cfg;

		LoadDatabaseFromLAMMPSMD("MD_LJ/1.dump", "MD_LJ/test.cfgs", stoi(argv[1]));

		MPI_Finalize();

	}
    catch(MlipException& excp) {
            cout << excp.What() << endl;
    }


	return 0;
}

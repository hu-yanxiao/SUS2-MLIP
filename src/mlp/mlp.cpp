/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include "../common/stdafx.h"
#include "train.h"
#include "calc_errors.h"
#include "self_test.h"
#include "mlp.h"
#include "mlp_commands.h"


int ExecuteCommand(const std::string& command,
	std::vector<std::string>& args,
	std::map<std::string, std::string>& opts)
{
	bool is_command_not_found = true;
	is_command_not_found = !Commands(command, args, opts);

	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	if (is_command_not_found)
		if (mpi_rank == 0) std::cout << "Error: command " << command << " does not exist." << std::endl;

	return 1;
}


int main(int argc, char *argv[])
{
#ifdef MLIP_MPI
	MPI_Init(&argc, &argv);
#endif

	std::vector<std::string> args;
	std::map<std::string, std::string> opts;

	// parse argv to extract args and opts
	std::string command = "help";
	if (argc >= 2) command = argv[1];
	try {
		ParseOptions(argc-1, argv+1, args, opts);
		ExecuteCommand(command, args, opts);
	}
	catch (const MlipException& e) { std::cerr << e.What(); return 1; }

#ifdef MLIP_MPI
	MPI_Finalize();
#endif

	return 0;
}

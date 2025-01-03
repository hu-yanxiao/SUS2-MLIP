/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *   This file contributors: Ivan Novikov, Konstantin Gubaev
 */


#ifdef MLIP_MPI
#	include <mpi.h>
#endif
#include "../../src/common/stdafx.h"
#include "dev_self_test.h"
#include "../../src/mlp/self_test.h"
#include "../../src/mlp/mlp.h"
#include "../../src/mlp/train.h"
#include "../../src/mlp/calc_errors.h"
#include "../../src/mlp/mlp_commands.h"
#include "../../dev_src/mlp/dev_mlp_commands.h"

int ExecuteCommand(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	bool is_command_found = false;

	is_command_found = Commands(command, args, opts) || is_command_found;

	if(!is_command_found || command == "list" || (command == "help" && args.size()==0))
		is_command_found = DevCommands(command, args, opts) || is_command_found;

	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	if (!is_command_found)
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

	try {
		int iarg = ParseCommonOptions(argc, argv);
		if ( 0 > iarg || false == CheckCommonOptions() ) return 2;
		// parse argv to extract args and opts
		std::string command = "help";
		if ( argc > iarg ) command = argv[iarg];

		ParseOptions(argc - iarg, argv + iarg, args, opts);
		ExecuteCommand(command, args, opts);
	}
	catch (const MlipException& e) { std::cerr << e.What(); return 1; }

#ifdef MLIP_MPI
	MPI_Finalize();
#endif

	return 0;
}

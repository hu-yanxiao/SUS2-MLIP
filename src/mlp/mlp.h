/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_UTILS_MLP_H
#define MLIP_UTILS_MLP_H

#include <iostream>
#include <string>

inline void PrintSUS2Banner(std::ostream& os, bool mpi_enabled, int mpi_size)
{
	os
		<< "  ____  _   _ ____  ____  \n"
		<< " / ___|| | | / ___||___ \\ \n"
		<< " \\___ \\| | | \\___ \\  __) |\n"
		<< "  ___) | |_| |___) |/ __/ \n"
		<< " |____/ \\___/|____/|_____|\n"
		<< '\n'
		<< "SUS2-MLIP Developer Version (2026-04-17)\n"
		<< "Branch: codex/developer\n"
		<< "Period Tag: 2026-04-17\n"
		<< "Author: Hu Yanxiao | SUSTech\n";

	if (mpi_enabled)
		os << "Mode: MPI (" << mpi_size << " ranks)\n";
	else
		os << "Mode: Serial\n";

	os
		<< "Please cite: https://doi.org/10.1073/pnas.2503439122\n"
		<< std::endl;
}

const std::string USAGE = \
"Usage:\n"
"mlp-sus2 help                 prints this message\n"
"mlp-sus2 list                 lists all the available commands\n"
"mlp-sus2 help [command]       prints the description of the command\n"
"mlp-sus2 [command] [options]  executes the command (with the options)\n";

#define BEGIN_COMMAND(command_name, descr, usage) \
	if(command == "list") \
		std::cout << "    "  << command_name << ": " << descr << '\n'; \
	else if((args.size()==1) && command == "help" && (args[0] == command_name)) \
		{std::cout << "Usage: \n" << usage; is_command_found = true; } \
	else if (command == command_name) { is_command_found = true;

#define BEGIN_UNDOCUMENTED_COMMAND(command_name, descr, usage) \
	if(command == "list-undocumented") \
		std::cout << "    "  << command_name << ": " << descr << '\n'; \
	else if((args.size()==1) && command == "help" && (args[0] == command_name)) \
		std::cout << "UNDOCUMENTED COMMAND!!!\nUsage: \n" << usage; \
	else if (command == command_name) { is_command_found = true;

#define END_COMMAND }

#endif // MLP

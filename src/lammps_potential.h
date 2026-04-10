/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_LAMMPS_INTERFACE_H
#define MLIP_LAMMPS_INTERFACE_H


#include "basic_potentials.h"


 // class allows one to use potentials embedded in LAMMPS. It is implemented via files exchange with LAMMPS that should be launched by MLIP for EFS calculation
class LAMMPS_potential : public AnyPotential, protected InitBySettings
{
private:
	const int io_max_attemps = 5;						// number of attempts to read/write files 

	std::string input_filename;								// input filename for LAMMPS with configuration (dump)
	std::string output_filename;								// input filename for LAMMPS with configuration (dump)
	std::string start_command;							// command to OS environment launching LAMMPS

	void InitSettings() // Sets correspondence between variables and setting names in settings file
	{
		MakeSetting(input_filename, "input-file");
		MakeSetting(output_filename, "output-file");
		MakeSetting(start_command, "start-command");
	}

public:
	LAMMPS_potential(const Settings& settings)
	{
		InitSettings();
		ApplySettings(settings);
	}
	LAMMPS_potential(	const std::string& _input_filename,
						const std::string& _output_filename,
						const std::string& _start_command) :
						input_filename(_input_filename),
						output_filename(_output_filename),
						start_command(_start_command) {}

	void CalcEFS(Configuration& conf);
};

#endif //#ifndef MLIP_LAMMPS_INTERFACE_H


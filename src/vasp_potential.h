/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_VASP_POTENTIAL_H
#define MLIP_VASP_POTENTIAL_H

#include "basic_potentials.h"

//! A class that calls VASP and returns EFS
class VASP_potential : public AnyPotential, protected InitBySettings
{
  private:
	const int io_max_attemps = 5;	//!< Number of reading/writing attempts while creating i/o files 
	bool CheckCfgNotChanged(Configuration& cfg_vrf, Configuration& cfg);

	std::string input_filename;		//!< Input configuration file name (e.g., vasp_dir/POSCAR)
	std::string output_filename;	//!< OUTCAR file name (e.g., vasp_dir/OUTCAR)
	std::string start_command;		//!< (e.g., ./run_vasp.sh)

	void InitSettings() // Sets correspondence between variables and setting names in settings file
	{
		MakeSetting(input_filename, "poscar");
		MakeSetting(output_filename, "outcar");
		MakeSetting(start_command, "start-command");
	}

  public:
	VASP_potential(const Settings& settings)
	{
		InitSettings();
		ApplySettings(settings);
	}

	VASP_potential( const std::string& _input_filename,
					const std::string& _output_filename,
					const std::string& _start_command) :
		input_filename(_input_filename),
		output_filename(_output_filename),
		start_command(_start_command) {}

	void CalcEFS(Configuration& cfg);
	void CalcE(Configuration& cfg)
	{
		CalcEFS(cfg);
	};
};

#endif //#ifndef MLIP_VASP_POTENTIAL_H


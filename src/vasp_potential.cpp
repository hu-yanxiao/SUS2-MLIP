/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */


#include "vasp_potential.h"


using namespace std;


bool VASP_potential::CheckCfgNotChanged(Configuration& cfg_vrf, Configuration& cfg)
{
	bool answer = true;

	if (norm(cfg.lattice - cfg_vrf.lattice) > 1.0e-4)
	{
		cfg_vrf.AppendToFile("Error.cfg_POSCAR");
		cfg.AppendToFile("Error.cfg_OUTCAR");
		Warning("VASP interface: lattice changed after EFS calculation!");
		answer &= false;
	}
	if (cfg.size() != cfg_vrf.size())
	{
		cfg_vrf.AppendToFile("Error.cfg_POSCAR");
		cfg.AppendToFile("Error.cfg_OUTCAR");
		Warning("VASP interface: atoms count changed after EFS calculation!");
		answer &= false;
	}
	for (int i=0; i<cfg.size(); i++)
	{
		Vector3 shift((cfg_vrf.pos(i) - cfg.pos(i)) * cfg_vrf.lattice.inverse());	// shift must has components close to 0 or 1
		if ((fabs(shift[0]) > 1.0e-4 && fabs(shift[0])-1.0 > 1.0e-4) ||
			(fabs(shift[1]) > 1.0e-4 && fabs(shift[1])-1.0 > 1.0e-4) ||
			(fabs(shift[2]) > 1.0e-4 && fabs(shift[2])-1.0 > 1.0e-4))
		{
			cfg_vrf.AppendToFile("Error.cfg_POSCAR");
			cfg.AppendToFile("Error.cfg_OUTCAR");
			Warning("VASP interface: position of atom#" + to_string(i+1) + 
					" changed after EFS calculation!");
			answer &= false;
		}
	}

	return answer;
}


//! create POSCAR, call VASP, and read OUTCAR
void VASP_potential::CalcEFS(Configuration & conf)
{
	Configuration cfg(conf);	// the next operation must not change conf
	cfg.MoveAtomsIntoCell();

	for (int attempt_cnt=1; attempt_cnt<=io_max_attemps; attempt_cnt++)
	{
		if (attempt_cnt > 1)
			Warning("VASP_potential: Restarting VASP calculation (attempts left: "
				+ to_string(io_max_attemps - attempt_cnt) + ")");

		try
		{
			cfg.WriteVaspPOSCAR(input_filename.c_str());
		}
		catch (MlipException& excep)
		{
			Warning("VASP_potential: POSCAR writing failed");

			cfg.has_energy(false);
			cfg.has_forces(false);
			cfg.has_stresses(false);

			if (attempt_cnt < io_max_attemps)
				continue;
			else
				throw excep;
		}

		// cleaning outcar
		ofstream ofs(output_filename);
		ofs.close();
		// starting VASP
		int res = system(("rm " + output_filename).c_str());
		//if (res != 0) warning("\"rm " + output_filename + "\" failed");
		res = system(start_command.c_str());
		if (res != 0)
			Message("VASP exits with code " + to_string(res));

		Configuration cfg_vrf(cfg); 
		try
		{
			cfg.LoadFromOUTCAR(output_filename);
		}
		catch (MlipException& excep)
		{
			Warning("VASP_potential: OUTCAR reading failed with " + excep.message);

			res = system(("rm " + output_filename).c_str());
			if (res != 0) Warning("VASP_potential: \"rm " + output_filename + "\" failed");
			string dir = output_filename.substr(0, output_filename.length()-6);
			res = system(("rm " + dir + "WAVECAR").c_str());
			if (res != 0) Warning("VASP_potential: \"rm " + output_filename + "\" failed");
			res = system(("rm " + dir + "CHGCAR").c_str());
			if (res != 0) Warning("VASP_potential: \"rm " + output_filename + "\" failed");

			cfg.has_energy(false);
			cfg.has_forces(false);
			cfg.has_stresses(false);

			if (attempt_cnt >= io_max_attemps)
				throw excep;

			continue;
		}

		if (!CheckCfgNotChanged(cfg_vrf, cfg))
			ERROR("Configuration changed after calculation on VASP");

		break;
	}

	if (!cfg.has_stresses())
		Warning("Stresses have not been calculated by VASP");

	conf = cfg;
	
	conf.features["EFS_by"] = "VASP";
}

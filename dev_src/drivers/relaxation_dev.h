/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_RELAXATION_DEV_H
#define MLIP_RELAXATION_DEV_H


#include "../../src/drivers/basic_drivers.h"
#include "../../src/common/bfgs.h"


class RelaxationDev : public AnyDriver
{
private:
	int pos_offset; // indicates initial position of atom coordinates in array 'x' (AWA forces in array 'grad'). In case of relaxation of cell and atoms ofset =6, otherwise =0
	Array1D temp_x; // temporary x for initing bfgs
	double bfgs_f = 0;
	Array1D bfgs_g;

	std::string relaxed_cfgs_filename;

	void InitDefaultSettings()
	{
		settings.clear();
		settings["DRVR_RLX_AtomsCellBoth"] = "2";
		settings["DRVR_RLX_BFGS_StopStep"] = "0.0001";
		settings["DRVR_RLX_BFGS_StopGrad"] = "0.001";
		settings["DRVR_RLX_BFGS_MaxStep"] = "0.5";
		settings["DRVR_RLX_BFGS_MinStep"] = "1.0e-8";
		settings["DRVR_RLX_BFGS_WolfeC1"] = "0.1";
		settings["DRVR_RLX_BFGS_WolfeC2"] = "0.5";
		settings["DRVR_RLX_Stop_ItrLimit"] = "1000";
		settings["DRVR_RLX_MinDist"] = "1.0";
		settings["DRVR_RLX_Log"] = "0";
		settings["DRVR_RLX_NoGradLineSearch"] = "0";
		settings["DRVR_RLX_LOG_Fnm"] = "relaxation.log";
		settings["DRVR_RLX_OutCfg"] = "0";
		settings["DRVR_RLX_OUTCFG_FileName"] = "relaxed.cfg";
	};

	double max_deformation; // the largest step in the lattice
	double max_atom_travel; // the largest step for the atoms

public://protected:
	std::map<std::string, std::string> settings;

	void PassEFStoBFGS();
	void GetCfgfromBFGS(); //!< Sets cfg from x. Also sets max_deformation and max_atom_travel
	void InitBFGSWithCfg();

public:
	AnyPotential* pot = nullptr;
	BFGS bfgs;
	Configuration cfg;

	bool relax_atoms;	//!< whether to relax atom relaxation (false means their fractional coordinates are fixed)
	bool relax_cell;	//!< whether to relax the supercell
	double tol_force = 1e-3;	//!< maximal force tolerance for convergence
	double tol_stress = 1e-2;	//!< maximal stress tolerance for convergence, in GPa
	double pressure = 0.0;		//!< external pressure, in eV/Angsrom^3
	double mindist_limit = 1.0;	//!< Minimal distance in Angstroms. Halts the relaxation if hit.
	double maxstep_atoms = 0.1;	//!< Maximal distance atoms are allowed to travel in one iteration (in Angsrom)
	double maxstep_lattice = 0.02;	//!< Maximal factor by which lattice can change in one iteration
	bool no_grads_for_linesearch = false;

	int cfg_count = 0; //!< Number of relaxed configurations.
	int iterations_count = 0; //!< Number of EFS calcs

	RelaxationDev(AnyPotential* _pot, const Configuration& _cfg, bool _relax_atoms, bool _relax_cell);
	RelaxationDev(AnyPotential* _pot, const Configuration& _cfg, std::map<std::string, std::string> _settings);
	~RelaxationDev();

	void Run(int max_itr_cnt); //!< Run the relaxation 

	void PrintDebug(const char* fnm);
};

#endif //#ifndef MLIP_RELAXATION_DEV_H

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_RELAXATION_H
#define MLIP_RELAXATION_H


#include "basic_drivers.h"
#include "../common/bfgs.h"


class Relaxation : public AnyDriver, protected InitBySettings
{
private:
	int offset = 0;		// indicates initial position of atom coordinates in array 'x' (AWA forces in array 'grad'). In case of relaxation of cell and atoms ofset =6, otherwise =0
	std::vector<double> x_prev;

	void InitSettings()
	{
		MakeSetting(pressure,				"pressure");
		MakeSetting(tol_stress,				"stress-tolerance");
		MakeSetting(tol_force,				"force-tolerance");
		MakeSetting(maxstep_constraint,		"max-step");
		MakeSetting(minstep_constraint,		"min-step");
		MakeSetting(mindist_constraint,		"min-dist");
		MakeSetting(bfgs.wolfe_c1,			"bfgs-wolfe-c1");
		MakeSetting(bfgs.wolfe_c2,			"bfgs-wolfe-c2");
		MakeSetting(itr_limit,				"iteration-limit");
		MakeSetting(log_output,				"log");
	};

protected:
    static const char* tagname;             // tag name of object
	bool relax_atoms_flag;					// indicator of atom relaxation
	bool relax_cell_flag;					// indicator of cell relaxation

	Array1D x;								// argument-vector holder for BFGS
	Array1D g;								// gradient-vector holder for BFGS
	double f;								// function-value holder for BFGS

	// temporal variables
	double curr_mindist;					// current minimal distance
	double max_force;						// maximal force magnitude on the current iteration
	double max_stress;						// maximal deviation of stress tensor from target (maximal stress, if external pressure is 0) on the current iteration
	double max_step;						// maximal change in atoms displacement and/or lattice vectors displacements on the last iteration

	void EFStoFuncGrad();					// transfers of EFS from configuration to f, g (function and gradient)
	void XtoLatPos();						// transfers from x-vector to lattice vectors and atomic positions. Also calculates max_atom_travel
	void LatPosToX();						// transfers lattice vectors and atomic positions to x-vector
	bool CheckConvergence();				// returns true if forces and stresses are less than tolerance
	bool FixStep();							// trying to reduce step to achieve agreement with maxstep_constraint and mindist_constraint constraints

	void Init();							// initiates relaxation process. Called by Run() before iterations begun

	void WriteLog();
	
public:
	BFGS bfgs;								// BFGS minimization algorithm
	Configuration cfg;						// configuration being relaxed
	
	// Settings
	double pressure = 0.0;					// external pressure in eV/Angsroms^3
	double tol_force = 1e-4;					// maximal force tolerance for convergence. If <= 0 then fractional coordinates are held (no atoms relaxation)
	double tol_stress = 1e-3;				// maximal stress tolerance for convergence, in GPa. If <= 0 then lattice vectors are held (no supercell relaxation)
	double maxstep_constraint = 0.05;		// maximal distance atoms and lattice vectors are allowed to travel in one iteration (in Angstrom)
	double minstep_constraint = 1.0e-8;		// minimal distance atoms and lattice vectors are allowed to travel in one iteration (in Angstrom)
	double mindist_constraint = 1.0;		// in angstroms. Required for prevention of MTP extrapolations
	int itr_limit = HUGE_INT;				// iteration limit
	std::string log_output = "";			// log

	int struct_cntr = 0;					// Relaxed structures counter
	int step = 0;							// EFS calcs counter for a given structure
	
	Relaxation(	AnyPotential* _p_potential, 
				const Configuration& _cfg, 
				double _tol_force = 1e-4,	// If _tol_force <= 0 fractional coordinates are hold(no atoms relaxation)
				double _tol_stress = 1e-3);	// If _tol_stress <= 0 lattice vectors are hold (no cupercell relaxation)
	Relaxation(	AnyPotential* _p_potential, const Settings&);
	~Relaxation();

	void Run() override;					// Run relaxation process. 

	void PrintDebug(const char* fnm);		// Prints configuration to file in debug purposes
};

#endif //#ifndef MLIP_RELAXATION_H

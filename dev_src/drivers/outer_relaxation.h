/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_OUTER_RELAXATION_H
#define MLIP_OUTER_RELAXATION_H


#include "../../src/drivers/basic_drivers.h"
#include "../../src/common/bfgs.h"
#include <set>


class OuterNBHRelaxation : public AnyDriver
{
private:
	const int pos_offset = 9; // indicates initial position of atom coordinates in array 'x' (AWA forces in array 'grad'). In case of relaxation of cell and atoms ofset =6, otherwise =0

	double max_deformation; // the largest step in the lattice
	double max_atom_travel; // the largest step for the atoms

	int fix_nbh_ind;
	std::set<int> ignored_atom_inds;

	double f;
	Array1D g;
	Array1D x;

public://protected:

	void EFStoFuncGrad();
	void XtoLatPos(); //!< Sets cfg from x. Also sets max_deformation and max_atom_travel
	void InitBFGSWithCfg();

public:
	AnyPotential* pot = nullptr;
	BFGS bfgs;
	Configuration cfg;

	double tol_force = 1e-3;		// maximal force tolerance for convergence
	double tol_stress = 1e-2;		// maximal stress tolerance for convergence, in GPa
	double pressure = 0.0;			// in eV/Angsrom^3
	double min_dist = 1.0;			// In Angstroms. Required for preventing extrapolation
	double maxstep_atoms = 0.1;		// Maximal distance atoms are allowed to travel in one iteration (in Angsrom)
	double maxstep_lattice = 0.02;	// Maximal factor by which lattice can change in one iteration

	int iter = 0;					// Number of EFS calcs

	OuterNBHRelaxation(AnyPotential* _pot, const Configuration& _cfg, int _fix_nbh_ind, double cutoff);

	void Run(int max_itr_cnt);		// Run relaxation process. 

	void PrintDebug(const char* fnm);
};

#endif //#ifndef MLIP_OUTER_RELAXATION_H

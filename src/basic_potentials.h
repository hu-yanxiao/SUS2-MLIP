/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_BASIC_POTENTIALS_H
#define MLIP_BASIC_POTENTIALS_H

#include "neighborhoods.h"

//	Virtual basic class to be inherited by all classes representing interatomic interaction models
class AnyPotential
{
protected:
	void ResetEFS(Configuration& cfg);								// Sets EFS to 0.0 and has_... to true					
public: 
	virtual void CalcEFS(Configuration& cfg) = 0;					// Calculates energy, forces and stresses for cfg
	virtual void CalcE(Configuration& cfg) {	CalcEFS(cfg);	};	// Calculates energy for cfg
	virtual ~AnyPotential() {};

	// Cheks forces and stresses to be energy derivative by finite differences. Returns true if deviation between FD-calculated and exact force is less than control_delta, 
	bool CheckEFSConsistency_debug(	Configuration cfg,				// checks whether f = -dE/dx and stress = Lattice^(-T) * dE/d(Lattice) by finite differences
									double displacement=0.001,		// used in debug purposes
									double control_delta=0.00001);	// displacement - finite difference step, control_delta - tolerance
};

//	This potential does nothing and can be used as dummy in some objects calling potential when no EFS calculation is required. 
class VoidPotential: public AnyPotential
{
	void CalcE(Configuration&) {};
	void CalcEFS(Configuration&) {};
};

#endif //#ifndef MLIP_BASIC_POTENTIALS_H


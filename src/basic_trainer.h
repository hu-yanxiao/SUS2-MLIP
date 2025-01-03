/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_BASIC_TRAINER_H
#define MLIP_BASIC_TRAINER_H

#include "basic_mlip.h"

// basic class for fitting MLIP
class AnyTrainer //: public LogWriting
{
protected:
	unsigned int learn_cntr = 0;
	unsigned int train_cntr = 0;

public:
	AnyLocalMLIP* p_mlip = nullptr;							// pointer to MLIP that is fitted

	double wgt_eqtn_energy = 1.0;							// Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem 
	double wgt_eqtn_forces = 1.0;							// Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem
	double wgt_eqtn_stress = 1.0;							// Optimization weights balancing equations for energy, forces on each atom of the system and stress components in minimizaion problem
	double wgt_eqtn_constr = 1.0;							// Optimization weight for constrains
	double wgt_rel_forces = 0.0;							// if <= 0 all forces in configuration are fitte with the same weight. Otherwise each force in cfg is fitted with its own weight equal wgt_eqtn_forces * wgt_rel_forces / (cfg.force(i).NormSq() + wgt_rel_forces)
	std::string weighting;									// weighting of the loss functional's components while training 

protected:
	AnyTrainer() {};
public:
	AnyTrainer(	AnyLocalMLIP* _p_mlip,						// Constructor requires MTP basis
				double _wgt_eqtn_energy = 1.0,				// Optional parameters are the weights coeficients of energy, forces and stresses equations in minimization problem
				double _wgt_eqtn_forces = 0.0,
				double _wgt_eqtn_stress = 0.0,
				double _wgt_rel_forces = 0.0,
				double _wgt_eqtn_constr = 0.0) :
		p_mlip(_p_mlip),
		wgt_eqtn_energy(_wgt_eqtn_energy),
		wgt_eqtn_forces(_wgt_eqtn_forces),
		wgt_eqtn_stress(_wgt_eqtn_stress),
		wgt_rel_forces(_wgt_rel_forces),
		wgt_eqtn_constr(_wgt_eqtn_constr) {
		weighting = "vibrations";
	};
	virtual ~AnyTrainer() {};
	 
	virtual void Train() { ERROR("Inapropriate call of \"Train()\""); };	// Required if training set is already submitted (via inherited functions)
	virtual void Train(std::vector<Configuration>& train_set) = 0;	// function starting training procedure
};

#endif // MLIP_BASIC_TRAINER_H

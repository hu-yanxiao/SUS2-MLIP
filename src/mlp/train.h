/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifndef MLIP_UTILS_TRAIN_H
#define MLIP_UTILS_TRAIN_H


#include "../linear_regression.h"

void Train(	const std::string& mtpfnm, 
			const std::string& cfgfnm, 
			double ene_eq_wgt = 1.0,
			double frc_eq_wgt = 0.001,
			double str_eq_wgt = 0.1,
			double relfrc_wgt = 0.0,
			const std::string mtrxfnm = "",
			std::ostream* p_log = &std::cout);

#endif

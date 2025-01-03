/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_DEV_UTILS_TRAIN_H
#define MLIP_DEV_UTILS_TRAIN_H

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include "../mtpr_trainer.h"
#include "../../src/error_monitor.h"

void AddConfigs(const char*, NonLinearRegression &, int, int);
void Rescale(MTPR_trainer& trainer, MLMTPR& mtpr);
void Train_MTPR(std::vector<std::string>& args,std::map<std::string, std::string>& opts);


#endif
	

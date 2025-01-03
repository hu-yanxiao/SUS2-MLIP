/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_CFG_GENERATOR_H
#define MLIP_CFG_GENERATOR_H


#include "../../src/common/utils.h"
#include "../../src/basic_potentials.h"


/*/ Driver generating configuration
class CfgGenerator : public AnyDriver
{
private:
	std::ifstream ifs;
	Configuration cfg;

public:
	// default settings
	double lattice_shake = 0.0;
	double pos_shake = 0.0;	
	int replics1 = 1;
	int replics2 = 1;
	int replics3 = 1;

	void Generate()
	{
		
	};
	void Run() {Generate(); pot->CalcEFS(); };
};*/


#endif //#ifndef MLIP_CFG_GENERATOR_H

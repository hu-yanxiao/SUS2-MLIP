/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file cntributors: Ivan Novikov
*/

#ifndef MLIP_MAXVOL_SAVER_H
#define MLIP_MAXVOL_SAVER_H


#include "../src/basic_potentials.h"

//! A potential that calls VASP and returns EFS
class MaxVolSaver : public AnyPotential
{

public:
	std::string output_filename;  //!< Name of the file for stock of selected configurations

	MaxVolSaver(const std::string _output_filename) :
		output_filename(_output_filename)
	{
		std::ofstream ofs(output_filename);
		ofs.close();
	}

	void CalcEFS(Configuration& cfg)
	{
		std::ofstream ofs(output_filename, std::ios::app);
		cfg.features["mindist"] = to_string(cfg.MinDist());
		cfg.Save(ofs);
		ofs.close();


	}
	void CalcE(Configuration& cfg)
	{
		CalcEFS(cfg);
	};
};

#endif //#ifndef MLIP_VASP_INTERFACE_H


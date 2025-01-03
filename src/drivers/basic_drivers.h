/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_BASIC_DRIVERS_H
#define MLIP_BASIC_DRIVERS_H



/////

#include "../mlip.h"
#include "../basic_potentials.h"
#include <iostream>


// Basic class for all drivers (algorithms requiring potential and modifing configurations, e.g. MD, Relaxation, etc.)
class AnyDriver // : protected InitBySettings//, protected LogWriting 
{
public:
	AnyPotential* p_potential;			// pointer to potential driver operates with
	virtual void Run() = 0;				// function starting the driving proces
	virtual ~AnyDriver() {};
};


// Driver taking configurations from file.
class CfgReader : public AnyDriver, protected InitBySettings
{
private:
	Configuration cfg;

protected:
  static const char* tagname;                           // tag name of object
public:
	// default settings
	std::string filename = "";			//	filename
	int max_cfg_cnt = HUGE_INT;			//	limit of configurations to be read
	std::string log_output = "";		//	log


	CfgReader(std::string _filename, AnyPotential *_p_potential);
	CfgReader(AnyPotential * _p_potential, const Settings& settings);

	void InitSettings();
	
	void Run();							//	starts reading configuations from file
};


#endif //#ifndef MLIP_BASIC_DRIVERS_H

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_ERROR_MONITOR_H
#define MLIP_ERROR_MONITOR_H


#include "configuration.h"


// calculates difference in EFS for configurations. Used to calculate fitting errors
class ErrorMonitor //: public LogWriting
{
protected:
  static const char* tagname;                           // tag name of object
public:
	class Quantity												// class storing some numbers related to errors 
	{
	public:
		double value = 0.0;										// exact value 
		double delta = 0.0;										// difference between exact and approximating value
		double valsq = 0.0;										// squared exact value
		double dltsq = 0.0;										// delta^2
		double reltv = 0.0;										// delta/value
		void clear() { value = delta = valsq = dltsq = 0.0; };
		Quantity() {};
	};

	class Accumulator											// class accumulating quantities (Quantity) and stores maximal and sum of them
	{
	public:
		int count;												// number of accumulated quantities
		Quantity max;											// maximal of quantities
		Quantity sum;											// summ of quantities
		Accumulator() { clear(); };
		void accumulate(Quantity& item)							// updates max and sum with item 
		{
			max.value = __max(max.value, item.value);
			max.delta = __max(max.delta, item.delta);
			max.valsq = __max(max.valsq, item.valsq);
			max.dltsq = __max(max.dltsq, item.dltsq);
			max.reltv = __max(max.reltv, item.reltv);

			sum.value += item.value;
			sum.delta += item.delta;
			sum.valsq += item.valsq;
			sum.dltsq += item.dltsq;
			sum.reltv += item.reltv;

			count++;
		};
		void accumulate(Accumulator& item)						// updates max and sum with item 
		{
			max.value = __max(max.value, item.max.value);
			max.delta = __max(max.delta, item.max.delta);
			max.valsq = __max(max.valsq, item.max.valsq);
			max.dltsq = __max(max.dltsq, item.max.dltsq);
			max.reltv = __max(max.reltv, item.max.reltv);

			sum.value += item.sum.value;
			sum.delta += item.sum.delta;
			sum.valsq += item.sum.valsq;
			sum.dltsq += item.sum.dltsq;
			sum.reltv += item.sum.reltv;

			count += item.count;
		};
		void clear()											// Resets accumulated quntities
		{
			count = 0;
			max.value = max.delta = max.valsq = max.dltsq = max.reltv = 0.0;
			sum.value = sum.delta = sum.valsq = sum.dltsq = sum.reltv = 0.0;
		};
	};

	double relfrc_regparam = 0.0;	// regularization parameter for calculation of relative forces errors. If =0 absolute values are accumulated

	int cfg_cntr;					// compared configuration counter

	// variables storing quantities for a pair configurations with exact and approximated EFS
	Quantity	ene_cfg;			// energy 
	Quantity	epa_cfg;			// energy per atom
	Quantity	str_cfg;			// stress
	Quantity	vir_cfg;			// virial stresses (in GPa)
	Accumulator frc_cfg;			// force

	// variables storing accumulated quantities
	Accumulator ene_all;			// energy 
	Accumulator epa_all;			// energy per atom
	Accumulator frc_all;			// stress
	Accumulator str_all;			// force
	Accumulator vir_all;			// virial stresses (in GPa)

	ErrorMonitor(double _relfrc_regparam = 0.0);
	~ErrorMonitor();
	void compare(Configuration& ValidCfg, Configuration& CheckCfg, double wgt=1.0);	// compare two configurations updating ???_cfg
	void collect(Configuration& ValidCfg, Configuration& CheckCfg, double wgt=1.0); // compare and accumulate deviation data updating accumulators ???_all
	void GetReport(std::string& log); // generate a report
	void reset();					// resets all accumulators

	inline double ene_rmsabs() { return sqrt(ene_all.sum.dltsq / (ene_all.count + 1.0e-300)); }		// root-mean-squred energy deviation
	inline double ene_aveabs() { return ene_all.sum.delta / (ene_all.count + 1.0e-300); }			// average energy deviation
	inline double ene_averel() { return ene_all.sum.reltv / (ene_all.count + 1.0e-300); }			// relative energy deviation
	inline double epa_rmsabs() { return sqrt(epa_all.sum.dltsq / (epa_all.count + 1.0e-300)); }		// root-mean-squred deviation of energy per atom
	inline double epa_aveabs() { return epa_all.sum.delta / (epa_all.count + 1.0e-300); }			// average deviation of energy per atom
	inline double epa_averel() { return epa_all.sum.reltv / (epa_all.count + 1.0e-300); }			// relative deviation of energy per atom
	inline double frc_aveabs() { return frc_all.sum.delta / (3*frc_all.count + 1.0e-300); }			// average absolute deviation of forces
	inline double frc_rmsabs() { return sqrt(frc_all.sum.dltsq / (3*frc_all.count + 1.0e-300)); }		// root-mean-squred deviation of forces
	inline double frc_averel() { return frc_all.sum.delta / (frc_all.sum.value + 1.0e-300); }			// average relative force error
	inline double frc_rmsrel() { return sqrt(frc_all.sum.dltsq / (frc_all.sum.valsq + 1.0e-300)); }	// root-mean-squred deviation of forces divided by root-mean-squred force 
	inline double str_rmsabs() { return sqrt(str_all.sum.dltsq / (9*str_all.count + 1.0e-300)); }		// root-mean-squred stress deviation
	inline double str_rmsrel() { return sqrt(str_all.sum.dltsq / (str_all.sum.valsq + 1.0e-300)); }	// root-mean-squred stress deviation divided by root-mean-squred  stress 
	inline double str_averel() { return str_all.sum.delta / (str_all.sum.value + 1.0e-300); }			// average relative stress 
	inline double str_aveabs() { return str_all.sum.delta / (9*str_all.count + 1.0e-300); }			// average absolute stress 
	inline double vir_rmsabs() { return sqrt(vir_all.sum.dltsq / (9*vir_all.count + 1.0e-300)); }		// root-mean-squred viriral stress deviation
	inline double vir_rmsrel() { return sqrt(vir_all.sum.dltsq / (vir_all.sum.valsq + 1.0e-300)); }	// root-mean-squred viriral stress deviation divided by root-mean-squred  stress 
	inline double vir_averel() { return vir_all.sum.delta / (vir_all.sum.value + 1.0e-300); }			// average relative viriral stress 
	inline double vir_aveabs() { return vir_all.sum.delta / (9*vir_all.count + 1.0e-300); }			// average absolute viriral stress 

	void MPI_Synchronize();

};

#endif //#ifndef MLIP_ERROR_MONITOR_H


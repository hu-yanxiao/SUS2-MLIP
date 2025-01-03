/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include <iostream>
#include <fstream>
#include "error_monitor.h"
#ifdef MLIP_MPI
#	include <mpi.h>
#endif // MLIP_MPI

#include <sstream>

const char* ErrorMonitor::tagname = {"check_errors"};

ErrorMonitor::ErrorMonitor(double _relfrc_regparam)
{
	relfrc_regparam = _relfrc_regparam;
	reset();
}

ErrorMonitor::~ErrorMonitor()
{
}

void ErrorMonitor::compare(Configuration & cfg_valid, Configuration & cfg_check, double wgt)
{
	if (cfg_valid.size() != cfg_check.size())
	{
		Warning("ErrorMonitor: attempting to compare configurations of different size");
		return;
	}
	int size = cfg_valid.size();

	if (cfg_valid.has_energy() && cfg_check.has_energy())
	{
		ene_cfg.delta = wgt * fabs(cfg_valid.energy - cfg_check.energy);
		ene_cfg.dltsq = ene_cfg.delta*ene_cfg.delta;
		ene_cfg.value = cfg_valid.energy;
		ene_cfg.reltv = ene_cfg.delta / ene_cfg.value;

		epa_cfg.delta = wgt * ene_cfg.delta / cfg_valid.size();
		epa_cfg.dltsq = epa_cfg.delta*epa_cfg.delta;
		epa_cfg.value = ene_cfg.value / cfg_valid.size();
		epa_cfg.reltv = ene_cfg.reltv;
	}
	else
		ene_cfg.clear();

	frc_cfg.clear();
	if (cfg_valid.has_forces() && cfg_check.has_forces())
	{
		for (int i = 0; i < cfg_valid.size(); i++)
		{
			Vector3 &f = cfg_valid.force(i);

			Quantity frc_one;

			frc_one.valsq = f.NormSq();
			frc_one.value = sqrt(frc_one.valsq);
			frc_one.dltsq = (cfg_check.force(i) - f).NormSq();
			frc_one.delta = sqrt(frc_one.dltsq);
			frc_one.reltv = frc_one.delta / (frc_one.value + relfrc_regparam + 1.0e-300);

			frc_cfg.accumulate(frc_one);
		}

		frc_cfg.max.delta *= wgt;
		frc_cfg.max.dltsq *= wgt*wgt;
		frc_cfg.max.reltv *= wgt;
		frc_cfg.sum.reltv *= wgt / size;
		frc_cfg.sum.delta *= wgt;
		frc_cfg.sum.dltsq *= wgt*wgt;
	}

	str_cfg.clear();
	if (cfg_valid.has_stresses() && cfg_check.has_stresses())
	{
		str_cfg.valsq = cfg_valid.stresses.NormFrobeniusSq();
		str_cfg.value = sqrt(str_cfg.valsq);
		str_cfg.dltsq = (cfg_check.stresses - cfg_valid.stresses).NormFrobeniusSq();
		str_cfg.delta = sqrt(str_cfg.dltsq);
		str_cfg.reltv = str_cfg.delta / (str_cfg.value + 1.0e-300);
	}

	vir_cfg.clear();
	if (cfg_valid.has_stresses() && cfg_check.has_stresses())
	{
		const double eVA3_to_GPa = 160.2176487;
		double volume = fabs(cfg_valid.lattice.det()) / eVA3_to_GPa;

		Matrix3 valid_virial(cfg_valid.stresses);
		Matrix3 check_virial(cfg_check.stresses);
		valid_virial *= 1.0 / volume;
		check_virial *= 1.0 / volume;

		vir_cfg.valsq = valid_virial.NormFrobeniusSq();
		vir_cfg.value = sqrt(vir_cfg.valsq);
		vir_cfg.dltsq = (check_virial - valid_virial).NormFrobeniusSq();
		vir_cfg.delta = sqrt(vir_cfg.dltsq);
		vir_cfg.reltv = vir_cfg.delta / (vir_cfg.value + 1.0e-300);
	}

  std::stringstream logstrm1;
	logstrm1	<< "Errors:\tcompared " << cfg_cntr+1
			<< "\tenergy diff:" << cfg_check.energy - cfg_valid.energy
			<< "\tRMS forces: " << sqrt(frc_cfg.sum.dltsq / size)
			<< "\tstress diff: " << vir_cfg.delta
			<< std::endl;
  MLP_LOG(tagname,logstrm1.str());
}

void ErrorMonitor::collect(Configuration& cfg_valid, Configuration& cfg_check, double wgt)
{
	compare(cfg_valid, cfg_check, wgt);

	if (cfg_valid.has_energy() && cfg_check.has_energy())
	{
		ene_all.accumulate(ene_cfg);
		epa_all.accumulate(epa_cfg);
	}
	if (cfg_valid.has_forces() && cfg_check.has_forces())
		frc_all.accumulate(frc_cfg);
	if (cfg_valid.has_stresses() && cfg_check.has_stresses())
	{
		str_all.accumulate(str_cfg);
		vir_all.accumulate(vir_cfg);
	}

	cfg_cntr++;
}

void ErrorMonitor::GetReport(std::string& log)
{
  std::stringstream logstrm1;
  logstrm1 << "_________________Errors report_________________\n"
			<< "Energy:\n"
			<< "\tErrors checked for " << ene_all.count << " configurations\n"
			<< "\tMaximal absolute difference = " << ene_all.max.delta << '\n'
			<< "\tAverage absolute difference = " << ene_aveabs() << '\n'
			<< "\tRMS     absolute difference = " << ene_rmsabs() << '\n'
			<< "\n"
			<< "Energy per atom:\n"
			<< "\tErrors checked for " << epa_all.count << " configurations\n"
			<< "\tMaximal absolute difference = " << epa_all.max.delta << '\n'
			<< "\tAverage absolute difference = " << epa_aveabs() << '\n'
			<< "\tRMS     absolute difference = " << epa_rmsabs() << '\n'
			<< "\n"
			<< "Forces:\n"
			<< "\tErrors checked for " << frc_all.count << " atoms\n"
			<< "\tMaximal absolute difference = " << frc_all.max.delta << '\n'
			<< "\tAverage absolute difference = " << frc_aveabs() << '\n'
			<< "\tRMS     absolute difference = " << frc_rmsabs() << '\n'
			//<< "\tAverage relative difference = " << frc_averel() << '\n'
			<< "\tMax(ForceDiff) / Max(Force) = " << frc_all.max.delta / 
													 (frc_all.max.value + 1.0e-300) << '\n'
			<< "\tRMS(ForceDiff) / RMS(Force) = " << frc_rmsrel() << '\n'
			<< "\n"
			<< "Stresses (in eV):\n"
			<< "\tErrors checked for " << str_all.count << " configurations\n"
			<< "\tMaximal absolute difference = " << str_all.max.delta << '\n'
			<< "\tAverage absolute difference = " << str_aveabs() << '\n'
			<< "\tRMS     absolute difference = " << str_rmsabs() << '\n'
			//<< "\tAverage relative difference = " << str_averel() << '\n'
			<< "\tMax(StresDiff) / Max(Stres) = " << str_all.max.delta / 
													 (str_all.max.value + 1.0e-300) << '\n'
			<< "\tRMS(StresDiff) / RMS(Stres) = " << str_rmsrel() << '\n'
			<< "\n"
			<< "Stresses (in GPa):\n"
			<< "\tErrors checked for " << vir_all.count << " configurations\n"
			<< "\tMaximal absolute difference = " << vir_all.max.delta << '\n'
			<< "\tAverage absolute difference = " << vir_aveabs() << '\n'
			<< "\tRMS     absolute difference = " << vir_rmsabs() << '\n'
			//<< "\tAverage relative difference = " << vir_averel() << '\n'
			<< "\tMax(StresDiff) / Max(Stres) = " << vir_all.max.delta / 
													 (vir_all.max.value + 1.0e-300) << '\n'
			<< "\tRMS(StresDiff) / RMS(Stres) = " << vir_rmsrel() << '\n'
			<< "_______________________________________________\n"
			<< std::endl;
  log = logstrm1.str();
}

void ErrorMonitor::reset()
{
	cfg_cntr = 0;	

	ene_cfg.clear();
	epa_cfg.clear();
	str_cfg.clear();
	frc_cfg.clear();
	vir_cfg.clear();

	ene_all.clear();
	epa_all.clear();
	frc_all.clear();
	str_all.clear();
	vir_all.clear();
}

void ErrorMonitor::MPI_Synchronize()
#ifdef MLIP_MPI
{
	ErrorMonitor buffer_errmon;

	MPI_Allreduce(&ene_all.max, &buffer_errmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&ene_all.sum, &buffer_errmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&epa_all.max, &buffer_errmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&epa_all.sum, &buffer_errmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&frc_all.max, &buffer_errmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&frc_all.sum, &buffer_errmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&str_all.max, &buffer_errmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&str_all.sum, &buffer_errmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&vir_all.max, &buffer_errmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&vir_all.sum, &buffer_errmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&ene_all.count, &buffer_errmon.ene_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&epa_all.count, &buffer_errmon.epa_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&frc_all.count, &buffer_errmon.frc_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&str_all.count, &buffer_errmon.str_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&vir_all.count, &buffer_errmon.vir_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&cfg_cntr, &buffer_errmon.cfg_cntr, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&relfrc_regparam, &buffer_errmon.relfrc_regparam, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(&vir_all.count, &buffer_errmon.vir_all.count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	ene_all.max = buffer_errmon.ene_all.max;
	ene_all.sum = buffer_errmon.ene_all.sum;
	epa_all.max = buffer_errmon.epa_all.max;
	epa_all.sum = buffer_errmon.epa_all.sum;
	frc_all.max = buffer_errmon.frc_all.max;
	frc_all.sum = buffer_errmon.frc_all.sum;
	str_all.max = buffer_errmon.str_all.max;
	str_all.sum = buffer_errmon.str_all.sum;
	vir_all.max = buffer_errmon.vir_all.max;
	vir_all.sum = buffer_errmon.vir_all.sum;
	ene_all.count = buffer_errmon.ene_all.count;
	epa_all.count = buffer_errmon.epa_all.count;
	frc_all.count = buffer_errmon.frc_all.count;
	str_all.count = buffer_errmon.str_all.count;
	vir_all.count = buffer_errmon.vir_all.count;
	cfg_cntr = buffer_errmon.cfg_cntr;
	relfrc_regparam = buffer_errmon.relfrc_regparam;
	vir_all.count = buffer_errmon.vir_all.count;
}
#else
{}
#endif 

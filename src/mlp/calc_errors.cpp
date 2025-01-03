/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif
#include <string>
#include <iostream>

#include "../error_monitor.h"
#include "../mtp.h"
#include "calc_errors.h"

using namespace std;


void CalcErrors(std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	int mpi_rank = 0;
	int mpi_size = 1;
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

	Configuration cfg;

	// why if I change MTP to MLMTPR then the results will be different?
	// test on Dropbox/../4Evgeny/bug2/
	AnyPotential* p_mtp = nullptr;

	// forward reading .mtp-file to find out the number of components of MTP
	int spec_count = 1;
	{
        ifstream ifs(args[0], std::ios::binary);
		if (!ifs.is_open())
			ERROR("Can't open file " + args[0] + " for reading");
		while (!ifs.eof())
		{
			string foo;
			ifs >> foo;
			if (foo == "species_count")
			{
				ifs.ignore(999, '=');
				ifs >> spec_count;
				break;
			}
		}
	}
	// to decide mtp or MTPR instantiate
	if (spec_count == 1)
	{
		p_mtp = new MTP(args[0]);
		Message("One-component MTP instantiated");
	}
	else
		ERROR("Public version has not support multicomponent MTP yet");


	ErrorMonitor errmon(0.0);
	Configuration cfg_orig;

	string cfg_save_filename = opts["save-to"];
	bool is_save_cfg = false;
	ofstream ofs;
	if (cfg_save_filename != "") {
		if (mpi_size == 1) is_save_cfg = true;
		else Warning("cannot save configuration in the mpi mode");
		ofs.open(cfg_save_filename, std::ios::binary);
	}

	ifstream ifs(args[1], std::ios::binary);
	int proc_count = 0;
	for (int count = 0; cfg.Load(ifs); count++)
		if (count % mpi_size == mpi_rank) {
			cfg_orig = cfg;
			p_mtp->CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
			if (is_save_cfg) {
				if (cfg.has_energy() && cfg_orig.has_energy())
					cfg_orig.features["energy-error"] = to_string(std::abs(cfg_orig.energy - cfg.energy));
				if (cfg.has_forces() && cfg_orig.has_forces()) {
					double rms_force_err = 0;
					double max_force_err = 0;
					for (int i = 0; i < cfg.size(); i++) {
						double df = (cfg_orig.force(i) - cfg.force(i)).Norm();
						rms_force_err += df*df;
						max_force_err = __max(max_force_err, df);
					}
					rms_force_err = sqrt(rms_force_err/ cfg.size());
					cfg_orig.features["force-error-rms"] = to_string(rms_force_err);
					cfg_orig.features["force-error-max"] = to_string(max_force_err);
					cfg_orig.Save(ofs);
				}
			}
			proc_count++;
		}
	ifs.close();

	delete p_mtp;

#ifdef MLIP_MPI
	errmon.MPI_Synchronize();

	if (mpi_rank == 0) {
      std::string report;
      errmon.GetReport(report);
      std::cout << report.c_str();
  }
#else
    std::string report;
    errmon.GetReport(report);
    std::cout << report.c_str();
#endif
}

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "basic_drivers.h"
#ifdef MLIP_MPI
#	include <mpi.h>
#endif // MLIP_MPI

#include <sstream>

using namespace std;

const char* CfgReader::tagname = {"cfg_reader"};

CfgReader::CfgReader(string _filename, AnyPotential * _p_potential)
{
	p_potential = _p_potential;
	filename = _filename;
}

CfgReader::CfgReader(AnyPotential * _p_potential, const Settings& settings)
{
		InitSettings();
		ApplySettings(settings);

		p_potential = _p_potential;

		SetTagLogStream("cfg_reader", log_output); // 

		PrintSettings();
}

void CfgReader::InitSettings()
{
	MakeSetting(filename, "filename");
	MakeSetting(max_cfg_cnt, "limit");
	MakeSetting(log_output, "log");
}

void CfgReader::Run()
{
#ifdef MLIP_MPI
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (!log_output.empty() && 
		log_output != "stdout" && 
		log_output != "stderr")
			log_output += '_' + to_string(mpi_rank);
#endif // MLIP_MPI

	std::ifstream ifs;

	if (filename.find(' ', 0) != string::npos)
		ERROR("Whitespaces in filename restricted");
	ifs.open(filename, ios::binary);
	if (!ifs.is_open())
		ERROR("Can't open file for reading configurations");
//	else
//		Message("Configurations will be read from file \"" + filename + "\"");

// 	SetLogStream(log_output);

	int cfgcntr = 0;
	while (cfg.Load(ifs) && cfgcntr<max_cfg_cnt)
	{
		cfgcntr++;

#ifdef MLIP_MPI
		if (cfgcntr % mpi_size != mpi_rank)
			continue;
#endif // MLIP_MPI

		cfg.features["from"] = "database:" + filename;

		std::stringstream logstrm1;
		logstrm1 << "Cfg reading: #" << cfgcntr 
				<< " size: " << cfg.size()
				<< ", EFS data: " 
				<< (cfg.has_energy() ? 'E' : ' ')
				<< (cfg.has_forces() ? 'F' : ' ') 
				<< (cfg.has_stresses() ? 'S' : '-')
				<< (cfg.has_site_energies() ? 'V' : '-')
				<< (cfg.has_charges() ? 'Q' : '-')
				<< std::endl;
		MLP_LOG(tagname,logstrm1.str());

		p_potential->CalcEFS(cfg);
	}
#ifdef MLIP_MPI
  if ( 0 == mpi_rank )
#endif // MLIP_MPI
	Message("Reading configuration complete. " + to_string(cfgcntr) +
			" configurations read");
}

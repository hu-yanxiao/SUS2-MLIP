/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "active_learning.h"
#ifdef MLIP_MPI
#	include <mpi.h>
#endif // MLIP_MPI

#include <sstream>

using namespace std;

const char* MaxvolSelection::tagname = {"select"};

MaxvolSelection::MaxvolSelection(	AnyLocalMLIP* _p_mlip, 
									double _MV_threshold_init,
									double _MV_threshold_select,
									double _MV_threshold_swap,
									double _MV_nbh_cmpnts_weight,
									double _MV_ene_cmpnts_weight,
									double _MV_frc_cmpnts_weight,
									double _MV_str_cmpnts_weight) :
	p_mlip(_p_mlip),
	threshold_select(_MV_threshold_select),
	threshold_swap(_MV_threshold_swap),
	threshold_init(_MV_threshold_init),
	MV_nbh_cmpnts_weight(_MV_nbh_cmpnts_weight),
	MV_ene_cmpnts_weight(_MV_ene_cmpnts_weight),
	MV_frc_cmpnts_weight(_MV_frc_cmpnts_weight),
	MV_str_cmpnts_weight(_MV_str_cmpnts_weight),
	maxvol(p_mlip->CoeffCount(), _MV_threshold_init)
{
	if (threshold_init <= 0.0 || threshold_select <= 1.0 || threshold_swap <= 1.0)
		ERROR("Can't initialize Maxvol: invalid threshold value");

	Configuration void_cfg;
	selected_cfgs.resize(p_mlip->CoeffCount(), void_cfg); // this allows initilization selected_cfgs by configurations with the same id

	weighting = "vibrations";
}

int MaxvolSelection::PrepareMatrix(Configuration& cfg)
{
	//	Number of rows to be selected/evaluated 
	int	 max_rows_cnt = (MV_nbh_cmpnts_weight != 0.0)*cfg.size() +
						(MV_ene_cmpnts_weight != 0.0)*1 +
						(MV_frc_cmpnts_weight != 0.0)*3*cfg.size() +
						(MV_str_cmpnts_weight != 0.0)*9;

	int row_count = maxvol.B.size1;
	maxvol.B.resize(maxvol.B.size1 + max_rows_cnt, p_mlip->CoeffCount());

	if (MV_nbh_cmpnts_weight != 0.0)		// Selection by neighborhoods
	{
		Neighborhoods neighborhoods(cfg, p_mlip->CutOff());

		vector<double> tmp(p_mlip->CoeffCount());
		for (Neighborhood nbh : neighborhoods) 
		{
			p_mlip->CalcSiteEnergyGrad(nbh, tmp);
			memcpy(&maxvol.B(row_count++, 0), &tmp[0], tmp.size()*sizeof(double));
		}
	}

	if (MV_ene_cmpnts_weight != 0.0 &&		// Selection by energy equation
		MV_frc_cmpnts_weight == 0.0 && 
		MV_str_cmpnts_weight == 0.0)
	{
		std::vector<double> tmp(p_mlip->CoeffCount());
		p_mlip->CalcEnergyGrad(cfg, tmp);

		if (weighting == "vibrations")  //default
		{
			for (int i = 0; i < p_mlip->CoeffCount(); i++)

				maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp[i] /cfg.size();
		}
		else if (weighting == "molecules")
		{
			for (int i = 0; i < p_mlip->CoeffCount(); i++)
				maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp[i];
		}
		else if (weighting == "structures")
		{
			for (int i = 0; i < p_mlip->CoeffCount(); i++)
				maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp[i] / cfg.size();
		}

		row_count++;
	}
	else if (MV_ene_cmpnts_weight != 0.0 || // Selection by all equations
			 MV_frc_cmpnts_weight != 0.0 || 
			 MV_str_cmpnts_weight != 0.0)
	{	// TODO: optimize it for different weights combinations
		Array1D tmp4ene(p_mlip->CoeffCount());
		Array3D tmp4frc(cfg.size(), 3, p_mlip->CoeffCount());
		Array3D tmp4str(3, 3, p_mlip->CoeffCount());

		p_mlip->CalcEFSGrads(cfg, tmp4ene, tmp4frc, tmp4str);

		if (MV_ene_cmpnts_weight != 0.0)
		{
			if (weighting == "vibrations")		//default
			{
				for (int i = 0; i < p_mlip->CoeffCount(); i++)
					maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp4ene[i] / cfg.size();
			}
			else if (weighting == "molecules")
			{
				for (int i = 0; i < p_mlip->CoeffCount(); i++)
					maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp4ene[i];
			}
			else if (weighting == "structures")
			{
				for (int i = 0; i < p_mlip->CoeffCount(); i++)
					maxvol.B(row_count, i) = MV_ene_cmpnts_weight * tmp4ene[i] / (cfg.size()*cfg.size());
			}

			row_count++;
		}

		if (MV_frc_cmpnts_weight != 0.0)
		{
			if (weighting == "structures")
			{
				for (int j = 0; j < cfg.size(); j++)
					for (int a = 0; a < 3; a++)
						for (int i = 0; i < p_mlip->CoeffCount(); i++)
							maxvol.B(row_count + (3 * j + a), i) = MV_frc_cmpnts_weight * tmp4frc(j, a, i) / cfg.size();
			}
			else	//default
			{
				for (int j = 0; j < cfg.size(); j++)
					for (int a = 0; a < 3; a++)
						for (int i = 0; i < p_mlip->CoeffCount(); i++)
							maxvol.B(row_count + (3 * j + a), i) = MV_frc_cmpnts_weight * tmp4frc(j, a, i);
			}

			row_count += 3*cfg.size();
		}

		if (MV_str_cmpnts_weight != 0.0)
		{
			if (weighting == "vibrations")  //default
			{
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						for (int i = 0; i < p_mlip->CoeffCount(); i++)
							maxvol.B(row_count + (3 * a + b), i) =
							MV_str_cmpnts_weight * tmp4str(a, b, i) / cfg.size();
			}
			else if (weighting == "structures")
			{
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						for (int i = 0; i < p_mlip->CoeffCount(); i++)
							maxvol.B(row_count + (3 * a + b), i) =
							MV_str_cmpnts_weight * tmp4str(a, b, i) / (cfg.size()*cfg.size());
			}
			else if (weighting == "molecules")
			{
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						for (int i = 0; i < p_mlip->CoeffCount(); i++)
							maxvol.B(row_count + (3 * a + b), i) =
							MV_str_cmpnts_weight * tmp4str(a, b, i);
			}

			row_count += 9;
		}
	}

	return row_count;
}

double MaxvolSelection::Grade(Neighborhood& nbh)
{
	if (MV_ene_cmpnts_weight != 0.0 &&
		MV_nbh_cmpnts_weight != 1.0 &&
		MV_frc_cmpnts_weight != 0.0 &&
		MV_str_cmpnts_weight != 0.0)
		ERROR("An attempt of inapropriate calculation of neighborhood grade detected");

	if (nbh.count > 0)
	{
		vector<double> tmp(p_mlip->CoeffCount());
		p_mlip->CalcSiteEnergyGrad(nbh, tmp);
		memcpy(&maxvol.B(0, 0), &tmp[0], tmp.size()*sizeof(double));
		maxvol.B.resize(0, 0);

		return maxvol.CalcSwapGrade(1);
	}
	else
		return 0.0;
}

double MaxvolSelection::GetGrade(Configuration& cfg)
{
	if (cfg.size() == 0)
		return 0;

	if (maxvol.B.size1 > 0)
		if (!cfgs_queue.empty())
			Select();	// Releases all accumulated configurations for multiple selection
		else //(cfgs_queue.empty())  
			maxvol.B.resize(0, p_mlip->CoeffCount());	// foolproof

	int rows_cnt = PrepareMatrix(cfg);

	double max_grade = maxvol.CalcSwapGrade(rows_cnt);
	cfg.features["MV_grade"] = (max_grade > 1.0e10) ? "9.9e99" : to_string(max_grade);
	cfg_eval_count++;

	return max_grade;
}

double MaxvolSelection::Grade(Configuration& cfg)
{
	double ret = GetGrade(cfg);

	maxvol.B.resize(0, p_mlip->CoeffCount());	

	return ret;
}

void MaxvolSelection::AddForSelection(Configuration cfg)
{
	if (cfg.size() == 0)
		return;
	else
		cfg.set_new_id();

	int rows_cnt = PrepareMatrix(cfg);

	cfgs_queue.resize(rows_cnt, cfg);	

	std::stringstream logstrm1;
  	logstrm1 << "Selection: " << cfg_eval_count
			<< "added for selection" << endl;
	MLP_LOG(tagname, logstrm1.str());

	cfg_added_count++;
}

int MaxvolSelection::UpateSelected(int swap_limit)
{
	int swap_count = maxvol.MaximizeVol(threshold_swap, maxvol.B.size1, swap_limit);

	for (const std::pair<int, int>& swapka : maxvol.swap_tracker)
		swap(cfgs_queue[swapka.first], selected_cfgs[swapka.second]);

	cfg_selct_count++;
	maxvol.B.resize(0, p_mlip->CoeffCount());

	return swap_count;
}

int MaxvolSelection::Select(int swap_limit)
{
	std::stringstream logstrm1;
	logstrm1 << "Selection: selecting " << cfg_added_count << " configurations" << endl;
	MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
	
	double max_grade = maxvol.CalcSwapGrade(maxvol.B.size1);
	cfg_eval_count += cfg_added_count;
	cfg_added_count = 0;

	// Selection procedure
	int swap_count = 0;
	if (max_grade > threshold_select)
		swap_count = UpateSelected(swap_limit);

	logstrm1 << "Selection: evaluated: " << cfg_eval_count 
			<< "\tgrade: " << max_grade
			<< "\tswaps: " << swap_count
			<< "\tTS size: " << UniqueCfgCount()
			<< endl;
	MLP_LOG(tagname,logstrm1.str());

	cfgs_queue.clear();

	return swap_count;
}

int MaxvolSelection::MakeSelection(double selection_grade, Configuration& cfg)
{
	// Selection procedure
	int swap_count = 0;
	if (selection_grade > threshold_select)
	{
		cfgs_queue.clear();			// Actually this is not required (cfgs_queue should be empty), but may prevent bugs in the case of inappropriate use of class (foolproof)
		cfgs_queue.resize(maxvol.B.size1, cfg);
		swap_count = UpateSelected();
	}

  std::stringstream logstrm1;
	logstrm1 << "Selection: cfg# " << cfg_eval_count
			<< "\tgrade: " << selection_grade
			<< "\tswaps: " << swap_count
			<< "\tselected: " << cfg_selct_count
			<< "\tTS size: " << UniqueCfgCount()
			<< "\tlog(vol): " << maxvol.log_n_det
			<< endl;
  MLP_LOG(tagname,logstrm1.str());

	return swap_count;
}

int MaxvolSelection::Select(Configuration& conf,int update)
{
#ifdef MLIP_MPI
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif


	if (conf.size() == 0)
		return 0;
	
	Configuration cfg(conf);
	cfg.set_new_id();

	double max_grade = GetGrade(cfg);

	int swap_count = 0;
	
	if (update==1)
		swap_count = MakeSelection(max_grade, cfg);

	cfgs_queue.clear();

	conf.features["MV_grade"] = cfg.features["MV_grade"];

	return swap_count;
}

int MaxvolSelection::UniqueCfgCount()
{
	set<long> ids;
	for (Configuration& cfg : selected_cfgs)
		if (cfg.size() > 0)
			ids.insert(cfg.id());
	return (int)ids.size();
}

void MaxvolSelection::LoadSelected(const string& filename)
{
	Message("Selection: loading and selecting configuration from " + filename);

	auto cfgs = LoadCfgs(filename);

	for (Configuration& cfg : cfgs)
		AddForSelection(cfg);
	Select();

	cfg_eval_count = 0;
	cfg_selct_count = 0;

	Message("Selection: loading selection set complete");
}

void MaxvolSelection::SaveSelected(const string& filename)
{
	ofstream ofs(filename);
	if (!ofs.is_open())
		ERROR("Can't open \"" + filename + "\" for writing configuration");

	int cntr=0;
	for (Configuration& cfg : selected_cfgs)
		if (cfg.size() > 0)
		{
			cfg.Save(ofs);
			cfg.features["TS_index"] = to_string(++cntr);
		}
	ofs.close();

	Message("Selection: training set saved to \"" + filename + '\"');
}

void MaxvolSelection::Save(const std::string & filename)
{
	// save MLIP
	p_mlip->Save(filename);
	// open file for write at the end
	ofstream ofs(filename, ios::binary | ios::app);
	ofs << "\n^"; // key symbol (end of mlip data)
	
	// writing selection weights
	ofs.write((char*)&MV_nbh_cmpnts_weight, sizeof(double));
	ofs.write((char*)&MV_ene_cmpnts_weight, sizeof(double));
	ofs.write((char*)&MV_frc_cmpnts_weight, sizeof(double));
	ofs.write((char*)&MV_str_cmpnts_weight, sizeof(double));
	
	ofs << "\n#"; // key symbol (end of weights data)

	// writing matrices maxvol.A and maxvol.invA
	maxvol.WriteData(ofs);
	ofs << '\n';
	// writing selected configurations
	set<long> ids;			// set of id
	map<long, int> id2ind;	// mapping id to a number from 0 to n
	int cntr = 0;			// counter for unique configurations
	for (int i=0; i<selected_cfgs.size(); i++)
	{
		Configuration& cfg = selected_cfgs[i];

		if (ids.count(cfg.id()) == 0)
			id2ind[cfg.id()] = cntr++;					// transforming id to number from 0 to n
		ids.insert(cfg.id());							// gathering ids
		ofs << i << ' ' << id2ind[cfg.id()] << '\n';	// writing index-to-number correspondence
	}

	// writing configurations
	for (Configuration& cfg : selected_cfgs)
		if (ids.count(cfg.id()) != 0)
		{
			cfg.Save(ofs, Configuration::SAVE_NO_LOSS);
			ids.erase(cfg.id());
		}

	Message("Selection: state saved to \"" + filename + '\"');
}

void MaxvolSelection::Load(const std::string & filename)
{
	// Message("Selection: loading state from \"" + filename + '\"');

	if (p_mlip == nullptr)
		ERROR("MLIP must be initialized before loading selection state");
	else
		p_mlip->Load(filename);
	ifstream ifs(filename, ios::binary);
	ifs.ignore(HUGE_INT,'^');
	if (ifs.fail() || ifs.eof())
		ERROR("Error reading active learning state from " + filename);

	// Loading selection weights
	ifs.read((char*)&MV_nbh_cmpnts_weight, sizeof(double));
	ifs.read((char*)&MV_ene_cmpnts_weight, sizeof(double));
	ifs.read((char*)&MV_frc_cmpnts_weight, sizeof(double));
	ifs.read((char*)&MV_str_cmpnts_weight, sizeof(double));

	ifs.ignore(HUGE_INT, '#');
	if (ifs.fail() || ifs.eof())
		ERROR("Error loading selection weights");

	// reseting maxvol
	maxvol.Reset(p_mlip->CoeffCount(), threshold_init);

	// reading matrices maxvol.A and maxvol.invA
	maxvol.ReadData(ifs);

	vector<int> cfg_numbers(p_mlip->CoeffCount());
	// reading configuration-to-matrix-row correspondence

	int foo;
	for (int i=0; i<p_mlip->CoeffCount(); i++)
	{
		ifs >> foo >> cfg_numbers[i];
	}
	if (ifs.fail() || ifs.eof())
		ERROR("Error loading configuration data");

	// reading configurations
	ostream* p_stream_backup = SetStreamForOutput(nullptr);	// To supress warnings on reading empty configurations (if partially initialized maxvol)
	vector<Configuration> cfgs;
	Configuration cfg;
	cfgs.push_back(cfg);
	while (cfgs.back().Load(ifs))
		cfgs.push_back(cfg);
	cfgs.pop_back();
	SetStreamForOutput(p_stream_backup);			// Restoring output stream

	// counting number of entries for each configuration
	vector<int> rates(cfgs.size());
	for (int i=0; i<p_mlip->CoeffCount(); i++)
		rates[cfg_numbers[i]]++;

	// creating temporal array for swapping (it helps to make the same id for identiacal configurations)
	vector<vector<Configuration>> cfgs4swap(cfgs.size());
	for (int i=0; i<cfgs.size(); i++)
		cfgs4swap[i].resize(rates[i], cfgs[i]);

	// filling selected cfgs
	selected_cfgs.resize(p_mlip->CoeffCount());
	for (int i=0; i<p_mlip->CoeffCount(); i++)
	{
		swap(selected_cfgs[i], cfgs4swap[cfg_numbers[i]][cfgs4swap[cfg_numbers[i]].size()-1]);
		cfgs4swap[cfg_numbers[i]].pop_back();
	}

	// Message("Selection: loading state complete");
}

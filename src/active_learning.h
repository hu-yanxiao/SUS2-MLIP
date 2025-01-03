/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_ACTIVELEARNING_H
#define MLIP_ACTIVELEARNING_H

#include "basic_mlip.h"
#include "basic_trainer.h"
#include "maxvol.h"

class LOTF; // forward declaration of a friend class


// A class for selection of configurations (active learning) for learning based on Maxvol algorithm
class MaxvolSelection //: public LogWriting
{
protected:
  static const char* tagname;                           // tag name of object
	std::vector<Configuration> cfgs_queue;					// storage for the configurations accumulated for selection

	int PrepareMatrix(Configuration& cfg);					// creates matrix B for MaxVol selection
	double GetGrade(Configuration& cfg);					// returns MV grade for cfg. Also adds "MV_grade" feature to cfg. Unsafe, because does not clear maxvol.B after. Does not change cfgs_queue. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection.
	int UpateSelected(int swap_limit=HUGE_INT);				// moves configurations between cfgs_queue and selected_cfgs according to maxvol.swap_tracker. Clears maxvol.B. Does not clear cfgs_queue 
	int MakeSelection(double selection_grade, Configuration& cfg);	// subroutine of Select(), required for LOTF. Does selection after selection grade 

public:
	AnyLocalMLIP* p_mlip = nullptr;							// pointer to actively learning MLIP

	std::vector<Configuration> selected_cfgs;				// storage of selected configurations

	MaxVol maxvol;											// MaxVol algorithm object

	// Selection weights
	double MV_nbh_cmpnts_weight;							// weight for site energies equaions in selection
	double MV_ene_cmpnts_weight;							// weight for energy equaions in selection
	double MV_frc_cmpnts_weight;							// weight for forces equaions in selection
	double MV_str_cmpnts_weight;							// weight for stress equaions in selection
	std::string weighting;									// division on different powers of cfg.size()

	// threshold parameters
	double threshold_init;									// Maxvol matrix is initialized by IdentityMatrix * threshold_init
	double threshold_select;								// >=1.0. is used in MaxVol routines. Configuration is selected if Grade() > 1+threshold_slct
	double threshold_swap;									// if configuration is selected it replaces threshold for further swaps (for this configuration)

	// Counters
	int cfg_added_count = 0;									// counter for configurations accumulated for selection (for multiple selection only)
	int cfg_selct_count = 0;									// in the case of single selection counter for ever-selected configurations. In the case of multiple selection counter for selection calls with at least one swap (selection grade > 1 + threshold)
	int cfg_eval_count = 0;									// evaluated configurations counter

public:
	MaxvolSelection(AnyLocalMLIP* p_mlip,
					double _MV_threshold_init = 1.0e-5,
					double _MV_threshold_slct = 1.0+1.0e-3,
					double _MV_threshold_swap = 1.0+1.0e-7,
					double _MV_nbh_cmpnts_weight = 0.0,
					double _MV_ene_cmpnts_weight = 1.0,
					double _MV_frc_cmpnts_weight = 0.0,
					double _MV_str_cmpnts_weight = 0.0);
	//~MaxvolSelection();

	double Grade(Neighborhood& nbh);						// returns MV grade for nbh
	double Grade(Configuration& cfg);						// returns MV grade for cfg. Also adds "MV_grade" feature to cfg. In contrast to GetGrade() is safe and clears maxvol.B after. Does not change cfgs_queue. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection
	void AddForSelection(Configuration cfg);				// Accumulates cfg for multiple selection
	int Select(int swap_limit = HUGE_INT);					// Performs multiple selection for accumulated configuration set. No "MV_grade" feature is added to selecting configurations.
	int Select(Configuration& cfg,int update=1);			// Single selection of accumulated configuration. Also adds "MV_grade" feature to cfg. If some configurations are accumulated for multiple selection calls Select() before procissing cfg to finalize their selection.
	int UniqueCfgCount();									// Counts the number of unique configurations in selected_cfgs. Unique is determined by id.

	void LoadSelected(const std::string& filename);
	void SaveSelected(const std::string& filename);			// writes MV_selected to a file
	void Save(const std::string& filename);					// Saves the state of MaxvolSelection to specified file
	void Load(const std::string& filename);					// Loads the state of MaxvolSelection to specified file (MLIP will be changed to loaded)

	friend LOTF;
};


#endif // MLIP_ACTIVELEARNING_H

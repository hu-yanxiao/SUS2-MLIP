/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_WRAPPER_H
#define MLIP_WRAPPER_H

#include "lotf.h"
#include "error_monitor.h"


// Class that allows working with MLIP in multiple regimes (e.g. fitting, EFS calculating, active learning, fitting error calculation, etc.)
class MLIP_Wrapper : public AnyPotential, protected InitBySettings //, public LogWriting
{
  protected:
  static const char* tagname;                           // tag name of object
  private:
	// settings	initialized by default values
	std::string mlip_type = "mtpr";			// MLIP type
	std::string mlip_fnm = "";				// file name, MLIP will be loaded from. MLIP is not required if empty
	bool enable_EFScalc = true;			// if true MLIP will calculate EFS
	bool enable_learn = false;				// if true MLIP fitting will be performed
	std::string mlip_fitted_fnm = "";		// filename for writing MLIP after fitting. Saving mlip is not required if empty
	double fit_ene_wgt = 1.0;				// energy equation weight in fitting procedure
	double fit_frc_wgt = 0.0;				// forces equation weight in fitting procedure
	double fit_str_wgt = 0.0;				// stress equation weight in fitting procedure
	double fit_rel_frc_wgt = 0.0;			// parameter enabling fitting to relative forces
	std::string fit_log = "";				// filename or "stdout" or "stderr" for fitting log output. No logging if empty
	bool enable_select = false;				// turns selection on of true
	double slct_trsh_init = 0.0000001;		// value for maxvol initiation
	double slct_trsh_slct = 1.001;			// selection threshold 
	double slct_trsh_swap = 1.001;			// selection threshold for continue swaps if cfg is selected. If 0.0 it will be set equal to slct_trsh_slct
	double slct_trsh_break = 0.0;			// threshold for breaking the run if exceeded. Activated if >0.0. If activated updates of selected set are restricted
	double slct_nbh_wgt = 0.0;				// site energy equation weight
	double slct_ene_wgt = 1.0;				// energy equation weight
	double slct_frc_wgt = 0.0;				// forces equation weight
	double slct_str_wgt = 0.0;				// stress equation weight

	std::string slct_weighting = "vibrations";	// weighting of energies/forces/stresses for different purposes

	std::string slct_ts_fnm = "";			// if not empty selected set will be saved to this file
	std::string slct_state_save_fnm = "";	// if not empty selection state will be saved to this file 
	std::string slct_state_load_fnm = "";	// if not empty selected set will be loaded from this file
	std::string slct_log = "";				// filename or "stdout" or "stderr" for selection log output. No logging if empty
	bool efs_ignored = false;				// if efs ignored by driver it is possible to speed up some scenarios
	std::string cfgs_fnm = "";				// filename for recording configurations (calculated by CalcEFS()). No configurations are recorded if empty
	int skip_saving_N_cfgs = 0;				// every skip_saving_N_cfgs+1 configuration will be saved (see previous option)
	bool monitor_errs = false;				// enables error monitoring (comparison of abinitio EFS and MLIP EFS and accumulation errors for averaging, in particular)
	std::string errmon_log = "";			// filename or "stdout" or "stderr" for error monitor log output. No logging if empty
	std::string log_output = "";			// filename or "stdout" or "stderr" for this object log output. No logging if empty


	Configuration cfg_valid;				// temporal for errmon

	int call_count = 0;						// number of processed cfg
	int learn_count = 0;						// number of learned cfg
	int MTP_count = 0;						// number of MTP calcs
	int abinitio_count = 0;					// number of abinitio calcs

    void InitSettings()                        // Sets correspondence between variables and setting names in settings file
    {
		// new settings:
		// old settings:
        MakeSetting(mlip_fnm,           "mtp-filename");       
        MakeSetting(enable_EFScalc,     "calculate-efs");        
        MakeSetting(enable_learn,       "fit");                  
        MakeSetting(mlip_fitted_fnm,    "fit:save");             
        MakeSetting(fit_ene_wgt,        "fit:energy-weight");    
        MakeSetting(fit_frc_wgt,        "fit:force-weight");     
        MakeSetting(fit_str_wgt,        "fit:stress-weight");    
        MakeSetting(fit_rel_frc_wgt,    "fit:scale-by-force");   
        MakeSetting(fit_log,            "fit:log");              
        MakeSetting(enable_select,      "select");               
        MakeSetting(slct_trsh_init,     "select:threshold-init");
        MakeSetting(slct_trsh_slct,     "select:threshold");     
        MakeSetting(slct_trsh_swap,     "select:threshold-swap");
        MakeSetting(slct_trsh_break,    "select:threshold-break");
        MakeSetting(slct_nbh_wgt,       "select:site-en-weight");
        MakeSetting(slct_ene_wgt,       "select:energy-weight"); 
        MakeSetting(slct_frc_wgt,       "select:force-weight");  
        MakeSetting(slct_str_wgt,       "select:stress-weight"); 

		MakeSetting(slct_weighting,		"select:weighting");

        MakeSetting(slct_ts_fnm,        "select:save-selected"); 
        MakeSetting(slct_state_save_fnm,"select:save-state");    
        MakeSetting(slct_state_load_fnm,"select:load-state");    
        MakeSetting(slct_log,           "select:log");           
        MakeSetting(efs_ignored,        "select:efs-ignored");   
        MakeSetting(cfgs_fnm,           "write-cfgs");           
        MakeSetting(skip_saving_N_cfgs, "write-cfgs:skip");      
        MakeSetting(errmon_log,         "check-errors");         
        MakeSetting(log_output,         "log");   

    };
    void SetUpMLIP();    // Initialize object state according to settings

  public:
    AnyPotential* p_abinitio = nullptr;        // pointer to abinitio potential
    AnyLocalMLIP* p_mlip = nullptr;            // pointer to MLIP
    AnyTrainer* p_learner = nullptr;        // pointer to training object
    MaxvolSelection* p_selector = nullptr;    // pointer to selecting object
    LOTF* p_lotf = nullptr;                    // pointer to LOTF object
    std::vector<Configuration> training_set;// training set (for pure training regime)
    ErrorMonitor errmon;                    // error monitoring object

    MLIP_Wrapper(const Settings&);
    virtual ~MLIP_Wrapper();
    
    void CalcEFS(Configuration& cfg);        // function that does work configured according settings file (for example may perform fitting, EFS calculation, error caclulation, LOTF, etc.)
    void CalcE(Configuration& cfg);

	inline double Threshold()
	{
		return slct_trsh_slct;
	}

	inline double ThresholdBreak()
	{
		return slct_trsh_break;
	}
};

#endif


/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_LOTF_H
#define MLIP_LOTF_H


#include "basic_trainer.h"
#include "active_learning.h"
#include "linear_regression.h"


// Learning on the fly
class LOTF : public AnyPotential
{
  private:
    LinearRegression* p_linreg = nullptr;

  protected:
    static const char* tagname;               // tag name of object
    std::vector<int> p_entering_cfgs;
    std::vector<int> p_leaving_cfgs;

    int FindEnteringCfgs(long id);          // Returns the number of cfg in selected_cfgs with the id()==id. Also fills and fills p_entering_cfgs
    int FindLeavingCfgs(long id);           // Returns the number of cfg leaving selected_cfgs after selection and fills p_leaving_cfgs (required for linear retraining)

    void Retrain(Configuration& cfg);

    void CopyEFS(Configuration& from, Configuration& to);

  public:
    AnyPotential* p_base_pot = nullptr;     // pointer to Ab-initio potential
    MaxvolSelection* p_selector = nullptr;  // pointer to selecting object
    AnyTrainer* p_trainer = nullptr;        // pointer to training object. Learning potential is p_trainer->p_mlip

    bool EFS_via_MTP = true;                // if false EFS are taken from ab-initio potential on configurations chosen for learning (a bit faster)
    bool collect_abinitio_cfgs = false;     // whether to collect abinitio_cfgs

    std::vector<Configuration> abinitio_cfgs; // configuration container with ab-initio EFS

    int pot_calcs_count = 0;                // counter for base_pot->CalcEFS() calls (Ab-initio calcs)
    int MTP_calcs_count = 0;                // counter for MTP->CalcEFS() calls
    int learn_count = 0;                    // counter for learner->AddForTrain() and learner->remove_from_SLAE() calls
    int train_count = 0;                    // counter for regression SLAE solving attmpt

  private:
    LOTF() {}
  public:
    LOTF(AnyPotential* _p_base_pot,         // default constructor
         MaxvolSelection* _p_selector,
         AnyTrainer* _p_trainer,
         bool _EFS_via_MTP = false) : p_base_pot(_p_base_pot),
                                      p_selector(_p_selector),
                                      p_trainer(_p_trainer),
                                      EFS_via_MTP(_EFS_via_MTP) {};
    LOTF(AnyPotential* _p_base_pot,         // This constructor creats optimized for the linear regression object (fast retraining instead of full training on train set in default constructor)
         MaxvolSelection* _p_selector,
         LinearRegression* _p_trainer,
         bool _EFS_via_MTP = false) : p_base_pot(_p_base_pot),
                                      p_selector(_p_selector),
                                      p_trainer(_p_trainer),
                                      p_linreg(_p_trainer),
                                      EFS_via_MTP(_EFS_via_MTP) {};

    void CalcEFS(Configuration & cfg) override; // decides if learning is required, querries ab-initio EFS and learns configuration if necessary, and calculates EFS
};

#endif // MLIP_LOTF_H

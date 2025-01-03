/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "lotf.h"

#include <sstream>

using namespace std;

const char* LOTF::tagname = {"Lotf"};

int LOTF::FindEnteringCfgs(long id)
{
    p_entering_cfgs.clear();
    for (int i=0; i<p_selector->selected_cfgs.size(); i++)
        if (p_selector->selected_cfgs[i].id() == id)
            p_entering_cfgs.push_back(i);
    return (int)p_entering_cfgs.size();
}

int LOTF::FindLeavingCfgs(long id)
{
    p_leaving_cfgs.clear();
    for (int i=0; i<p_selector->cfgs_queue.size(); i++)
        if (p_selector->cfgs_queue[i].id() != id)
            p_leaving_cfgs.push_back(i);
    return (int)p_leaving_cfgs.size();
}

void LOTF::Retrain(Configuration & cfg)
{
    int wgt = FindEnteringCfgs(cfg.id());
    for (int i : p_entering_cfgs)
        CopyEFS(cfg, p_selector->selected_cfgs[i]);

    if (p_linreg == nullptr)
        p_trainer->Train(p_selector->selected_cfgs);
    else
    {
        for (int i : p_entering_cfgs)
            p_linreg->AddToSLAE(p_selector->selected_cfgs[i], wgt);
        FindLeavingCfgs(cfg.id());
        for (int i : p_leaving_cfgs)
            p_linreg->AddToSLAE(p_selector->cfgs_queue[i], -1);
        p_linreg->Train();
    }

    train_count++;
}

void LOTF::CopyEFS(Configuration & from, Configuration & to)// Copying EFS data from one configuration to another
{
    to.has_energy(from.has_energy());
    to.has_forces(from.has_forces());
    to.has_stresses(from.has_stresses());
    to.has_site_energies(from.has_site_energies());

    if (from.size() != to.size())
        to.resize(from.size());

    if (from.has_forces())
        memcpy(&to.force(0, 0), &from.force(0, 0), from.size() * sizeof(Vector3));
    if (from.has_stresses())
        to.stresses = from.stresses;
    if (from.has_energy())
        to.energy = from.energy;
    if (from.has_site_energies())
        memcpy(&to.site_energy(0), &from.site_energy(0), from.size() * sizeof(double));
}

void LOTF::CalcEFS(Configuration & cfg)
{
    if (cfg.size() == 0)
        return;

    double grade = p_selector->GetGrade(cfg);

    bool selected = (grade > p_selector->threshold_select);

    if (selected)
    {
        try
        {
            p_base_pot->CalcEFS(cfg);
        }
        catch (int errcode)    /// TODO: fix it!!!!!!!!!!!!!
        {
            Warning("LOTF: Ab-initio model failed to calculate EFS. " + to_string(errcode));
            ERROR("Unable to perform ab-initio EFS calculation");
        }
        pot_calcs_count++;
        abinitio_cfgs.emplace_back(cfg);
    }

    p_selector->MakeSelection(grade, cfg);    // if !selected this just writes log

    if (selected)
    {
        Retrain(cfg);

        p_selector->cfgs_queue.clear();    // not before Retrain()

        if (EFS_via_MTP)
        {
            p_trainer->p_mlip->CalcEFS(cfg);
            MTP_calcs_count++;
        }
    }
    else
    {
        p_trainer->p_mlip->CalcEFS(cfg);
        MTP_calcs_count++;
    }

  std::stringstream logstrm1;
  logstrm1 << "\tLOTF: cfg# "        << p_selector->cfg_eval_count
           << "\tAbinitio_calcs "    << pot_calcs_count
           << "\tMTP_calcs "        << MTP_calcs_count
           << std::endl;
  MLP_LOG(tagname,logstrm1.str());
}

/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "mlip_wrapper.h"
#include "mtp.h"
#include "linear_regression.h"
#include "pair_potentials.h"
#include "vasp_potential.h"
#include "lammps_potential.h"
#include "../dev_src/mtpr_trainer.h"

#ifdef MLIP_DEV
#   include "../dev_src/sw_basis.h"
#endif
#ifdef MLIP_MPI
#    include <mpi.h>
#endif

#include <sstream>

using namespace std;

const char* MLIP_Wrapper::tagname = {"mlip"};

void MLIP_Wrapper::SetUpMLIP()
{

#ifdef MLIP_DEV // developer's version
    StillingerWeberRadialBasis* p_sw = nullptr;
    LinearRegression* p_linreg = nullptr;
#endif

MLMTPR* p_mtpr = nullptr;
//
//    if (mlip_type == "mtp")
//    {
//        p_mlip = new MTP(mlip_fnm);
//        Message("Linearly parametrized MTP instantiated");
//    }
//    else if (mlip_type == "mtpr")
//    {
        p_mlip = p_mtpr = new MLMTPR(mlip_fnm);
//        Message("Multicomponent non-linearly parametrized MTP instantiated");
//    }
//#ifdef MLIP_DEV
//    else if (mlip_type == "sw")
//    {
//        //p_mlip = p_sw = new StillingerWeberRadialBasis(mlip_fnm);
//        Message("Multicomponent Stillinger-Weber instantiated");
//        if (enable_learn)
//            ERROR("Incorrect settings. Active Learning for SW has not been implemented yet!");
//        if (enable_select) { 
//            if (slct_frc_wgt!=0 || slct_str_wgt!=0 || slct_nbh_wgt!=0) {
//                ERROR("Incorrect settings. Selection for SW has been implemented only for the \
//                        case slct_ene_wgt != 0");
//            }
//        }
//    }
//#endif
//    else if (mlip_type == "void")
//    {
//        if (p_abinitio != nullptr)
//            ERROR("No MLIP features activated");
//        else
//            ERROR("Incorrect settings. Neither MLIP nor ab initio model specified");
//
//        enable_EFScalc = false;
//        enable_learn = false;
//        enable_select = false;
//        monitor_errs = false;
//        cfgs_fnm.clear();
//
//        return;
//    }
//    else
//        ERROR("Improper MLIP type specified. (\"mtp\" or \"mtpr\" is required)");

    if (!enable_EFScalc && !enable_learn && !enable_select) // direct ab initio calculation. No reading of MTP-file is required
    {
//        if (p_abinitio != nullptr)
//            Warning("No MLIP activated. Ab-initio model and driver will be linked directly");
//        else
//            ERROR("Incorrect settings. Neither MLIP nor ab initio model specified");
            ERROR("Incorrect settings. MTP is not activated.");
    }


    if (enable_learn)
    {
#ifdef MLIP_DEV
        if (p_mtpr == nullptr)
            p_learner = p_linreg =
            new LinearRegression(p_mlip, fit_ene_wgt, fit_frc_wgt, fit_str_wgt, fit_rel_frc_wgt);
        else
            p_learner = new MTPR_trainer(    p_mtpr, 
                                            fit_ene_wgt, 
                                            fit_frc_wgt, 
                                            fit_str_wgt, 
                                            fit_rel_frc_wgt, 
                                            1.0e-6);
#else
        if (p_mtpr != nullptr)
            p_learner = new MTPR_trainer(    p_mtpr, 
                                            fit_ene_wgt, 
                                            fit_frc_wgt, 
                                            fit_str_wgt, 
                                            fit_rel_frc_wgt, 
                                            1.0e-6);
        else
            p_learner = new LinearRegression(    p_mlip, 
                                            fit_ene_wgt, 
                                            fit_frc_wgt, 
                                            fit_str_wgt, 
                                            fit_rel_frc_wgt);
#endif
//         p_learner->SetLogStream(fit_log);
    }

    if (enable_select)
    {
        if (slct_trsh_init <= 0.0 || slct_trsh_slct <= 1.0)
            ERROR("Invalid threshold specified");
        if (slct_trsh_swap <= 1.0)
            slct_trsh_swap = slct_trsh_slct;

        if (slct_trsh_break <= 1.0)
            p_selector = new MaxvolSelection(    p_mlip,
                                                slct_trsh_init, 
                                                slct_trsh_slct, 
                                                slct_trsh_swap,
                                                slct_nbh_wgt, 
                                                slct_ene_wgt, 
                                                slct_frc_wgt, 
                                                slct_str_wgt    );
        else // "Selection by generation" mode is activated
            p_selector = new MaxvolSelection(    p_mlip,
                                                slct_trsh_init,
                                                slct_trsh_break,    // !
                                                slct_trsh_swap,
                                                slct_nbh_wgt,
                                                slct_ene_wgt,
                                                slct_frc_wgt,
                                                slct_str_wgt);

        p_selector->weighting = slct_weighting;   //NEW!!!!

        if (!slct_state_load_fnm.empty())
            p_selector->Load(slct_state_load_fnm);
    
//         p_selector->SetLogStream(slct_log);
    }
    
    // lotf scenario initialization
    if (enable_EFScalc && enable_learn && enable_select)
    {
#ifdef MLIP_DEV
        if (p_mtpr == nullptr)    
            p_lotf = new LOTF(p_abinitio, p_selector, p_linreg, !efs_ignored);
        else
            p_lotf = new LOTF(p_abinitio, p_selector, p_learner, !efs_ignored);
#else
        p_lotf = new LOTF(p_abinitio, p_selector, (LinearRegression*)p_learner, !efs_ignored);
#endif

        if (!slct_state_load_fnm.empty())    // important in case of loading selection without learning
            p_learner->Train(p_selector->selected_cfgs);
    }

    // Configuring error accumulation
    if (monitor_errs = (!errmon_log.empty()))
    {
//         errmon.SetLogStream(errmon_log);
        
        if (p_lotf != nullptr)
            p_lotf->collect_abinitio_cfgs = true;
    }

    // checking and cleaning output configurations file (it is better to crash with error in the beginning)
    if (!cfgs_fnm.empty())
    {
        ofstream ofs_tmp(cfgs_fnm);
        if (!ofs_tmp.is_open())
            ERROR("Can't open .cfgs file \"" + cfgs_fnm + "\" for output");
    }

//     SetLogStream(log_output);
    // Cheking Ab-initio potential
    if (p_abinitio == nullptr &&
        (monitor_errs || enable_learn || (!enable_EFScalc && !efs_ignored)))
        ERROR("Ab-initio model is not specified but required");
}

MLIP_Wrapper::MLIP_Wrapper(const Settings& settings)
{
    InitSettings();
    ApplySettings(settings);

    //Message("MLIP initialization");
    
    //PrintSettings();

#ifdef MLIP_MPI    // multiprocessor logging preparing
    int mpi_rank;
    int mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (!errmon_log.empty() &&
        errmon_log != "stdout" &&
        errmon_log != "stderr")
            errmon_log += "_" + to_string(mpi_rank);
    if (!fit_log.empty() &&
        fit_log != "stdout" &&
        fit_log != "stderr")
            fit_log    += "_" + to_string(mpi_rank);
    if (!slct_log.empty() &&
        slct_log != "stdout" &&
        slct_log != "stderr")
            slct_log += "_" + to_string(mpi_rank);
    if (!log_output.empty() &&
        log_output != "stdout" &&
        log_output != "stderr")
            log_output += "_" + to_string(mpi_rank);
    
    if (!slct_ts_fnm.empty())
        slct_ts_fnm += "_" + to_string(mpi_rank);
    if (!slct_state_save_fnm.empty())
        slct_state_save_fnm += "_" + to_string(mpi_rank);
#endif

    SetTagLogStream("mlip", log_output); // 
    if (enable_learn) SetTagLogStream("fit", fit_log); // fit_log
    if (enable_select) SetTagLogStream("select", slct_log); // slct_log
    SetTagLogStream("check_errors", errmon_log); // slct_log

    // Initialize abinitio, temporary code
    const string abinitio_type = settings["abinitio"];
    if (abinitio_type == "null" || abinitio_type == "") {
//        Message("No abinitio model is set");
        p_abinitio = nullptr;
    }
    else if (abinitio_type == "void" || abinitio_type == "") {
        Message("abinitio EFS data is provided by the driver");
        p_abinitio = new VoidPotential();
    }
    else if (abinitio_type == "lj") {
        Message("abinitio model: Lennard-Jones pair potential");
        p_abinitio = new LJ(settings.ExtractSubSettings("abinitio"));
    }
    else if (abinitio_type == "vasp") {
        Message("abinitio model: DFT by VASP");
        p_abinitio = new VASP_potential(settings.ExtractSubSettings("abinitio"));
    }
    else if (abinitio_type == "lammps") {
        Message("abinitio model: potential through LAMMPS (defined in launch script)");
        p_abinitio = new LAMMPS_potential(settings.ExtractSubSettings("abinitio"));
    }
    else if (abinitio_type == "mtp") {
        Message("abinitio model: pre-learned MTP");
        p_abinitio = new MTP(settings.ExtractSubSettings("mtp"));
    }
    else ERROR("Inapropriate option for abintio model is specified");

    SetUpMLIP();

    if (((p_mlip == nullptr) && (p_abinitio == nullptr)) ||
        (enable_learn && (p_abinitio == nullptr)))
        ERROR("Incompatible settings");

//    Message("MLIP initialization complete");
}

MLIP_Wrapper::~MLIP_Wrapper()
{
    std::stringstream logstrm1;
    logstrm1 << endl;    // to keep previous messages printed with '\r'
    MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");

    if (monitor_errs)
    {
        errmon.MPI_Synchronize();
#ifdef MLIP_MPI
        int mpi_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        if (mpi_rank==0)
#endif // MLIP_MPI
        {
         std::string report;
         errmon.GetReport(report);
         MLP_LOG("check_errors",report);
        }
    }
    if (enable_select)
    {
        try
        {
#ifdef MLIP_MPI
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if (mpi_rank==0)
#endif // MLIP_MPI
            {
                if (!slct_ts_fnm.empty() && slct_trsh_break <= 0.0)    // (slct_trsh_break > 0.0) is prevention saving TS if selection mode activated
                    p_selector->SaveSelected(slct_ts_fnm);
                if (!slct_state_save_fnm.empty())
                    p_selector->Save(slct_state_save_fnm);
            }
        }
        catch (MlipException& excp)
        {
            Message(excp.What());
        }
    }

    if (enable_learn)
    {
        if (!enable_EFScalc)
            if (enable_select)    // EFS calculation with abinitio potential before training
            {
                if (efs_ignored)
                {
                    logstrm1 << "MLIP: calculating EFS for selected configurations" << std::endl;
                    MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
                    int cntr=0;
                    for (Configuration& cfg : p_selector->selected_cfgs)
                    {
                        p_abinitio->CalcEFS(cfg);
                        logstrm1 << "MLIP: " << ++cntr << " processed\r";
                        MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
//                         if (GetLogStream() != nullptr) GetLogStream()->flush();
                    }
                }
                p_learner->Train(p_selector->selected_cfgs);
            }
            else
                p_learner->Train(training_set);
        //else {}        // Training have already to be done in the other case

        try
        {
#ifdef MLIP_MPI
            int mpi_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            if (mpi_rank==0)
#endif
            {
                if (!mlip_fitted_fnm.empty())
                    p_mlip->Save(mlip_fitted_fnm);
            }
        }
        catch (MlipException& excp)
        {
            Message(excp.What());
        }
    }

    if (p_selector != nullptr)
        delete p_selector;

    if (p_learner != nullptr)
        delete p_learner;

    if (p_mlip != nullptr)
        delete p_mlip;

    // Message("MLIP object has been destroyed");
}

void MLIP_Wrapper::CalcEFS(Configuration & cfg)
{
    cfg_valid = cfg; // This placed before CalcEFS(cfg) for correct work with VoidPotential

    // Consider all possible combinations of enable_EFScalc, enable_learn, enable_select flags
    if (!enable_EFScalc && !enable_learn && !enable_select)
    {
        p_abinitio->CalcEFS(cfg);
        abinitio_count++;

        if (monitor_errs)
            cfg_valid = cfg;
    }
    else if (enable_EFScalc && !enable_learn && !enable_select)        // just EFS calculation mode
    {
        p_mlip->CalcEFS(cfg);
        MTP_count++;

        if (monitor_errs)
        {
            p_abinitio->CalcEFS(cfg_valid);
            abinitio_count++;
        }
    }
    //We have to remember about it)))
    else if (!enable_EFScalc && enable_learn && !enable_select)    // Learn all configuration mode
    {
        p_abinitio->CalcEFS(cfg);
        abinitio_count++;
        training_set.push_back(cfg);
        learn_count++;

        if (monitor_errs)
            cfg_valid = cfg;
    }
    else if (!enable_EFScalc && !enable_learn && enable_select)    // Select MV-set mode
    {
        if (!efs_ignored || monitor_errs)
        {
            p_abinitio->CalcEFS(cfg);
            abinitio_count++;
        }
        if (monitor_errs)
            cfg_valid = cfg;

        p_selector->Select(cfg);
    }
    else if (enable_EFScalc && enable_learn && !enable_select)    // Potential retrains and calculate EFS after adding each configuration (quite strange regime)
    {
        p_abinitio->CalcEFS(cfg);
        abinitio_count++;
        if (monitor_errs)
            cfg_valid = cfg;

        training_set.push_back(cfg);
        p_learner->Train(training_set);
        learn_count++;

        p_mlip->CalcEFS(cfg);
        MTP_count++;
    }
    else if (enable_EFScalc && !enable_learn && enable_select)    // EFS calculation combined with selection in one run
    {
        p_mlip->CalcEFS(cfg);
        MTP_count++;
        p_selector->Select(cfg,0);

        if (monitor_errs)
        {
            p_abinitio->CalcEFS(cfg_valid);
            abinitio_count++;
        }
    }
    else if (!enable_EFScalc && enable_learn && enable_select)    // Configurations are selecting while run, selected to be learned in destructor. EFS are calculated with Ab-initio only for selected configurations 
    {
        if (!efs_ignored || monitor_errs)
        {
            p_abinitio->CalcEFS(cfg);
            abinitio_count++;
        }
        if (monitor_errs)
            cfg_valid = cfg;

        p_selector->Select(cfg);
    }
    else if (enable_EFScalc && enable_learn && enable_select)    // LOTF scenario
    {
        int abinitio_clcs = p_lotf->pot_calcs_count;
        p_lotf->CalcEFS(cfg);

        abinitio_count = learn_count = p_lotf->pot_calcs_count;
        MTP_count = p_lotf->MTP_calcs_count;

        if (monitor_errs)
            if (abinitio_clcs < p_lotf->pot_calcs_count) // abinitio EFS happened
            {// No additional abinitio calcs. Take EFS for cfg_valid from accumulator
                cfg_valid = p_lotf->abinitio_cfgs.back();
                p_lotf->abinitio_cfgs.clear();
            }
            else
            {
                p_abinitio->CalcEFS(cfg_valid);
                abinitio_count = call_count+1;
            }
    }

    if (monitor_errs)
        errmon.collect(cfg_valid, cfg);

    if (!cfgs_fnm.empty() && (call_count % (skip_saving_N_cfgs+1) == 0))
        cfg.AppendToFile(cfgs_fnm);
    
    call_count++;
    std::stringstream logstrm1;
    logstrm1 << "MLIP: processed "    << call_count 
//            << "\tAbinitio_calcs "    << abinitio_count
            << "\tMTP_calcs "        << MTP_count
//            << "\tcfg_learned "        << learn_count
            << endl;
    MLP_LOG(tagname,logstrm1.str());

    if (enable_select && slct_trsh_break > 1.0)    // "Selection by generation" mode is activated 
    {
        if (cfg.features.count("MV_grade") != 1 || cfg.features.at("MV_grade") == "")
            MlipException("Inappropriate MV_grade detected");

        double grade = stod(cfg.features["MV_grade"]);
        cfg_valid.features["MV_grade"] = cfg.features["MV_grade"];

        if (grade > slct_trsh_slct)        // saving configuration
            cfg_valid.AppendToFile(slct_ts_fnm);

        if (grade > slct_trsh_break)    // break if necessary
            throw MlipException("Breaking threshold exceeded (MV-grade: " + cfg.features["MV_grade"] + ")");
    }

}

void MLIP_Wrapper::CalcE(Configuration & cfg)
{
    bool EFSfromAbInitio = false;

    if (enable_EFScalc || enable_learn || enable_select)
    {
        if (monitor_errs)    // This placed before CalcEFS(cfg) for correct work with VoidPotential
            cfg_valid = cfg;

        // Consider all possible combinations of enable_EFScalc, enable_learn, enable_select
        if (enable_EFScalc && !enable_learn && !enable_select)        // just E calculation mode
        {
            p_mlip->CalcE(cfg);
        }
        else if (!enable_EFScalc && enable_learn && !enable_select)    // Learn all configuration mode
        {
            if (p_abinitio != nullptr)
                p_abinitio->CalcEFS(cfg);
            else
                ERROR("Attempting learning on configurations with no EFS");

            training_set.push_back(cfg);
        }
        else if (!enable_EFScalc && !enable_learn && enable_select)    // Select MV-set mode
        {
            p_selector->Select(cfg);
        }
        else if (enable_EFScalc && enable_learn && !enable_select)    // Potential retrains and calculate EFS after adding each configuration
        {
            if (p_abinitio != nullptr)
                p_abinitio->CalcEFS(cfg);
            else
                ERROR("Attempting learning on configurations with no EFS");

            training_set.push_back(cfg);
            p_learner->Train(training_set);

            p_mlip->CalcE(cfg);
        }
        else if (enable_EFScalc && !enable_learn && enable_select)    // EFS calculation combined with selection in one run
        {
            p_mlip->CalcE(cfg);
            p_selector->Select(cfg);
        }
        else if (!enable_EFScalc && enable_learn && enable_select)    // Configurations are selecting while run, selected to be leaned in destructor. EFS are calculated with Ab-initio only for selected configurations 
        {
            p_selector->Select(cfg);
        }
        else if (enable_EFScalc && enable_learn && enable_select)    // LOTF scenario
            p_lotf->CalcEFS(cfg);

        if (monitor_errs)
        {
            if (!EFSfromAbInitio)
            {
                p_abinitio->CalcEFS(cfg_valid);
                errmon.collect(cfg_valid, cfg);
            }
            else
                errmon.collect(cfg, cfg);
        }
    }
    else if (p_abinitio != nullptr)
        p_abinitio->CalcEFS(cfg);

    if (!cfgs_fnm.empty() && (call_count % (skip_saving_N_cfgs+1) == 0))
        cfg.AppendToFile(cfgs_fnm);

  std::stringstream logstrm1;
    logstrm1 << "MLIP: processed " << ++call_count << " configurations" << endl;
  MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
}

